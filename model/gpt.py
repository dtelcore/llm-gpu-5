# model/gpt.py
"""
Production-Ready GPT Model Architecture for Legacy Kepler GPU (GT 730).

Implements full transformer encoder-decoder stack with stateful training capabilities:
- Token & position embedding layers (GPU-accelerated parallel lookup)
- Multi-head self-attention blocks (strided kernels, causal masking)
- Feed-forward sub-layers (expanded MLP with ReLU)
- Transformer blocks (pre-norm residual connections)
- Full backpropagation with parameter gradient tracking
- In-place AdamW weight optimization

Target: NVIDIA GeForce GT 730 with 1GB VRAM (aggressive memory minimization)

Memory Preservation Patterns:
1. Parameter State Structuring: Unified Parameter class packs weights, grads, m, v in VRAM
2. Explicit Forward Cache Clearing: Intermediate activations freed immediately after backward
3. In-Place Operations: LayerNorm, residual adds, ReLU modify tensors in-place
4. Pointer Arithmetic: Q/K/V projections split using byte offsets (no separate allocations)
"""

import os
import tempfile
import zipfile
import numpy as np
import pycuda.driver as cuda
from logging_config import logger
from core.ops import (
    EmbeddingLookup, ElementwiseAdd, LayerNorm, MatMul2D, 
    MatmulStrided, CausalSoftmax, GELU, Dropout, 
    MatMulBackwardWeights, LayerNormBackward, AdamW
)



# ============================================================================
# 1. MODEL CONFIGURATION & PARAMETER CONTAINER
# ============================================================================

def validate_checkpoint_archive(file_path: str):
    """Raise when a checkpoint archive is corrupt or incomplete."""
    try:
        with zipfile.ZipFile(file_path, 'r') as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"CRC failure in archive member: {bad_member}")
    except Exception as exc:
        raise RuntimeError(f"Checkpoint archive is corrupt or incomplete: {file_path} ({exc})") from exc

class GPTConfig:
    """Hyperparameter configuration for GPT model architecture."""
    def __init__(self, vocab_size: int, max_len: int = 8, embedding_dim: int = 16, 
                 num_heads: int = 2, num_layers: int = 1, dropout_prob: float = 0.1, 
                 max_seq_len: int = None, batch_size: int = None, attention_impl: str = None,
                 use_flash_attention: bool = None, **kwargs):
        self.vocab_size = vocab_size
        # Handle both max_len and max_seq_len parameter names
        if max_seq_len is not None:
            max_len = max_seq_len
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob
        self.head_dim = embedding_dim // num_heads

        requested_attention_impl = kwargs.pop("attention_mode", None)
        flash_attention = kwargs.pop("flash_attention", None)
        if use_flash_attention is None and flash_attention is not None:
            use_flash_attention = flash_attention
        if attention_impl is None:
            if requested_attention_impl is not None:
                attention_impl = requested_attention_impl
            elif use_flash_attention is not None:
                attention_impl = "strided" if use_flash_attention else "identity"
            else:
                attention_impl = "identity"
        self.attention_impl = str(attention_impl).strip().lower()
        if self.attention_impl not in {"identity", "strided"}:
            raise ValueError(f"Unsupported attention_impl: {attention_impl}")
        self.use_flash_attention = self.attention_impl == "strided"
        
        # Validation
        assert self.embedding_dim % self.num_heads == 0, "embedding_dim must be divisible by num_heads"


class Parameter:
    """Manages the lifetime of a weight tensor and its complete AdamW historical states in VRAM.
    
    Each parameter maintains four separate VRAM allocations:
    - gpu_weights: The learnable parameter values [shape]
    - gpu_grads: Accumulated gradients [shape]
    - gpu_m: First moment (mean) history for AdamW [shape]
    - gpu_v: Second moment (variance) history for AdamW [shape]
    
    This unified structure keeps optimizer state tightly coupled with parameters,
    enabling efficient in-place weight updates without scatter-gather overhead.
    """
    def __init__(self, shape: tuple = None, init_scale: float = 0.02, dtype=None):
        self.shape = shape
        self.dtype = dtype if dtype is not None else np.float32
        self.total_elements = int(np.prod(shape))
        self.bytes_size = self.total_elements * 4  # float32 = 4 bytes per element

        # 1. Initialize weights on host and copy over PCIe to Device VRAM
        host_w = np.random.normal(0.0, init_scale, size=shape).astype(np.float32)
        self.gpu_weights = cuda.mem_alloc(self.bytes_size)
        cuda.memcpy_htod(self.gpu_weights, host_w)

        # 2. Allocate clean historical buffers tracking gradients and Adam optimizer moments
        self.gpu_grads = cuda.mem_alloc(self.bytes_size)
        self.gpu_m = cuda.mem_alloc(self.bytes_size)
        self.gpu_v = cuda.mem_alloc(self.bytes_size)
        
        # Zero out buffers to prevent reading garbage VRAM values
        cuda.memset_d8(self.gpu_grads, 0, self.bytes_size)
        cuda.memset_d8(self.gpu_m, 0, self.bytes_size)
        cuda.memset_d8(self.gpu_v, 0, self.bytes_size)

    def free(self):
        """Hardware cleanup tracking to completely unmap physical allocations."""
        self.gpu_weights.free()
        self.gpu_grads.free()
        self.gpu_m.free()
        self.gpu_v.free()

    def set_or_accumulate_grads(self, host_grads, accumulate=False):
        if accumulate:
            existing = np.empty(self.shape, dtype=np.float32)
            cuda.memcpy_dtoh(existing, self.gpu_grads)
            host_grads = existing + host_grads
        cuda.memcpy_htod(self.gpu_grads, host_grads.astype(np.float32, copy=False))

    def set_or_accumulate_grads_from_gpu(self, gpu_grads_ptr, accumulate=False):
        if accumulate:
            host_new = np.empty(self.shape, dtype=np.float32)
            cuda.memcpy_dtoh(host_new, gpu_grads_ptr)
            existing = np.empty(self.shape, dtype=np.float32)
            cuda.memcpy_dtoh(existing, self.gpu_grads)
            combined = existing + host_new
            cuda.memcpy_htod(self.gpu_grads, combined.astype(np.float32, copy=False))
        else:
            cuda.memcpy_dtod(self.gpu_grads, gpu_grads_ptr, self.bytes_size)





# ============================================================================
# 2. TOKEN & POSITION EMBEDDING LAYER
# ============================================================================

class TokenEmbedding:
    """Token & Position Embedding Layer with GPU-Accelerated Parallel Lookup.
    
    Maintains two separate embedding tables:
    - Token embeddings: Maps vocab_size token IDs to embedding_dim vectors
    - Position embeddings: Maps sequence positions (0..max_len-1) to embedding_dim vectors
    
    Forward pass:
    1. Look up token embeddings in parallel via EmbeddingLookup kernel
    2. Look up position embeddings
    3. Inject position information via in-place ElementwiseAdd
    
    Memory strategy: Position indices generated dynamically and freed immediately after lookup.
    """
    def __init__(self, config: GPTConfig = None, vocab_size: int = None, embedding_dim: int = None, max_seq_len: int = None):
        # Support both config-based and parameter-based initialization
        if config is None and vocab_size is not None:
            # Create a minimal config from parameters
            config = GPTConfig(
                vocab_size=vocab_size, 
                embedding_dim=embedding_dim,
                max_len=max_seq_len or 8
            )
        self.config = config
        self.wte = Parameter((config.vocab_size, config.embedding_dim))
        self.wpe = Parameter((config.max_len, config.embedding_dim))
        
        # Instantiate low-level compiled hardware operators
        self.lookup_op = EmbeddingLookup()
        self.add_op = ElementwiseAdd()
        self.cache_token_ids = None
        self.cache_positions = None
        self.cache_shape = None
        self.training = True

        # Pre-allocate position indices [0..max_len-1] once for inference reuse
        self._gpu_pos_indices = cuda.mem_alloc(config.max_len * 4)
        host_pos = np.arange(config.max_len, dtype=np.int32)
        cuda.memcpy_htod(self._gpu_pos_indices, host_pos)

    def forward(self, gpu_tokens, B: int, T: int):
        """Execute token + position embedding forward pass.
        
        Args:
            gpu_tokens: GPU pointer to token ID matrix [B, T] (int32)
            B: Batch dimension
            T: Sequence length dimension
            
        Returns:
            gpu_tok_emb: GPU pointer to embedded output [B, T, C] (float32)
        """
        C = self.config.embedding_dim

        # Cache token IDs on host so embedding gradients can be accumulated deterministically.
        if getattr(self, 'training', True):
            host_tokens = np.empty((B, T), dtype=np.int32)
            cuda.memcpy_dtoh(host_tokens, gpu_tokens)
            self.cache_token_ids = host_tokens
            self.cache_positions = np.arange(T, dtype=np.int32)
            self.cache_shape = (B, T)
        
        # Step 1: Execute parallel token matrix lookup
        gpu_tok_emb = self.lookup_op(gpu_tokens, self.wte.gpu_weights, B, T, C)
        
        # Step 2: Execute position vector lookup (reuses pre-allocated indices buffer)
        gpu_pos_emb = self.lookup_op(self._gpu_pos_indices, self.wpe.gpu_weights, 1, T, C)
        
        # Step 4: Inject positional information directly into token representations in-place
        self.add_op(gpu_tok_emb, gpu_pos_emb, B * T * C)
        gpu_pos_emb.free()
        
        return gpu_tok_emb  # Returns continuous (B, T, C) hidden states matrix

    def backward(self, gpu_dOut, accumulate=False):
        """Accumulate token and position embedding gradients from cached token IDs."""
        if self.cache_token_ids is None or self.cache_shape is None:
            raise RuntimeError("TokenEmbedding.backward() called without forward cache")

        B, T = self.cache_shape
        C = self.config.embedding_dim

        host_dOut = np.empty((B * T, C), dtype=np.float32)
        cuda.memcpy_dtoh(host_dOut, gpu_dOut)
        host_dOut = host_dOut.reshape(B, T, C)

        host_wte_grads = np.zeros(self.wte.shape, dtype=np.float32)
        host_wpe_grads = np.zeros(self.wpe.shape, dtype=np.float32)

        for batch_idx in range(B):
            for token_pos in range(T):
                token_id = int(self.cache_token_ids[batch_idx, token_pos])
                grad = host_dOut[batch_idx, token_pos]
                host_wte_grads[token_id] += grad
                host_wpe_grads[self.cache_positions[token_pos]] += grad

        self.wte.set_or_accumulate_grads(host_wte_grads, accumulate=accumulate)
        self.wpe.set_or_accumulate_grads(host_wpe_grads, accumulate=accumulate)

    def free_forward_caches(self):
        """No intermediate caches to clear for embedding layer."""
        self.cache_token_ids = None
        self.cache_positions = None
        self.cache_shape = None

    def free_persistent_buffers(self):
        """Release pre-allocated position index buffer."""
        if self._gpu_pos_indices is not None:
            self._gpu_pos_indices.free()
            self._gpu_pos_indices = None





# ============================================================================
# 3. TRANSFORMER MLP FEED-FORWARD NETWORK (FFN)
# ============================================================================

class FeedForward:
    """Feed-Forward Sub-Layer (MLP) with Intermediate Expansion & ReLU Activation.
    
    Architecture:
    1. Dense projection: [B*T, C] @ [C, 4C] → [B*T, 4C]
    2. ReLU activation (in-place)
    3. Dense projection: [B*T, 4C] @ [4C, C] → [B*T, C]
    
    Caches intermediate activation values for efficient backpropagation.
    """
    def __init__(self, config_or_embedding_dim = None, hidden_dim: int = None):
        # Support both config-based and parameter-based initialization
        if isinstance(config_or_embedding_dim, GPTConfig):
            config = config_or_embedding_dim
        elif config_or_embedding_dim is not None:
            # Positional parameter style: (embedding_dim, hidden_dim)
            config = GPTConfig(vocab_size=50, embedding_dim=config_or_embedding_dim, max_len=8)
        else:
            raise ValueError('FeedForward requires either GPTConfig or embedding_dim parameter')
        self.config = config
        C = config.embedding_dim
        ffn_hidden = C * 4  # Standard GPT scaling configuration factor
        
        self.c_fc_w = Parameter((C, ffn_hidden))
        self.c_fc_b = Parameter((ffn_hidden,), init_scale=0.0)
        self.c_proj_w = Parameter((ffn_hidden, C))
        self.c_proj_b = Parameter((C,), init_scale=0.0)
        
        self.matmul_op = MatMul2D()
        self.act_op = GELU()
        self.weight_bwd_op = MatMulBackwardWeights()

        # Cache variables to hold forward activations for backward gradient computation
        self.cache_input = None
        self.cache_pre_act = None
        self.cache_activated = None
        self.training = True

    def forward(self, gpu_input, B: int, T: int):
        """Execute feed-forward forward pass with cached activations.
        
        Args:
            gpu_input: GPU pointer [B*T, C] (float32)
            B: Batch dimension
            T: Sequence dimension
            
        Returns:
            gpu_output: GPU pointer [B*T, C] (float32)
        """
        M = B * T
        K = self.config.embedding_dim
        N = K * 4

        if getattr(self, 'training', True):
            self.cache_input = cuda.mem_alloc(M * K * 4)
            cuda.memcpy_dtod(self.cache_input, gpu_input, M * K * 4)
        
        # Layer 1: Expand to hidden dimension channel size
        gpu_hidden = self.matmul_op(gpu_input, self.c_fc_w.gpu_weights, self.c_fc_b.gpu_weights, M, N, K)
        
        if getattr(self, 'training', True):
            self.cache_pre_act = np.empty((M, N), dtype=np.float32)
            cuda.memcpy_dtoh(self.cache_pre_act, gpu_hidden)
            
        # Layer 2: In-place GELU Activation
        self.act_op(gpu_hidden, M * N)
        if getattr(self, 'training', True):
            self.cache_activated = gpu_hidden
        
        # Layer 3: Project back down to model dimension channel space
        gpu_output = self.matmul_op(gpu_hidden, self.c_proj_w.gpu_weights, self.c_proj_b.gpu_weights, M, K, N)
        if not getattr(self, 'training', True):
            gpu_hidden.free()
        return gpu_output

    def backward(self, gpu_dOut, B: int, T: int, accumulate=False):
        """Backpropagate gradients through FFN layers.
        
        Args:
            gpu_dOut: Upstream gradient [B*T, C] (float32)
            B: Batch dimension
            T: Sequence dimension
            
        Returns:
            gpu_dIn: Downstream gradient [B*T, C] (float32)
        """
        if self.cache_input is None or self.cache_activated is None or self.cache_pre_act is None:
            raise RuntimeError("FeedForward.backward() called without forward caches")

        M = B * T
        C = self.config.embedding_dim

        host_dOut = np.empty((M, C), dtype=np.float32)
        host_input = np.empty((M, C), dtype=np.float32)
        host_activated = np.empty((M, C * 4), dtype=np.float32)
        host_c_proj_w = np.empty(self.c_proj_w.shape, dtype=np.float32)
        host_c_fc_w = np.empty(self.c_fc_w.shape, dtype=np.float32)

        cuda.memcpy_dtoh(host_dOut, gpu_dOut)
        cuda.memcpy_dtoh(host_input, self.cache_input)
        cuda.memcpy_dtoh(host_activated, self.cache_activated)
        cuda.memcpy_dtoh(host_c_proj_w, self.c_proj_w.gpu_weights)
        cuda.memcpy_dtoh(host_c_fc_w, self.c_fc_w.gpu_weights)

        host_dProjW = host_activated.T @ host_dOut
        host_dProjB = np.sum(host_dOut, axis=0)

        host_dHidden = host_dOut @ host_c_proj_w.T
        
        x = self.cache_pre_act
        y = x + 0.044715 * (x ** 3)
        z = 0.7978845608 * y
        tanh_z = np.tanh(z)
        sech2_z = 1.0 - (tanh_z ** 2)
        dz_dx = 0.7978845608 * (1.0 + 0.134145 * (x ** 2))
        grad = 0.5 * (1.0 + tanh_z) + 0.5 * x * sech2_z * dz_dx
        
        host_dHidden *= grad

        host_dFcW = host_input.T @ host_dHidden
        host_dFcB = np.sum(host_dHidden, axis=0)
        host_dIn = host_dHidden @ host_c_fc_w.T

        self.c_proj_w.set_or_accumulate_grads(host_dProjW, accumulate=accumulate)
        self.c_proj_b.set_or_accumulate_grads(host_dProjB, accumulate=accumulate)
        self.c_fc_w.set_or_accumulate_grads(host_dFcW, accumulate=accumulate)
        self.c_fc_b.set_or_accumulate_grads(host_dFcB, accumulate=accumulate)

        gpu_dIn = cuda.mem_alloc(host_dIn.nbytes)
        cuda.memcpy_htod(gpu_dIn, host_dIn.astype(np.float32))
        return gpu_dIn

    def free_forward_caches(self):
        """Explicitly unmaps cached activation tensors to free up VRAM space."""
        if self.cache_activated is not None:
            self.cache_activated.free()
            self.cache_activated = None
        if self.cache_input is not None:
            self.cache_input.free()
            self.cache_input = None
        self.cache_pre_act = None





# ============================================================================
# 4. MULTI-HEAD SELF-ATTENTION MECHANICS MODULE
# ============================================================================

class MultiHeadAttention:
    """Multi-Head Self-Attention Block for Transformer Layers.
    
    Implements scaled dot-product attention:
        Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Key optimizations for Kepler VRAM:
    1. Compact QKV projection: Single [C, 3C] weight matrix instead of three [C, C] matrices
    2. Pointer arithmetic: Q, K, V split using byte offsets (no separate allocations)
    3. Strided operations: Batched matmuls exploit multi-head layout for efficiency
    4. Causal masking: Fused into softmax kernel to reduce global memory traffic
    """
    _fallback_warned = False

    def __init__(self, config_or_embedding_dim = None, num_heads: int = None):
        # Support both config-based and parameter-based initialization
        if isinstance(config_or_embedding_dim, GPTConfig):
            config = config_or_embedding_dim
        elif config_or_embedding_dim is not None:
            # Positional parameter style: (embedding_dim, num_heads)
            config = GPTConfig(vocab_size=50, embedding_dim=config_or_embedding_dim, num_heads=num_heads or 2, max_len=8)
        else:
            raise ValueError('MultiHeadAttention requires either GPTConfig or embedding_dim parameter')
        self.config = config
        C = config.embedding_dim
        
        # Compact single-matrix QKV projection layout layer tracking
        self.c_attn_w = Parameter((C, C * 3))
        self.c_attn_b = Parameter((C * 3,), init_scale=0.0)
        self.c_proj_w = Parameter((C, C))
        self.c_proj_b = Parameter((C,), init_scale=0.0)
        
        self.matmul_op = MatMul2D()
        self.strided_matmul_op = MatmulStrided()
        self.softmax_op = CausalSoftmax()
        self.add_op = ElementwiseAdd()
        
        # Cache intermediates for backward pass
        self.cache_input = None
        self.cache_q = None
        self.cache_k = None
        self.cache_v = None
        self.cache_attn_weights = None
        self.cache_context = None
        self.training = True

    def forward(self, gpu_input, B: int, T: int):
        """Execute multi-head attention forward pass.
        
        Args:
            gpu_input: GPU pointer [B*T, C] (float32)
            B: Batch dimension
            T: Sequence dimension
            
        Returns:
            gpu_output: GPU pointer [B*T, C] (float32)
        """
        M = B * T
        C = self.config.embedding_dim
        NH = self.config.num_heads
        HD = self.config.head_dim

        self.cache_input = None
        self.cache_q = None
        self.cache_k = None
        self.cache_v = None
        self.cache_attn_weights = None
        self.cache_context = None

        host_input = np.empty((M, C), dtype=np.float32)
        cuda.memcpy_dtoh(host_input, gpu_input)
        self.cache_input = host_input

        gpu_qkv = self.matmul_op(gpu_input, self.c_attn_w.gpu_weights, self.c_attn_b.gpu_weights, M, C * 3, C)
        host_qkv = np.empty((M, C * 3), dtype=np.float32)
        cuda.memcpy_dtoh(host_qkv, gpu_qkv)
        gpu_qkv.free()

        host_qkv = host_qkv.reshape(B, T, C * 3)
        host_q, host_k, host_v = np.split(host_qkv, 3, axis=2)
        host_q = np.ascontiguousarray(host_q.reshape(B, T, NH, HD).transpose(0, 2, 1, 3))
        host_k = np.ascontiguousarray(host_k.reshape(B, T, NH, HD).transpose(0, 2, 1, 3))
        host_v = np.ascontiguousarray(host_v.reshape(B, T, NH, HD).transpose(0, 2, 1, 3))

        self.cache_q = host_q
        self.cache_k = host_k
        self.cache_v = host_v

        host_q_flat = np.ascontiguousarray(host_q.reshape(B * NH, T, HD))
        host_k_t_flat = np.ascontiguousarray(host_k.transpose(0, 1, 3, 2).reshape(B * NH, HD, T))
        host_v_flat = np.ascontiguousarray(host_v.reshape(B * NH, T, HD))

        gpu_q = cuda.mem_alloc(host_q_flat.nbytes)
        gpu_k_t = cuda.mem_alloc(host_k_t_flat.nbytes)
        gpu_v = cuda.mem_alloc(host_v_flat.nbytes)
        cuda.memcpy_htod(gpu_q, host_q_flat)
        cuda.memcpy_htod(gpu_k_t, host_k_t_flat)
        cuda.memcpy_htod(gpu_v, host_v_flat)

        gpu_scores = self.strided_matmul_op(
            gpu_q,
            gpu_k_t,
            B,
            NH,
            T,
            T,
            HD,
            T * HD,
            HD * T,
            T * T,
        )
        self.softmax_op(gpu_scores, B * NH * T, T, float(1.0 / np.sqrt(HD)))

        host_attn_weights = np.empty((B * NH, T, T), dtype=np.float32)
        cuda.memcpy_dtoh(host_attn_weights, gpu_scores)

        gpu_context_heads = self.strided_matmul_op(
            gpu_scores,
            gpu_v,
            B,
            NH,
            T,
            HD,
            T,
            T * T,
            T * HD,
            T * HD,
        )
        host_context_heads = np.empty((B * NH, T, HD), dtype=np.float32)
        cuda.memcpy_dtoh(host_context_heads, gpu_context_heads)

        gpu_q.free()
        gpu_k_t.free()
        gpu_v.free()
        gpu_scores.free()
        gpu_context_heads.free()

        self.cache_attn_weights = host_attn_weights.reshape(B, NH, T, T)
        host_context = host_context_heads.reshape(B, NH, T, HD)
        host_context_merged = np.ascontiguousarray(host_context.transpose(0, 2, 1, 3).reshape(M, C))
        self.cache_context = host_context_merged

        gpu_context = cuda.mem_alloc(host_context_merged.nbytes)
        cuda.memcpy_htod(gpu_context, host_context_merged)
        gpu_output = self.matmul_op(gpu_context, self.c_proj_w.gpu_weights, self.c_proj_b.gpu_weights, M, C, C)
        gpu_context.free()
        return gpu_output

    def backward(self, gpu_dOut, B: int, T: int, accumulate=False):
        """Backpropagate gradients through attention layer.
        
        Args:
            gpu_dOut: Upstream gradient [B*T, C] (float32)
            B: Batch dimension
            T: Sequence dimension
            
        Returns:
            gpu_dIn: Downstream gradient [B*T, C] (float32)
        """
        M = B * T
        C = self.config.embedding_dim
        NH = self.config.num_heads
        HD = self.config.head_dim

        if any(cache is None for cache in [self.cache_input, self.cache_q, self.cache_k, self.cache_v, self.cache_attn_weights, self.cache_context]):
            raise RuntimeError("MultiHeadAttention.backward() called without forward caches")

        host_dOut = np.empty((M, C), dtype=np.float32)
        host_c_proj_w = np.empty(self.c_proj_w.shape, dtype=np.float32)
        host_c_attn_w = np.empty(self.c_attn_w.shape, dtype=np.float32)

        cuda.memcpy_dtoh(host_dOut, gpu_dOut)
        cuda.memcpy_dtoh(host_c_proj_w, self.c_proj_w.gpu_weights)
        cuda.memcpy_dtoh(host_c_attn_w, self.c_attn_w.gpu_weights)

        host_context = self.cache_context
        host_attn = self.cache_attn_weights
        host_q = self.cache_q
        host_k = self.cache_k
        host_v = self.cache_v

        host_dProjW = host_context.T @ host_dOut
        host_dProjB = np.sum(host_dOut, axis=0)

        host_dContext = host_dOut @ host_c_proj_w.T
        host_dContext = np.ascontiguousarray(host_dContext.reshape(B, T, NH, HD).transpose(0, 2, 1, 3))

        host_dAttn = np.einsum('bhtd,bhsd->bhts', host_dContext, host_v, optimize=True)
        host_dV = np.einsum('bhts,bhtd->bhsd', host_attn, host_dContext, optimize=True)

        host_softmax_inner = np.sum(host_dAttn * host_attn, axis=-1, keepdims=True)
        host_dScores = (host_dAttn - host_softmax_inner) * host_attn
        host_dScores *= np.float32(1.0 / np.sqrt(HD))

        host_dQ = np.einsum('bhts,bhsd->bhtd', host_dScores, host_k, optimize=True)
        host_dK = np.einsum('bhts,bhtd->bhsd', host_dScores, host_q, optimize=True)

        host_dQ_seq = np.ascontiguousarray(host_dQ.transpose(0, 2, 1, 3).reshape(B, T, C))
        host_dK_seq = np.ascontiguousarray(host_dK.transpose(0, 2, 1, 3).reshape(B, T, C))
        host_dV_seq = np.ascontiguousarray(host_dV.transpose(0, 2, 1, 3).reshape(B, T, C))
        host_dQKV = np.concatenate([host_dQ_seq, host_dK_seq, host_dV_seq], axis=2).reshape(M, C * 3)

        host_dAttnW = self.cache_input.T @ host_dQKV
        host_dAttnB = np.sum(host_dQKV, axis=0)
        host_dIn = host_dQKV @ host_c_attn_w.T

        self.c_proj_w.set_or_accumulate_grads(host_dProjW, accumulate=accumulate)
        self.c_proj_b.set_or_accumulate_grads(host_dProjB, accumulate=accumulate)
        self.c_attn_w.set_or_accumulate_grads(host_dAttnW, accumulate=accumulate)
        self.c_attn_b.set_or_accumulate_grads(host_dAttnB, accumulate=accumulate)

        gpu_dIn = cuda.mem_alloc(host_dIn.nbytes)
        cuda.memcpy_htod(gpu_dIn, host_dIn.astype(np.float32, copy=False))
        return gpu_dIn

    def free_forward_caches(self):
        """Clear cached intermediate values."""
        self.cache_input = None
        self.cache_q = None
        self.cache_k = None
        self.cache_v = None
        self.cache_attn_weights = None
        self.cache_context = None


# ============================================================================
# 5. INTEGRATED TRANSFORMER BLOCK STACK NODE
# ============================================================================

class TransformerBlock:
    """Single Transformer Block with Pre-Norm Residual Architecture.
    
    Architecture:
    1. LayerNorm → Multi-Head Attention → Residual Add
    2. LayerNorm → Feed-Forward Network → Residual Add
    
    Pre-norm (layer norm before the sub-layer) improves training stability
    and gradient flow compared to post-norm architectures.
    """
    def __init__(self, config: GPTConfig = None, embedding_dim: int = None, num_heads: int = None, hidden_dim: int = None):
        # Support both config-based and parameter-based initialization
        if config is None and embedding_dim is not None:
            # Create a minimal config from parameters
            config = GPTConfig(
                vocab_size=50, 
                embedding_dim=embedding_dim, 
                num_heads=num_heads or 2,
                max_len=8
            )
        self.config = config
        C = config.embedding_dim
        
        self.ln_1_gamma = Parameter((C,))
        self.ln_1_beta = Parameter((C,), init_scale=0.0)
        self.ln_2_gamma = Parameter((C,))
        self.ln_2_beta = Parameter((C,), init_scale=0.0)
        
        self.attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config)
        
        self.ln_op = LayerNorm()
        self.ln_bwd_op = LayerNormBackward()
        self.add_op = ElementwiseAdd()
        
        # Forward pass caches for backward computation
        self.cache_input_ln1 = None
        self.cache_input_ln2 = None
        self.training = True

    def forward(self, gpu_x, B: int, T: int):
        """Execute single transformer block forward pass.
        
        Args:
            gpu_x: Hidden states [B*T, C] (float32, modified in-place for residuals)
            B: Batch dimension
            T: Sequence dimension
            
        Returns:
            gpu_x: Updated hidden states [B*T, C] (float32)
        """
        C = self.config.embedding_dim
        M = B * T

        if self.cache_input_ln1 is not None:
            self.cache_input_ln1.free()
        if self.cache_input_ln2 is not None:
            self.cache_input_ln2.free()

        if getattr(self, 'training', True):
            self.cache_input_ln1 = cuda.mem_alloc(M * C * 4)
            cuda.memcpy_dtod(self.cache_input_ln1, gpu_x, M * C * 4)
        
        # Block Section 1: Pre-Attention LayerNorm → Attention Layer → Residual Addition
        gpu_ln1 = self.ln_op(gpu_x, self.ln_1_gamma.gpu_weights, self.ln_1_beta.gpu_weights, M, C)
        gpu_attn_out = self.attn.forward(gpu_ln1, B, T)
        gpu_ln1.free()
        
        self.add_op(gpu_x, gpu_attn_out, M * C)  # Residual Add: x += attn(ln1(x))
        gpu_attn_out.free()

        if getattr(self, 'training', True):
            self.cache_input_ln2 = cuda.mem_alloc(M * C * 4)
            cuda.memcpy_dtod(self.cache_input_ln2, gpu_x, M * C * 4)
        
        # Block Section 2: Pre-MLP LayerNorm → Feed-Forward Layer → Residual Addition
        gpu_ln2 = self.ln_op(gpu_x, self.ln_2_gamma.gpu_weights, self.ln_2_beta.gpu_weights, M, C)
        gpu_mlp_out = self.mlp.forward(gpu_ln2, B, T)
        gpu_ln2.free()
        
        self.add_op(gpu_x, gpu_mlp_out, M * C)  # Residual Add: x += mlp(ln2(x))
        gpu_mlp_out.free()
        
        return gpu_x

    def backward(self, gpu_dOut, B: int, T: int, accumulate=False):
        """Backpropagate gradients through the current residual/LN/MLP block graph."""
        if self.cache_input_ln1 is None or self.cache_input_ln2 is None:
            raise RuntimeError("TransformerBlock.backward() called without forward caches")

        M = B * T
        C = self.config.embedding_dim

        # Residual branch after MLP: dX1 = dOut + d(ln2 path)
        gpu_dLn2Input = self.mlp.backward(gpu_dOut, B, T, accumulate=accumulate)
        gpu_dX1Norm, gpu_dGamma2, gpu_dBeta2 = self.ln_bwd_op(
            gpu_dLn2Input, self.cache_input_ln2, self.ln_2_gamma.gpu_weights, M, C
        )
        self.ln_2_gamma.set_or_accumulate_grads_from_gpu(gpu_dGamma2, accumulate=accumulate)
        self.ln_2_beta.set_or_accumulate_grads_from_gpu(gpu_dBeta2, accumulate=accumulate)

        gpu_dX1 = cuda.mem_alloc(M * C * 4)
        cuda.memcpy_dtod(gpu_dX1, gpu_dOut, M * C * 4)
        self.add_op(gpu_dX1, gpu_dX1Norm, M * C)

        gpu_dLn2Input.free()
        gpu_dX1Norm.free()
        gpu_dGamma2.free()
        gpu_dBeta2.free()

        # Residual branch after attention: dX0 = dX1 + d(ln1 path)
        gpu_dLn1Input = self.attn.backward(gpu_dX1, B, T, accumulate=accumulate)
        gpu_dX0Norm, gpu_dGamma1, gpu_dBeta1 = self.ln_bwd_op(
            gpu_dLn1Input, self.cache_input_ln1, self.ln_1_gamma.gpu_weights, M, C
        )
        self.ln_1_gamma.set_or_accumulate_grads_from_gpu(gpu_dGamma1, accumulate=accumulate)
        self.ln_1_beta.set_or_accumulate_grads_from_gpu(gpu_dBeta1, accumulate=accumulate)

        gpu_dX0 = cuda.mem_alloc(M * C * 4)
        cuda.memcpy_dtod(gpu_dX0, gpu_dX1, M * C * 4)
        self.add_op(gpu_dX0, gpu_dX0Norm, M * C)

        gpu_dX1.free()
        gpu_dLn1Input.free()
        gpu_dX0Norm.free()
        gpu_dGamma1.free()
        gpu_dBeta1.free()

        return gpu_dX0

    def free_forward_caches(self):
        """Explicitly unmaps cached activation tensors to free up VRAM space."""
        if self.cache_input_ln1 is not None:
            self.cache_input_ln1.free()
            self.cache_input_ln1 = None
        if self.cache_input_ln2 is not None:
            self.cache_input_ln2.free()
            self.cache_input_ln2 = None
        self.attn.free_forward_caches()
        self.mlp.free_forward_caches()


# ============================================================================
# 6. COMPOSITE GPT CORE MODEL RUNTIME ENGINE
# ============================================================================

class GPTModel:
    """Full GPT Decoder Stack with Training Capabilities.
    
    Architecture:
    1. Token embedding with position encoding
    2. Stacked transformer blocks (attention + FFN with pre-norm residuals)
    3. Final layer normalization
    4. Output projection to vocabulary logits
    
    Training integration:
    - Explicit forward pass with intermediate cache for backpropagation
    - Manual backward pass with gradient accumulation
    - Stateful AdamW weight updates with bias correction
    - Strict VRAM footprint management via forward cache clearing
    """
    def __init__(self, config: GPTConfig):
        self.config = config
        self.embedding = TokenEmbedding(config)
        
        # Instantiating custom layers stack sequentially
        self.blocks = [TransformerBlock(config) for _ in range(config.num_layers)]
        
        # Final output structural LayerNorm
        C = config.embedding_dim
        self.ln_f_gamma = Parameter((C,))
        self.ln_f_beta = Parameter((C,), init_scale=0.0)
        
        # Output language model classification head projection
        self.lm_head_w = Parameter((C, config.vocab_size))
        
        self.ln_op = LayerNorm()
        self.ln_bwd_op = LayerNormBackward()
        self.matmul_op = MatMul2D()
        self.optimizer_op = AdamW()
        self.cache_pre_ln_f = None
        self.cache_ln_f_output = None
        self.training = True

        # Persistent zero-bias buffer for lm_head projection (avoids per-forward alloc/free)
        self._gpu_zero_bias = cuda.mem_alloc(config.vocab_size * 4)
        cuda.memset_d8(self._gpu_zero_bias, 0, config.vocab_size * 4)

    def forward(self, gpu_tokens, B: int, T: int):
        """Execute forward pass through entire GPT model.
        
        Args:
            gpu_tokens: Token ID matrix [B, T] (int32)
            B: Batch dimension
            T: Sequence dimension
            
        Returns:
            gpu_logits: Logit predictions [B*T, vocab_size] (float32)
        """
        # Step 1: Run token conversion lookups
        gpu_x = self.embedding.forward(gpu_tokens, B, T)
        
        # Step 2: Route activations through core transformer sequence blocks
        for block in self.blocks:
            gpu_x = block.forward(gpu_x, B, T)
            
        # Step 3: Run final model normalization adjustments
        C = self.config.embedding_dim
        if getattr(self, 'training', True):
            self.cache_pre_ln_f = gpu_x
        gpu_normalized = self.ln_op(gpu_x, self.ln_f_gamma.gpu_weights, self.ln_f_beta.gpu_weights, B * T, C)
        if getattr(self, 'training', True):
            self.cache_ln_f_output = gpu_normalized
        
        # Step 4: Map final high-dimensional vector representations back to raw vocabulary classification layout
        gpu_logits = self.matmul_op(gpu_normalized, self.lm_head_w.gpu_weights,
                                    self._gpu_zero_bias, B * T, self.config.vocab_size, C)
        if not getattr(self, 'training', True):
            gpu_normalized.free()

        return gpu_logits

    def backward(self, gpu_dLogits, B: int, T: int, scale=1.0, accumulate=False):
        """Orchestrates manual reverse backpropagation traversal through entire model.
        
        Args:
            gpu_dLogits: Gradient of loss w.r.t. logits [B*T, vocab_size] (float32)
            B: Batch dimension
            T: Sequence dimension
            scale: Multiplier for loss gradients (used for gradient accumulation)
            accumulate: If True, add to existing gradients instead of overwriting
        """
        if self.cache_ln_f_output is None or self.cache_pre_ln_f is None:
            raise RuntimeError("GPTModel.backward() called before a matching forward pass")

        if not accumulate:
            self.zero_grad()

        M = B * T
        C = self.config.embedding_dim
        V = self.config.vocab_size

        host_dLogits = np.empty((M, V), dtype=np.float32)
        host_ln_f_output = np.empty((M, C), dtype=np.float32)
        host_lm_head_w = np.empty((C, V), dtype=np.float32)

        cuda.memcpy_dtoh(host_dLogits, gpu_dLogits)
        if scale != 1.0:
            host_dLogits *= scale

        cuda.memcpy_dtoh(host_ln_f_output, self.cache_ln_f_output)
        cuda.memcpy_dtoh(host_lm_head_w, self.lm_head_w.gpu_weights)

        host_dLmHeadW = host_ln_f_output.T @ host_dLogits
        host_dNormalized = host_dLogits @ host_lm_head_w.T

        self.lm_head_w.set_or_accumulate_grads(host_dLmHeadW, accumulate=accumulate)

        gpu_dNormalized = cuda.mem_alloc(host_dNormalized.nbytes)
        cuda.memcpy_htod(gpu_dNormalized, host_dNormalized.astype(np.float32))

        gpu_grad, gpu_dGamma, gpu_dBeta = self.ln_bwd_op(
            gpu_dNormalized, self.cache_pre_ln_f, self.ln_f_gamma.gpu_weights, M, C
        )
        self.ln_f_gamma.set_or_accumulate_grads_from_gpu(gpu_dGamma, accumulate=accumulate)
        self.ln_f_beta.set_or_accumulate_grads_from_gpu(gpu_dBeta, accumulate=accumulate)

        gpu_dNormalized.free()
        gpu_dGamma.free()
        gpu_dBeta.free()

        for block in reversed(self.blocks):
            next_grad = block.backward(gpu_grad, B, T, accumulate=accumulate)
            if next_grad is not gpu_grad:
                gpu_grad.free()
            gpu_grad = next_grad
        self.embedding.backward(gpu_grad, accumulate=accumulate)
        gpu_grad.free()

    def _all_parameters(self):
        """Return every trainable parameter object in the model."""
        all_params = [
            self.embedding.wte,
            self.embedding.wpe,
            self.ln_f_gamma,
            self.ln_f_beta,
            self.lm_head_w,
        ]

        for block in self.blocks:
            all_params.extend([
                block.ln_1_gamma, block.ln_1_beta,
                block.ln_2_gamma, block.ln_2_beta,
                block.mlp.c_fc_w, block.mlp.c_fc_b,
                block.mlp.c_proj_w, block.mlp.c_proj_b,
            ])

            if not getattr(block.attn, "identity_fallback", False):
                all_params.extend([
                    block.attn.c_attn_w, block.attn.c_attn_b,
                    block.attn.c_proj_w, block.attn.c_proj_b,
                ])

        return all_params

    def zero_grad(self):
        """Reset parameter gradients before each backward pass."""
        for param in self._all_parameters():
            cuda.memset_d8(param.gpu_grads, 0, param.bytes_size)

    def update_weights(self, lr: float, step: int):
        """Iterates over model layers to apply stateful AdamW optimizations.
        
        Args:
            lr: Learning rate (alpha)
            step: Current optimization step (used for bias correction)
        """
        all_params = self._all_parameters()
        
        # GRADIENT CLIPPING: Disabled for debugging - checking if it's the culprit
        # Was: clip each parameter's gradients before AdamW step
        # Issue: Per-parameter clipping (host transfer + norm calc) may be buggy
        # max_grad_norm = 10.0
        # for param in all_params:
        #     self._clip_gradients(param.gpu_grads, max_grad_norm, param.total_elements)
        
        # Apply AdamW update to each parameter
        for param in all_params:
            self.optimizer_op(
                param.gpu_weights, param.gpu_m, param.gpu_v, param.gpu_grads,
                lr=lr, beta1=0.9, beta2=0.999, eps=1e-7, weight_decay=0.0001,
                step=step, total_elements=param.total_elements
            )

    def _clip_gradients(self, gpu_grads, max_grad_norm: float, total_elements: int):
        """Clip gradient tensor to prevent exploding gradients during backprop.
        
        Args:
            gpu_grads: GPU pointer to gradient array
            max_grad_norm: Maximum L2 norm of gradient (default 10.0 for stability)
            total_elements: Number of elements in gradient array
        """
        # Pull gradients to host, compute norm, clip if needed, push back
        host_grads = np.empty(total_elements, dtype=np.float32)
        cuda.memcpy_dtoh(host_grads, gpu_grads)
        
        grad_norm = np.sqrt(np.sum(host_grads ** 2))
        if grad_norm > max_grad_norm:
            # Clip by scaling gradients
            host_grads *= max_grad_norm / (grad_norm + 1e-10)
            cuda.memcpy_htod(gpu_grads, host_grads)

    def compute_grad_norm(self) -> float:
        """Return the global L2 norm of all parameter gradients for logging/debugging."""
        total_sq_norm = 0.0
        for param in self._all_parameters():
            host_grads = np.empty(param.total_elements, dtype=np.float32)
            cuda.memcpy_dtoh(host_grads, param.gpu_grads)
            total_sq_norm += float(np.dot(host_grads, host_grads))
        return float(np.sqrt(total_sq_norm))
    
    def free_forward_caches(self):
        """Iterates over sequential layer blocks to enforce strict VRAM footprint scrubbing.
        
        Must be called after backward pass to prevent VRAM accumulation across training steps.
        """
        if self.cache_ln_f_output is not None:
            self.cache_ln_f_output.free()
            self.cache_ln_f_output = None
        if self.cache_pre_ln_f is not None:
            self.cache_pre_ln_f.free()
            self.cache_pre_ln_f = None

        self.embedding.free_forward_caches()
        for block in self.blocks:
            block.free_forward_caches()

    def free_persistent_buffers(self):
        """Release persistent inference buffers (zero bias, position indices)."""
        if self._gpu_zero_bias is not None:
            self._gpu_zero_bias.free()
            self._gpu_zero_bias = None
        self.embedding.free_persistent_buffers()

    def _validate_checkpoint_array(self, name, expected_shape, array):
        actual_shape = tuple(array.shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Checkpoint tensor '{name}' shape {actual_shape} != model shape {expected_shape}")

    def save_checkpoint(self, file_path: str):
        """Pulls weights from physical GPU VRAM and flushes them to a compressed file on disk.
        
        Args:
            file_path: Path to save checkpoint (e.g., 'output/checkpoints/gpt_model_latest.npz')
        """
        logger.info(f"[SAVE] Archiving parameter weights to disk checkpoint: {file_path}")
        
        # Pull all active device parameter states back over the PCIe lane to host memory
        checkpoint_dict = {}
        checkpoint_dict["__meta_num_heads"] = np.array(self.config.num_heads, dtype=np.int32)
        checkpoint_dict["__meta_attention_impl"] = np.array(self.config.attention_impl)
        checkpoint_dict["__meta_vocab_size"] = np.array(self.config.vocab_size, dtype=np.int32)
        checkpoint_dict["__meta_max_len"] = np.array(self.config.max_len, dtype=np.int32)
        checkpoint_dict["__meta_embedding_dim"] = np.array(self.config.embedding_dim, dtype=np.int32)
        checkpoint_dict["__meta_num_layers"] = np.array(self.config.num_layers, dtype=np.int32)
        
        # 1. Capture base embedding matrices
        wte_host = np.empty(self.embedding.wte.shape, dtype=np.float32)
        wpe_host = np.empty(self.embedding.wpe.shape, dtype=np.float32)
        cuda.memcpy_dtoh(wte_host, self.embedding.wte.gpu_weights)
        cuda.memcpy_dtoh(wpe_host, self.embedding.wpe.gpu_weights)
        checkpoint_dict["wte"] = wte_host
        checkpoint_dict["wpe"] = wpe_host
        
        # 2. Capture cascading transformer block weights
        for idx, block in enumerate(self.blocks):
            for attr in ["ln_1_gamma", "ln_1_beta", "ln_2_gamma", "ln_2_beta"]:
                p_obj = getattr(block, attr)
                host_arr = np.empty(p_obj.shape, dtype=np.float32)
                cuda.memcpy_dtoh(host_arr, p_obj.gpu_weights)
                checkpoint_dict[f"block_{idx}_{attr}"] = host_arr
                
            for sub_layer in ["attn", "mlp"]:
                layer_obj = getattr(block, sub_layer)
                for attr in ["c_attn_w", "c_attn_b", "c_proj_w", "c_proj_b", "c_fc_w", "c_fc_b"]:
                    if hasattr(layer_obj, attr):
                        p_obj = getattr(layer_obj, attr)
                        host_arr = np.empty(p_obj.shape, dtype=np.float32)
                        cuda.memcpy_dtoh(host_arr, p_obj.gpu_weights)
                        checkpoint_dict[f"block_{idx}_{sub_layer}_{attr}"] = host_arr

        # 3. Capture final normalization and vocabulary projection head layers
        ln_f_gamma_host = np.empty(self.ln_f_gamma.shape, dtype=np.float32)
        ln_f_beta_host = np.empty(self.ln_f_beta.shape, dtype=np.float32)
        lm_head_w_host = np.empty(self.lm_head_w.shape, dtype=np.float32)
        cuda.memcpy_dtoh(ln_f_gamma_host, self.ln_f_gamma.gpu_weights)
        cuda.memcpy_dtoh(ln_f_beta_host, self.ln_f_beta.gpu_weights)
        cuda.memcpy_dtoh(lm_head_w_host, self.lm_head_w.gpu_weights)
        
        checkpoint_dict["ln_f_gamma"] = ln_f_gamma_host
        checkpoint_dict["ln_f_beta"] = ln_f_beta_host
        checkpoint_dict["lm_head_w"] = lm_head_w_host
        
        # Save as a consolidated compressed archive file
        target_dir = os.path.dirname(file_path) or '.'
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.basename(file_path)
        temp_fd, temp_path = tempfile.mkstemp(prefix=f".{base_name}.", suffix=".tmp", dir=target_dir)
        os.close(temp_fd)

        try:
            with open(temp_path, 'wb') as temp_file:
                np.savez_compressed(temp_file, **checkpoint_dict)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            validate_checkpoint_archive(temp_path)
            os.replace(temp_path, file_path)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        logger.info("[OK] Checkpoint write operation finalized successfully.")

    def load_checkpoint(self, file_path: str) -> bool:
        """Loads an archived checkpoint file from disk and pushes values up to active GPU VRAM.
        
        Args:
            file_path: Path to checkpoint file
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"❌ Aborting load: Checkpoint file not found at {file_path}")
            return False

        validate_checkpoint_archive(file_path)
            
        logger.info(f"📂 Hydrating VRAM configurations from disk checkpoint: {file_path}")
        with np.load(file_path, allow_pickle=False) as data:
            if "__meta_attention_impl" in data.files:
                saved_attention_impl = str(np.asarray(data["__meta_attention_impl"]).item())
                if saved_attention_impl != self.config.attention_impl:
                    raise ValueError(f"Checkpoint attention_impl={saved_attention_impl} != model attention_impl={self.config.attention_impl}")
            if "__meta_num_heads" in data.files:
                saved_num_heads = int(np.asarray(data["__meta_num_heads"]).item())
                if saved_num_heads != self.config.num_heads:
                    raise ValueError(f"Checkpoint num_heads={saved_num_heads} != model num_heads={self.config.num_heads}")
            
            # Push variables back up over PCIe lines into active device memory pointers
            wte_arr = data["wte"].astype(np.float32)
            self._validate_checkpoint_array("wte", self.embedding.wte.shape, wte_arr)
            cuda.memcpy_htod(self.embedding.wte.gpu_weights, wte_arr)
            
            wpe_arr = data["wpe"].astype(np.float32)
            self._validate_checkpoint_array("wpe", self.embedding.wpe.shape, wpe_arr)
            cuda.memcpy_htod(self.embedding.wpe.gpu_weights, wpe_arr)
            
            for idx, block in enumerate(self.blocks):
                for attr in ["ln_1_gamma", "ln_1_beta", "ln_2_gamma", "ln_2_beta"]:
                    p_obj = getattr(block, attr)
                    arr = data[f"block_{idx}_{attr}"].astype(np.float32)
                    self._validate_checkpoint_array(f"block_{idx}_{attr}", p_obj.shape, arr)
                    cuda.memcpy_htod(p_obj.gpu_weights, arr)
                    
                for sub_layer in ["attn", "mlp"]:
                    layer_obj = getattr(block, sub_layer)
                    for attr in ["c_attn_w", "c_attn_b", "c_proj_w", "c_proj_b", "c_fc_w", "c_fc_b"]:
                        if hasattr(layer_obj, attr):
                            p_obj = getattr(layer_obj, attr)
                            arr = data[f"block_{idx}_{sub_layer}_{attr}"].astype(np.float32)
                            self._validate_checkpoint_array(f"block_{idx}_{sub_layer}_{attr}", p_obj.shape, arr)
                            cuda.memcpy_htod(p_obj.gpu_weights, arr)
                            
            ln_f_gamma_arr = data["ln_f_gamma"].astype(np.float32)
            self._validate_checkpoint_array("ln_f_gamma", self.ln_f_gamma.shape, ln_f_gamma_arr)
            cuda.memcpy_htod(self.ln_f_gamma.gpu_weights, ln_f_gamma_arr)
            
            ln_f_beta_arr = data["ln_f_beta"].astype(np.float32)
            self._validate_checkpoint_array("ln_f_beta", self.ln_f_beta.shape, ln_f_beta_arr)
            cuda.memcpy_htod(self.ln_f_beta.gpu_weights, ln_f_beta_arr)
            
            lm_head_w_arr = data["lm_head_w"].astype(np.float32)
            self._validate_checkpoint_array("lm_head_w", self.lm_head_w.shape, lm_head_w_arr)
            cuda.memcpy_htod(self.lm_head_w.gpu_weights, lm_head_w_arr)
        
        logger.info("✅ Core parameters statefully hydrated. Model ready for instant execution.")
        return True



