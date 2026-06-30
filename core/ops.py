# core/ops.py
"""
Production-Ready PyCUDA Operator Wrappers for GPU Kernel Execution.

Provides object-oriented abstractions over 13 CUDA kernels (7 forward + 6 backward).
All kernels are compiled in a single pass for optimal startup performance on Windows.

Design Principles:
- Single-pass compilation: All kernels fused into one SourceModule to eliminate multi-invocation overhead
- Explicit type binding: All scalar parameters cast to np.int32, np.float32, etc. for PyCUDA safety
- Memory initialization: Weight gradient accumulators cleared with cuda.memset_d8 before atomic reductions
- In-place operations: Forward/backward layers modify tensors in-place where possible to save VRAM
"""

import numpy as np
import env_config
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from core.kernels import KERNELS

# ============================================================================
# CENTRALIZED COMPILATION LAYER
# ============================================================================
def _compile_gpu_kernels():
    """Concatenates and JIT compiles all 13 kernels in a single compiler pass."""
    nvcc_options = [
        "-ccbin", env_config.MSVC_142_BIN,
        "-O3",
        "--use_fast_math",
        "-Xcompiler", "/w",
    ]
    unified_source = "\n\n".join(KERNELS.values())
    return SourceModule(unified_source, options=nvcc_options)

# Instantiated once at module-load time to eliminate multi-compilation overhead
_SHARED_GPU_MODULE = _compile_gpu_kernels()


# ============================================================================
# FORWARD INFERENCE OPERATOR WRAPPERS
# ============================================================================

class EmbeddingLookup:
    """Token ID → Embedding Vector Lookup via Thread-Per-Element Parallelism.
    
    Each thread computes exactly one output element: output[b,t,c] = weights[token_id[b,t], c]
    """
    def __init__(self, vocab_size=None, embedding_dim=None):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.func = _SHARED_GPU_MODULE.get_function("embedding_lookup_kernel")
        
    def __call__(self, gpu_tokens, gpu_weights, B: int, T: int, C: int):
        """Execute embedding lookup kernel.
        
        Args:
            gpu_tokens: GPU pointer to token ID matrix [B, T] (int32)
            gpu_weights: GPU pointer to embedding weight table [Vocab, C] (float32)
            B: Batch dimension
            T: Sequence length dimension
            C: Embedding/channel dimension
            
        Returns:
            gpu_output: GPU pointer to embedding output [B, T, C] (float32)
        """
        total_elements = B * T * C
        gpu_output = cuda.mem_alloc(total_elements * 4)  # 4 bytes per float32
        
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.func(gpu_tokens, gpu_weights, gpu_output, np.int32(total_elements), np.int32(C),
                  block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_output


class ElementwiseAdd:
    """In-Place Element-Wise Addition: target += source.
    
    Used for residual connections and position embedding injection.
    Modifies target tensor in-place to save VRAM.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("elementwise_add_kernel")
        
    def __call__(self, gpu_target, gpu_source, total_elements: int):
        """Execute in-place addition kernel.
        
        Args:
            gpu_target: GPU pointer to target array (modified in-place) (float32)
            gpu_source: GPU pointer to source array (read-only) (float32)
            total_elements: Total number of float elements
        """
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.func(gpu_target, gpu_source, np.int32(total_elements),
                  block=(threads, 1, 1), grid=(blocks, 1))


class LayerNorm:
    """Per-Row Layer Normalization with Learnable Scale and Bias.
    
    Computes: output = (input - mean) / sqrt(variance) * gamma + beta
    """
    def __init__(self, embedding_dim=None, eps=1e-5):
        self.embedding_dim = embedding_dim
        self.normalized_shape = embedding_dim  # PyTorch-compatible attribute
        self.eps = eps
        self.func = _SHARED_GPU_MODULE.get_function("layernorm_kernel")
        
    def __call__(self, gpu_input, gpu_gamma, gpu_beta, num_rows: int, channels: int):
        """Execute layer normalization kernel.
        
        Args:
            gpu_input: GPU pointer to input [num_rows, channels] (float32)
            gpu_gamma: GPU pointer to scale vector [channels] (float32)
            gpu_beta: GPU pointer to bias vector [channels] (float32)
            num_rows: Number of rows (typically batch_size * seq_length)
            channels: Number of channels per row
            
        Returns:
            gpu_output: GPU pointer to normalized output [num_rows, channels] (float32)
        """
        gpu_output = cuda.mem_alloc(num_rows * channels * 4)
        
        threads = 256
        blocks = (num_rows + threads - 1) // threads
        
        self.func(gpu_input, gpu_output, gpu_gamma, gpu_beta, np.int32(num_rows), np.int32(channels),
                  block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_output


class MatMul2D:
    """Dense Linear Transformation with Fused Bias Addition: C = A @ B + Bias.
    
    Used for QKV projection, output projection, and FFN layers.
    Bias is added during the accumulation step to reduce global memory traffic.
    """
    def __init__(self, in_features=None, out_features=None):
        self.in_features = in_features
        self.out_features = out_features
        self.func = _SHARED_GPU_MODULE.get_function("matmul_2d_kernel")
        
    def __call__(self, gpu_A, gpu_B, gpu_bias, M: int, N: int, K: int):
        """Execute 2D matrix multiplication with bias kernel.
        
        Args:
            gpu_A: GPU pointer to input matrix [M, K] (float32)
            gpu_B: GPU pointer to weight matrix [K, N] (float32)
            gpu_bias: GPU pointer to bias vector [N] (float32)
            M: Batch/sequence dimension
            N: Output feature dimension
            K: Input feature dimension
            
        Returns:
            gpu_C: GPU pointer to output [M, N] (float32)
        """
        gpu_C = cuda.mem_alloc(M * N * 4)
        
        block_dim = 16
        block = (block_dim, block_dim, 1)
        grid = ((N + block_dim - 1) // block_dim, (M + block_dim - 1) // block_dim)
        
        self.func(gpu_A, gpu_B, gpu_bias, gpu_C, np.int32(M), np.int32(N), np.int32(K),
                  block=block, grid=grid)
        return gpu_C


class MatmulStrided:
    """Batched 3D Matrix Multiplication for Multi-Head Self-Attention.
    
    Efficiently computes batched matrix multiplications across batch and head dimensions.
    Used for Q@K^T (attention scores) and Attention@V (context mixing).
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("matrix_multiply_strided_kernel")
        
    def __call__(self, gpu_A, gpu_B, B: int, NH: int, M: int, N: int, K: int, 
                 stride_A: int, stride_B: int, stride_C: int):
        """Execute batched strided matrix multiplication kernel.
        
        Args:
            gpu_A: GPU pointer to 3D input tensor [B, NH, M, K] (float32)
            gpu_B: GPU pointer to 3D input tensor [B, NH, K, N] (float32)
            B: Batch dimension
            NH: Number of attention heads
            M: Sequence length (for Q@K^T: M=T, N=T; for Att@V: M=T, N=C/NH)
            N: Output feature/sequence dimension per head
            K: Input feature dimension per head
            stride_A: Byte stride between matrix blocks in A (for 3D indexing)
            stride_B: Byte stride between matrix blocks in B (for 3D indexing)
            stride_C: Byte stride between matrix blocks in output (for 3D indexing)
            
        Returns:
            gpu_C: GPU pointer to batched output [B, NH, M, N] (float32)
        """
        gpu_C = cuda.mem_alloc(B * NH * M * N * 4)
        
        block_dim = 16
        block = (block_dim, block_dim, 1)
        grid = ((N + block_dim - 1) // block_dim, (M + block_dim - 1) // block_dim, B * NH)
        
        self.func(gpu_A, gpu_B, gpu_C, np.int32(M), np.int32(N), np.int32(K),
                  np.int32(stride_A), np.int32(stride_B), np.int32(stride_C),
                  block=block, grid=grid)
        return gpu_C


class CausalSoftmax:
    """Fused Autoregressive Masking + Numerically Stable Softmax.
    
    Applies lower-triangular causal mask and computes softmax in-place using three-pass reduction:
    1. Max-reduction (for numerical stability)
    2. Exp-subtraction
    3. Normalization
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("causal_softmax_kernel")
        
    def __call__(self, gpu_scores, total_rows: int, T: int, scale_factor: float):
        """Execute causal softmax kernel.
        
        Args:
            gpu_scores: GPU pointer to attention score matrix [total_rows, T] (float32, modified in-place)
            total_rows: Number of rows = batch_size * num_heads * seq_length
            T: Sequence length (for causal mask triangle)
            scale_factor: Scaling factor (typically 1 / sqrt(head_dim))
        """
        threads = 256
        blocks = (total_rows + threads - 1) // threads
        
        self.func(gpu_scores, np.int32(total_rows), np.int32(T), np.float32(scale_factor),
                  block=(threads, 1, 1), grid=(blocks, 1))


class GELU:
    """In-Place GELU Activation Function.
    
    Applies GELU to tensor in-place, used in FFN hidden layers.
    Saves VRAM by modifying input tensor directly.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("activation_kernel")
        
    def __call__(self, gpu_data, total_elements: int):
        """Execute GELU activation kernel.
        
        Args:
            gpu_data: GPU pointer to tensor (modified in-place) (float32)
            total_elements: Total number of float elements
        """
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.func(gpu_data, np.int32(total_elements), block=(threads, 1, 1), grid=(blocks, 1))


# ============================================================================
# BACKWARD TRAINING OPERATOR WRAPPERS
# ============================================================================

class Dropout:
    """Dropout with Inverted Scaling and Saved Mask for Backpropagation.
    
    Forward: Applies random masking with 1/(1-p) scaling and saves 1-byte mask
    Backward: Routes gradients through saved mask with matching scaling
    
    1-byte mask storage saves 4× VRAM compared to storing full float32 masks.
    """
    def __init__(self, dropout_prob=0.0):
        self.dropout_prob = dropout_prob
        self.dropout_rate = dropout_prob  # PyTorch-compatible attribute name
        self.training = True  # Track training/eval mode
        self.fwd_func = _SHARED_GPU_MODULE.get_function("dropout_forward_kernel")
        self.bwd_func = _SHARED_GPU_MODULE.get_function("dropout_backward_kernel")
    
    def eval(self):
        """Set dropout to evaluation mode (no dropout applied)."""
        self.training = False
        return self
    
    def train(self):
        """Set dropout to training mode (dropout applied)."""
        self.training = True
        return self
        
    def forward(self, gpu_input, total_elements: int, dropout_prob: float, seed: int):
        """Execute dropout forward pass with XORShift32 PRNG.
        
        Args:
            gpu_input: GPU pointer to input tensor (float32)
            total_elements: Total number of float elements
            dropout_prob: Probability of dropping each element (e.g., 0.1)
            seed: Random seed for PRNG (typically step counter)
            
        Returns:
            (gpu_output, gpu_mask): Output tensor (float32) and 1-byte mask (uint8)
        """
        gpu_output = cuda.mem_alloc(total_elements * 4)
        gpu_mask = cuda.mem_alloc(total_elements * 1)  # Allocated as 1-byte unsigned chars
        
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.fwd_func(gpu_input, gpu_output, gpu_mask, np.int32(total_elements), 
                      np.float32(dropout_prob), np.uint32(seed),
                      block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_output, gpu_mask

    def backward(self, gpu_dOut, gpu_mask, total_elements: int, dropout_prob: float):
        """Execute dropout backward pass to propagate gradients through saved mask.
        
        Args:
            gpu_dOut: GPU pointer to upstream gradient (float32)
            gpu_mask: GPU pointer to saved mask from forward pass (uint8)
            total_elements: Total number of elements
            dropout_prob: Same dropout probability as forward
            
        Returns:
            gpu_dIn: Downstream gradient tensor (float32)
        """
        gpu_dIn = cuda.mem_alloc(total_elements * 4)
        
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.bwd_func(gpu_dOut, gpu_dIn, gpu_mask, np.int32(total_elements), np.float32(dropout_prob),
                      block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_dIn


class ReLUBackward:
    """ReLU Gradient Gate: Pass Gradients Only Through Active Channels.
    
    If forward activation > 0, pass gradient; else block gradient (set to 0).
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("relu_backward_kernel")
        
    def __call__(self, gpu_dOut, gpu_forward_act, total_elements: int):
        """Execute ReLU backward kernel.
        
        Args:
            gpu_dOut: GPU pointer to upstream gradient (float32)
            gpu_forward_act: GPU pointer to saved forward activation [total_elements] (float32)
            total_elements: Total number of elements
            
        Returns:
            gpu_dIn: Downstream gradient (float32), zeroed where forward < 0
        """
        gpu_dIn = cuda.mem_alloc(total_elements * 4)
        
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.func(gpu_dOut, gpu_forward_act, gpu_dIn, np.int32(total_elements),
                  block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_dIn


class MatMulBackwardWeights:
    """Weight Gradient Computation via Transposed Input Reduction: dW = X^T @ dOut.
    
    Computes weight gradients for backpropagation through dense layers.
    Output is zeroed before kernel launch to allow thread-safe atomic accumulations.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("matmul_backward_weights_kernel")
        
    def __call__(self, gpu_X, gpu_dOut, M: int, N: int, K: int):
        """Execute weight gradient computation kernel.
        
        Args:
            gpu_X: GPU pointer to forward input activations [M, K] (float32)
            gpu_dOut: GPU pointer to upstream gradient [M, N] (float32)
            M: Batch/sequence dimension
            N: Output feature dimension
            K: Input feature dimension (weight matrix rows)
            
        Returns:
            gpu_dW: Weight gradient tensor [K, N] (float32)
        """
        gpu_dW = cuda.mem_alloc(K * N * 4)
# Removed explicit zeroing to avoid unnecessary overhead; kernel computes gradients directly.
        
        block_dim = 16
        block = (block_dim, block_dim, 1)
        grid = ((N + block_dim - 1) // block_dim, (K + block_dim - 1) // block_dim)
        
        self.func(gpu_X, gpu_dOut, gpu_dW, np.int32(M), np.int32(N), np.int32(K),
                  block=block, grid=grid)
        return gpu_dW


class LayerNormBackward:
    """LayerNorm Backward: Fused Mean/Variance Gradient Reduction with Atomic Accumulation.
    
    Computes:
    - Downstream gradient: dIn = rsqrt(var) * (dOut * gamma - mean(dOut * gamma) - x_hat * mean(dOut * gamma * x_hat))
    - Parameter gradients: dGamma = sum(dOut * x_hat), dBeta = sum(dOut)
    
    Uses atomicAdd for thread-safe concurrent gradient accumulation on Kepler hardware.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("layernorm_backward_kernel")
        
    def __call__(self, gpu_dOut, gpu_input, gpu_gamma, num_rows: int, channels: int):
        """Execute layer norm backward kernel.
        
        Args:
            gpu_dOut: GPU pointer to upstream gradient [num_rows, channels] (float32)
            gpu_input: GPU pointer to forward input [num_rows, channels] (float32)
            gpu_gamma: GPU pointer to forward scale weights [channels] (float32)
            num_rows: Number of rows
            channels: Number of channels
            
        Returns:
            (gpu_dIn, gpu_dGamma, gpu_dBeta): Downstream gradient and parameter gradients
        """
        gpu_dIn = cuda.mem_alloc(num_rows * channels * 4)
        gpu_dGamma = cuda.mem_alloc(channels * 4)
        gpu_dBeta = cuda.mem_alloc(channels * 4)
        
        # Initialize weight accumulators to zero to allow thread-safe global atomic additions
        cuda.memset_d8(gpu_dGamma, 0, channels * 4)
        cuda.memset_d8(gpu_dBeta, 0, channels * 4)
        
        threads = 256
        blocks = (num_rows + threads - 1) // threads
        
        self.func(gpu_dOut, gpu_input, gpu_gamma, gpu_dIn, gpu_dGamma, gpu_dBeta,
                  np.int32(num_rows), np.int32(channels),
                  block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_dIn, gpu_dGamma, gpu_dBeta


class GELUBackward:
    """GELU Gradient via Tanh-Approximation Derivative.

    Recomputes the GELU derivative per-element directly from the saved pristine
    pre-activation (pre-GELU) state, avoiding any host round-trip.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("gelu_backward_kernel")

    def __call__(self, gpu_dOut, gpu_forward_act, total_elements: int):
        """Execute GELU backward kernel.

        Args:
            gpu_dOut: GPU pointer to upstream gradient (float32)
            gpu_forward_act: GPU pointer to saved pristine pre-activation state [total_elements] (float32)
            total_elements: Total number of elements

        Returns:
            gpu_dIn: Downstream gradient (float32)
        """
        gpu_dIn = cuda.mem_alloc(total_elements * 4)

        threads = 256
        blocks = (total_elements + threads - 1) // threads

        self.func(gpu_dOut, gpu_forward_act, gpu_dIn, np.int32(total_elements),
                  block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_dIn


class ReduceSumAxis0:
    """Column-Sum-Over-Rows Reduction via Atomic Accumulation.

    Computes output[c] = sum_row input[row, c] for a logically (num_rows, channels)
    tensor, supporting an arbitrary row stride for non-contiguous layouts.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("reduce_sum_axis0_kernel")

    def __call__(self, gpu_input, num_rows: int, channels: int, stride_row: int):
        """Execute axis-0 reduction kernel.

        Args:
            gpu_input: GPU pointer to input tensor (float32)
            num_rows: Number of rows to reduce over
            channels: Number of output columns
            stride_row: Elements between consecutive rows in gpu_input

        Returns:
            gpu_output: GPU pointer to column-sum accumulator [channels] (float32)
        """
        gpu_output = cuda.mem_alloc(channels * 4)
        cuda.memset_d8(gpu_output, 0, channels * 4)

        threads = 256
        blocks = (num_rows + threads - 1) // threads

        self.func(gpu_input, gpu_output, np.int32(num_rows), np.int32(channels), np.int32(stride_row),
                  block=(threads, 1, 1), grid=(blocks, 1))
        return gpu_output


class MatMulBackwardInput:
    """Input Gradient Computation via Implicit Transpose-B Matmul: dX = dOut @ W^T.

    Computes input gradients without physically transposing the weight matrix,
    reading W in its natural [K, N] on-device layout.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("matmul_backward_input_kernel")

    def __call__(self, gpu_dOut, gpu_W, M: int, N: int, K: int):
        """Execute input-gradient matmul kernel.

        Args:
            gpu_dOut: GPU pointer to upstream gradient [M, N] (float32)
            gpu_W: GPU pointer to forward weight matrix, natural layout [K, N] (float32)
            M: Batch/sequence dimension
            N: Contraction dimension (dOut's feature dimension)
            K: Output feature dimension (dIn's feature dimension, W's first axis)

        Returns:
            gpu_dIn: Input gradient tensor [M, K] (float32)
        """
        gpu_dIn = cuda.mem_alloc(M * K * 4)

        block_dim = 16
        block = (block_dim, block_dim, 1)
        grid = ((K + block_dim - 1) // block_dim, (M + block_dim - 1) // block_dim)

        self.func(gpu_dOut, gpu_W, gpu_dIn, np.int32(M), np.int32(N), np.int32(K),
                  block=block, grid=grid)
        return gpu_dIn


class AdamW:
    """Stateful AdamW Optimizer: Decoupled Weight Decay + Exponential Moving Averages.
    
    Maintains first moment (m) and second moment (v) histories on GPU.
    Updates parameters in-place with bias-corrected learning rates.
    Applies decoupled weight decay independently from gradient-based updates.
    """
    def __init__(self):
        self.func = _SHARED_GPU_MODULE.get_function("adamw_update_kernel")
        
    def __call__(self, gpu_weights, gpu_m, gpu_v, gpu_grads, lr: float, beta1: float, beta2: float,
                 eps: float, weight_decay: float, step: int, total_elements: int):
        """Execute AdamW optimizer step kernel.
        
        Args:
            gpu_weights: GPU pointer to parameter weights [total_elements] (float32, modified in-place)
            gpu_m: GPU pointer to first moment (mean) state [total_elements] (float32, modified in-place)
            gpu_v: GPU pointer to second moment (variance) state [total_elements] (float32, modified in-place)
            gpu_grads: GPU pointer to computed gradients [total_elements] (float32)
            lr: Learning rate (alpha)
            beta1: Exponential decay rate for first moment (typically 0.9)
            beta2: Exponential decay rate for second moment (typically 0.999)
            eps: Small constant for numerical stability (typically 1e-8)
            weight_decay: Decoupled L2 regularization coefficient (typically 0.01)
            step: Current optimization step number (used for bias correction)
            total_elements: Total number of parameters
        """
        # Calculate mathematical temporal scaling scalars on Host CPU
        bias_correction1 = 1.0 - (beta1 ** step)
        bias_correction2 = 1.0 - (beta2 ** step)
        
        threads = 256
        blocks = (total_elements + threads - 1) // threads
        
        self.func(
            gpu_weights, gpu_m, gpu_v, gpu_grads,
            np.float32(lr), np.float32(beta1), np.float32(beta2), np.float32(eps), np.float32(weight_decay),
            np.float32(bias_correction1), np.float32(bias_correction2), np.int32(total_elements),
            block=(threads, 1, 1), grid=(blocks, 1)
        )
