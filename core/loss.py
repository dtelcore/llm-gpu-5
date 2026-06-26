# core/loss.py
"""
Fused Softmax Cross-Entropy Loss Operator for GPU-Accelerated Training.

Combines forward loss calculation and backward gradient derivation into a single
parallel kernel to eliminate intermediate probability matrix storage in VRAM.
Each GPU thread handles a single sequence element, calculating row-wise softmax
stably using the log-sum-exp trick and generating corresponding logit gradients.

Target: NVIDIA GeForce GT 730 (Kepler, sm_35)
"""

import numpy as np
import env_config
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


# ============================================================================
# FUSED SOFTMAX CROSS-ENTROPY LOSS CUDA KERNEL
# ============================================================================

FUSED_LOSS_KERNEL = """
extern "C" {
    __global__ void fused_softmax_cross_entropy_kernel(
        const float* __restrict__ d_logits,    // Input flat logits matrix: shape (N, V)
        const int* __restrict__ d_targets,     // Correct token indices: shape (N)
        float* __restrict__ d_losses,          // Output loss value per row: shape (N)
        float* __restrict__ d_dLogits,         // Output loss gradients matrix: shape (N, V)
        const int N,                           // Total sequence items across batch (B * T)
        const int V,                           // Vocabulary size dimensions
        const int pad_token_id                 // Token ID to ignore in loss (typically PAD/BOS)
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= N) return;

        int row_offset = idx * V;
        int target_label = d_targets[idx];

        // ========== MASK OUT PAD TOKENS ==========
        // Skip loss computation for padding tokens
        if (target_label == pad_token_id) {
            d_losses[idx] = 0.0f;
            for (int v = 0; v < V; ++v) {
                d_dLogits[row_offset + v] = 0.0f;
            }
            return;
        }

        // 1. Find max logit value in row to maintain strict numerical stability (Log-Sum-Exp trick)
        float max_val = d_logits[row_offset];
        for (int v = 1; v < V; ++v) {
            float val = d_logits[row_offset + v];
            if (val > max_val) {
                max_val = val;
            }
        }

        // 2. Compute normalizer sum of exponentials safely
        float sum_exp = 0.0f;
        for (int v = 0; v < V; ++v) {
            sum_exp += __expf(d_logits[row_offset + v] - max_val);
        }

        // 3. Calculate negative log-likelihood loss for this specific row item
        float target_logit = d_logits[row_offset + target_label];
        d_losses[idx] = __logf(sum_exp) - (target_logit - max_val);

        // 4. Derive downstream softmax probabilities and build matching gradient distributions
        // Formula: dLogits[i, j] = (prob[j] - Indicator(j == target)) / N
        for (int v = 0; v < V; ++v) {
            float prob = __expf(d_logits[row_offset + v] - max_val) / sum_exp;
            float indicator = (v == target_label) ? 1.0f : 0.0f;
            
            // Standardized normalization over total sequence instances (N) to match batch means
            d_dLogits[row_offset + v] = (prob - indicator) / (float)N;
        }
    }
}
"""


class SoftmaxCrossEntropy:
    """Handles memory streaming operations computing cross-entropy losses and analytical gradients.
    
    Fused kernel computes:
    - Forward: Cross-entropy loss per sequence position (negative log-likelihood)
    - Backward: Analytical gradient derivation from softmax probabilities
    
    Memory efficient: No intermediate probability matrix stored in VRAM.
    Numerical stable: Log-sum-exp trick prevents over/underflow on Kepler FPU.
    """
    
    def __init__(self):
        """Initialize CUDA kernel compilation for Kepler GT 730 (sm_35)."""
        nvcc_options = [
            "-ccbin", env_config.MSVC_142_BIN,
            "-O3",
            "--use_fast_math",
            "-Xcompiler", "/w",
        ]
        self.mod = SourceModule(FUSED_LOSS_KERNEL, options=nvcc_options)
        self.func = self.mod.get_function("fused_softmax_cross_entropy_kernel")

    def __call__(self, gpu_logits, gpu_targets, N: int, V: int, pad_token_id: int = -1):
        """Execute parallel loss metrics calculation and gradient derivation.
        
        Args:
            gpu_logits: GPU pointer to logit matrix [N, V] (float32)
            gpu_targets: GPU pointer to target token indices [N] (int32)
            N: Total sequence positions (B * T)
            V: Vocabulary size
            pad_token_id: Token ID to mask during loss calculation (default: -1 = no masking)
            
        Returns:
            host_mean_loss (float): Averaged batch loss scalar pulled from VRAM
            gpu_dLogits (DeviceAllocation): Input layer gradients [N, V] (float32)
        """
        # Allocate flat arrays to store calculation output states
        gpu_losses = cuda.mem_alloc(N * 4)           # 4 bytes per float32
        gpu_dLogits = cuda.mem_alloc(N * V * 4)      # Full gradient dimension map matrix
        
        # Thread-per-element parallelization (256 threads per block is optimal for GT730)
        threads = 256
        blocks = (N + threads - 1) // threads
        
        # Launch fused kernel with PAD token masking
        self.func(
            gpu_logits, gpu_targets, gpu_losses, gpu_dLogits,
            np.int32(N), np.int32(V), np.int32(pad_token_id),
            block=(threads, 1, 1), grid=(blocks, 1)
        )
        
        # Pull row-wise losses back across the PCIe bus to calculate the average on host CPU
        host_losses = np.empty(N, dtype=np.float32)
        cuda.memcpy_dtoh(host_losses, gpu_losses)
        gpu_losses.free()  # Free localized temporary array immediately
        
        mean_loss = float(np.mean(host_losses))
        
        # Detect NaN/Inf losses (numerical instability indicator)
        if not np.isfinite(mean_loss):
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[ERROR] NaN/Inf loss detected: {mean_loss}. Gradient explosion likely. Reduce learning rate or enable gradient clipping.")
        
        return mean_loss, gpu_dLogits
