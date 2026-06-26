"""
CUDA C++ Kernel Definitions for Legacy Kepler GPU (GT 730 / GK208 / Compute Capability 3.5).
Optimized for low-bandwidth VRAM caching, thread coalescing, and manual byte-striding.

Contains 13 essential kernels for a complete GPT transformer stack with full backpropagation:

Forward/Inference Kernels (7):
1. Embedding lookup (token → hidden embedding)
2. Elementwise add (residual connections, position embedding injection)
3. Layer normalization (pre-attention/FFN normalization)
4. 2D matrix multiplication with bias (dense projections)
5. Batched strided matrix multiplication (multi-head attention Q@K^T and Attention@V)
6. Fused causal softmax (autoregressive masking + stability normalization)
7. Activation function (ReLU for FFN hidden layer)

Training/Backward Kernels (6):
8. Dropout forward (XORShift32 PRNG with 1-byte mask for VRAM efficiency)
9. Dropout backward (gradient routing through saved mask)
10. ReLU backward (gradient gating for active channels)
11. MatMul backward for weights (dW = X^T @ dOut)
12. LayerNorm backward (fused mean/variance reduction with atomicAdd)
13. AdamW update (stateful optimizer with first/second moment tracking)
"""

# ============================================================================
# 1. EMBEDDING LOOKUP KERNEL (Thread-Per-Element Parallelism)
# ============================================================================
EMBEDDING_LOOKUP_KERNEL = """
extern "C" {
    __global__ void embedding_lookup_kernel(
        const int* __restrict__ d_tokens,      // Input tensor: shape (B, T)
        const float* __restrict__ d_weights,   // Weight table: shape (Vocab_Size, C)
        float* __restrict__ d_output,          // Output tensor: shape (B, T, C)
        const int total_elements,              // B * T * C
        const int embedding_dim                // C
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;
        
        int token_index = idx / embedding_dim;  
        int channel     = idx % embedding_dim;  
        
        int token_id = d_tokens[token_index];
        int weight_idx = (token_id * embedding_dim) + channel;
        
        d_output[idx] = d_weights[weight_idx];
    }
}
"""

# ============================================================================
# 2. ELEMENTWISE ADD / RESIDUAL KERNEL
# ============================================================================
ELEMENTWISE_ADD_KERNEL = """
extern "C" {
    __global__ void elementwise_add_kernel(
        float* __restrict__ d_target,          // Array to modify in-place: shape (B, T, C)
        const float* __restrict__ d_source,    // Array to read from: shape (B, T, C)
        const int total_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;
        
        // Parallel in-place summation for residual streams and position embeddings
        d_target[idx] += d_source[idx];
    }
}
"""

# ============================================================================
# 3. LAYER NORMALIZATION KERNEL (Thread-Per-Row Layout)
# ============================================================================
LAYERNORM_KERNEL = """
extern "C" {
    __global__ void layernorm_kernel(
        const float* __restrict__ d_input,     // Input tensor: shape (B*T, C)
        float* __restrict__ d_output,          // Output tensor: shape (B*T, C)
        const float* __restrict__ d_gamma,     // Scale vector: shape (C)
        const float* __restrict__ d_beta,      // Bias vector: shape (C)
        const int num_rows,                    // B * T
        const int channels                     // C
    ) {
        // Each thread processes exactly one row of channels (Highly efficient for small C dimensions)
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows) return;
        
        int row_offset = row * channels;
        
        // Step 1: Compute Mean
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            sum += d_input[row_offset + c];
        }
        float mean = sum / channels;
        
        // Step 2: Compute Variance
        float variance_sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            float diff = d_input[row_offset + c] - mean;
            variance_sum += diff * diff;
        }
        float variance = variance_sum / channels;
        float rsqrt_std = rsqrtf(variance + 1e-5f); // Kepler intrinsic fast reciprocal square root
        
        // Step 3: Standardize, Scale (Gamma), and Shift (Beta)
        for (int c = 0; c < channels; ++c) {
            int idx = row_offset + c;
            d_output[idx] = ((d_input[idx] - mean) * rsqrt_std) * d_gamma[c] + d_beta[c];
        }
    }
}
"""

# ============================================================================
# 4. GENERAL 2D MATRIX MULTIPLICATION WITH BIAS KERNEL
# ============================================================================
MATMUL_2D_KERNEL = """
extern "C" {
    __global__ void matmul_2d_kernel(
        const float* __restrict__ d_A,         // Input matrix: shape (M, K) [e.g. Activated Tokens]
        const float* __restrict__ d_B,         // Weight matrix: shape (K, N) [e.g. Linear Projection Weights]
        const float* __restrict__ d_bias,      // Bias vector: shape (N)
        float* __restrict__ d_C,               // Output matrix: shape (M, N)
        const int M, const int N, const int K
    ) {
        int row = blockIdx.y * blockDim.y + threadIdx.y; // M index
        int col = blockIdx.x * blockDim.x + threadIdx.x; // N index
        
        if (row >= M || col >= N) return;
        
        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            accum += d_A[row * K + k] * d_B[k * N + col];
        }
        
        // Fuse the Linear Bias Addition step directly during compute writeout
        d_C[row * N + col] = accum + d_bias[col];
    }
}
"""

# ============================================================================
# 5. BATCHED STRIDED MATRIX MULTIPLICATION (Multi-Head Attention Engine)
# ============================================================================
MATMUL_STRIDED_KERNEL = """
extern "C" {
    __global__ void matrix_multiply_strided_kernel(
        const float* __restrict__ d_A,         // Batch tensor A: shape (B, NH, M, K)
        const float* __restrict__ d_B,         // Batch tensor B: shape (B, NH, K, N)
        float* __restrict__ d_C,               // Output batch tensor: shape (B, NH, M, N)
        const int M, const int N, const int K,
        const int stride_A,                    // Offset bytes to jump between internal head blocks
        const int stride_B,
        const int stride_C
    ) {
        int batch_head = blockIdx.z;           // Combines Batch and Number of Heads indices
        int row = blockIdx.y * blockDim.y + threadIdx.y; // M index
        int col = blockIdx.x * blockDim.x + threadIdx.x; // N index
        
        if (row >= M || col >= N) return;
        
        // Relocate execution pointers to the current head workspace context
        const float* A_head = d_A + (batch_head * stride_A);
        const float* B_head = d_B + (batch_head * stride_B);
        float* C_head = d_C + (batch_head * stride_C);
        
        float accum = 0.0f;
        for (int k = 0; k < K; ++k) {
            accum += A_head[row * K + k] * B_head[k * N + col];
        }
        
        C_head[row * N + col] = accum;
    }
}
"""

# ============================================================================
# 6. FUSED CAUSAL MASK & SOFTMAX KERNEL (Autoregressive Attention Enforcer)
# ============================================================================
CAUSAL_SOFTMAX_KERNEL = """
extern "C" {
    __global__ void causal_softmax_kernel(
        float* __restrict__ d_scores,          // Raw attention matrix: shape (B * NH * T, T)
        const int total_rows,                  // B * NH * T
        const int T,                           // Max Sequence Length
        const float scale_factor               // 1.0 / sqrt(head_dim)
    ) {
        int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (row_idx >= total_rows) return;
        
        int offset = row_idx * T;
        int current_token_pos = row_idx % T;   // Tracks causal step location inside the time-series matrix
        
        // Pass 1: Scale raw attention score and apply strict Causal Lower-Triangular Masking
        float max_val = -1e20f; // Track max for stable softmax normalization math
        for (int t = 0; t < T; ++t) {
            if (t > current_token_pos) {
                // Future tokens are masked out to prevent cheating
                d_scores[offset + t] = -1e20f; 
            } else {
                d_scores[offset + t] *= scale_factor;
                if (d_scores[offset + t] > max_val) {
                    max_val = d_scores[offset + t];
                }
            }
        }
        
        // Pass 2: Compute Exponentials & Sum denominators safely
        float sum = 0.0f;
        for (int t = 0; t < T; ++t) {
            if (t <= current_token_pos) {
                d_scores[offset + t] = __expf(d_scores[offset + t] - max_val); // Kepler fast native exp intrinsic
                sum += d_scores[offset + t];
            } else {
                d_scores[offset + t] = 0.0f;
            }
        }
        
        // Pass 3: Complete division probability assignment
        float inv_sum = 1.0f / (sum + 1e-9f);
        for (int t = 0; t < T; ++t) {
            if (t <= current_token_pos) {
                d_scores[offset + t] *= inv_sum;
            }
        }
    }
}
"""

# ============================================================================
# 7. HIGH-SPEED ELEMENT-WISE ACTIVATION KERNEL (ReLU/GELU Alternative)
# ============================================================================
ACTIVATION_KERNEL = """
extern "C" {
    __global__ void activation_kernel(
        float* __restrict__ d_data,            // Target intermediate layer array
        const int total_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;
        
        // Fast in-place Rectified Linear Activation for our FFN layer
        float val = d_data[idx];
        d_data[idx] = val > 0.0f ? val : 0.0f;
    }
}
"""

# ============================================================================
# 8. FUSED DROPOUT FORWARD KERNEL (With Inline XORShift PRNG)
# ============================================================================
DROPOUT_FORWARD_KERNEL = """
extern "C" {
    __global__ void dropout_forward_kernel(
        const float* __restrict__ d_input,    // Input tensor
        float* __restrict__ d_output,          // Scaled output tensor
        unsigned char* __restrict__ d_mask,    // 1-Byte Mask output to save VRAM: shape (N)
        const int total_elements,
        const float dropout_prob,              // e.g. 0.1f
        const unsigned int seed                // Dynamic per-epoch step seed
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;

        // Establish an isolated PRNG state sequence per thread
        unsigned int x = seed + idx;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        float rand_val = (float)x / (float)4294967295U; // Normalize to [0.0, 1.0]

        float scale = 1.0f / (1.0f - dropout_prob);

        if (rand_val < dropout_prob) {
            d_mask[idx] = 0;
            d_output[idx] = 0.0f;
        } else {
            d_mask[idx] = 1;
            d_output[idx] = d_input[idx] * scale; // Inverted dropout scaling
        }
    }
}
"""

# ============================================================================
# 9. DROPOUT BACKWARD KERNEL
# ============================================================================
DROPOUT_BACKWARD_KERNEL = """
extern "C" {
    __global__ void dropout_backward_kernel(
        const float* __restrict__ d_dOut,         // Incoming upstream gradient
        float* __restrict__ d_dIn,                // Downstream gradient out
        const unsigned char* __restrict__ d_mask, // Saved forward mask
        const int total_elements,
        const float dropout_prob
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;

        float scale = 1.0f / (1.0f - dropout_prob);
        // Route gradients strictly back through active channels
        d_dIn[idx] = (d_mask[idx] == 1) ? d_dOut[idx] * scale : 0.0f;
    }
}
"""

# ============================================================================
# 10. RECTIFIED LINEAR ACTIVATION (ReLU) BACKWARD KERNEL
# ============================================================================
RELU_BACKWARD_KERNEL = """
extern "C" {
    __global__ void relu_backward_kernel(
        const float* __restrict__ d_dOut,          // Upstream gradient
        const float* __restrict__ d_forward_act,   // Saved forward post-activation state
        float* __restrict__ d_dIn,                 // Resulting gradient
        const int total_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;

        // If the forward element was active (>0), pass gradient; else, terminate gradient path
        d_dIn[idx] = (d_forward_act[idx] > 0.0f) ? d_dOut[idx] : 0.0f;
    }
}
"""

# ============================================================================
# 11. GENERAL 2D MATRIX MULTIPLICATION BACKWARD (W.R.T WEIGHTS)
# ============================================================================
MATMUL_BACKWARD_WEIGHTS_KERNEL = """
extern "C" {
    __global__ void matmul_backward_weights_kernel(
        const float* __restrict__ d_X,        // Forward Input activations: shape (M, K)
        const float* __restrict__ d_dOut,     // Upstream gradient tensor: shape (M, N)
        float* __restrict__ d_dW,             // Output Gradient weight matrix: shape (K, N)
        const int M, const int N, const int K
    ) {
        // Computes dW = X^T * dOut
        int row = blockIdx.y * blockDim.y + threadIdx.y; // K Index
        int col = blockIdx.x * blockDim.x + threadIdx.x; // N Index

        if (row >= K || col >= N) return;

        float accum = 0.0f;
        for (int m = 0; m < M; ++m) {
            accum += d_X[m * K + row] * d_dOut[m * N + col];
        }
        d_dW[row * N + col] = accum;
    }
}
"""

# ============================================================================
# 12. LAYER NORMALIZATION BACKWARD KERNEL (Thread-Per-Row, Fused Reductions)
# ============================================================================
LAYERNORM_BACKWARD_KERNEL = """
extern "C" {
    __global__ void layernorm_backward_kernel(
        const float* __restrict__ d_dOut,     // Upstream Gradient: (B*T, C)
        const float* __restrict__ d_input,    // Original forward input layer: (B*T, C)
        const float* __restrict__ d_gamma,    // LayerNorm Scale vector: (C)
        float* __restrict__ d_dIn,            // Output Downstream Gradient: (B*T, C)
        float* __restrict__ d_dGamma,         // Output Gamma Gradients: (C)
        float* __restrict__ d_dBeta,          // Output Beta Gradients: (C)
        const int num_rows,                   // B * T
        const int channels                    // C
    ) {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row >= num_rows) return;

        int row_offset = row * channels;

        // Recompute Mean and Variance on-the-fly to conserve precious VRAM bandwidth
        float sum = 0.0f;
        for (int c = 0; c < channels; ++c) sum += d_input[row_offset + c];
        float mean = sum / channels;

        float var_sum = 0.0f;
        for (int c = 0; c < channels; ++c) {
            float diff = d_input[row_offset + c] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / channels;
        float rsqrt_std = rsqrtf(variance + 1e-5f);

        // Reduction variables for row-wise gradients
        float dxhat_sum = 0.0f;
        float dxhat_x_sum = 0.0f;

        for (int c = 0; c < channels; ++c) {
            int idx = row_offset + c;
            float x_hat = (d_input[idx] - mean) * rsqrt_std;
            float dy = d_dOut[idx];

            // Safely accumulate global weight gradients via Kepler native Atomic Add
            atomicAdd(&d_dGamma[c], dy * x_hat);
            atomicAdd(&d_dBeta[c], dy);

            float dxhat = dy * d_gamma[c];
            dxhat_sum += dxhat;
            dxhat_x_sum += dxhat * x_hat;
        }

        // Distribute downstream standard normalization gradient to inputs
        for (int c = 0; c < channels; ++c) {
            int idx = row_offset + c;
            float x_hat = (d_input[idx] - mean) * rsqrt_std;
            float dxhat = d_dOut[idx] * d_gamma[c];
            
            d_dIn[idx] = (rsqrt_std / channels) * (channels * dxhat - dxhat_sum - x_hat * dxhat_x_sum);
        }
    }
}
"""

# ============================================================================
# 13. STATEFUL ADAMW OPTIMIZER WEIGHT UPDATE KERNEL
# ============================================================================
ADAMW_UPDATE_KERNEL = """
extern "C" {
    __global__ void adamw_update_kernel(
        float* __restrict__ d_weights,         // Weight matrix to optimize: shape (N)
        float* __restrict__ d_m,               // First Moment Vector History Array: shape (N)
        float* __restrict__ d_v,               // Second Moment Vector History Array: shape (N)
        const float* __restrict__ d_grads,     // Computed gradients for current step: shape (N)
        const float lr,                        // Learning rate (alpha)
        const float beta1,                     // e.g. 0.9f
        const float beta2,                     // e.g. 0.999f
        const float eps,                       // e.g. 1e-8f
        const float weight_decay,              // Decoupled regularization rate (e.g. 0.01f)
        const float bias_correction1,          // 1.0f - powf(beta1, step)
        const float bias_correction2,          // 1.0f - powf(beta2, step)
        const int total_elements
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= total_elements) return;

        float g = d_grads[idx];

        // 1. Apply strict AdamW decoupled weight decay step
        d_weights[idx] -= lr * weight_decay * d_weights[idx];

        // 2. Update biased first moment running average
        float m_t = beta1 * d_m[idx] + (1.0f - beta1) * g;
        d_m[idx] = m_t;

        // 3. Update biased second raw moment running average
        float v_t = beta2 * d_v[idx] + (1.0f - beta2) * (g * g);
        d_v[idx] = v_t;

        // 4. Compute bias-corrected adjustments
        float m_hat = m_t / bias_correction1;
        float v_hat = v_t / bias_correction2;

        // 5. Update parameter weights in VRAM
        d_weights[idx] -= (lr / (sqrtf(v_hat) + eps)) * m_hat;
    }
}
"""

# ============================================================================
# Kernel Registry Map
# ============================================================================
KERNELS = {
    "embedding_lookup_kernel": EMBEDDING_LOOKUP_KERNEL,
    "elementwise_add_kernel": ELEMENTWISE_ADD_KERNEL,
    "layernorm_kernel": LAYERNORM_KERNEL,
    "matmul_2d_kernel": MATMUL_2D_KERNEL,
    "matrix_multiply_strided_kernel": MATMUL_STRIDED_KERNEL,
    "causal_softmax_kernel": CAUSAL_SOFTMAX_KERNEL,
    "activation_kernel": ACTIVATION_KERNEL,
    
    # New Training and State Kernels
    "dropout_forward_kernel": DROPOUT_FORWARD_KERNEL,
    "dropout_backward_kernel": DROPOUT_BACKWARD_KERNEL,
    "relu_backward_kernel": RELU_BACKWARD_KERNEL,
    "matmul_backward_weights_kernel": MATMUL_BACKWARD_WEIGHTS_KERNEL,
    "layernorm_backward_kernel": LAYERNORM_BACKWARD_KERNEL,
    "adamw_update_kernel": ADAMW_UPDATE_KERNEL,
}
