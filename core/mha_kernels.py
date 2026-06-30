"""
Phase 3: GPU-Resident Multi-Head Attention Kernels (Score-Space / Projection-Space / Meta-Space).

Pure PyCUDA JIT kernel string. No nvcc build system, no ctypes, no C++ host wrappers.
Compiled directly via SourceModule(MHA_KERNELS_STRING) and dispatched from Python
(grid/block decided in Python, exactly like core/kernels.py).

Three-Algebra Memory Model (kept strictly separate, no unified indexing):
    Score-Space:      Scores, probs, dScores, dProbs -> [H, M, M]   idx = h*M*M + i*M + j
    Projection-Space: Q, K, V, dOut, dQ, dK, dV       -> [H, M, D]   idx = h*M*D + i*D + d
    Meta-Space:       row_max, row_sum                -> [H, M]      idx = h*M + i

Kernel inventory:
    1. matmul_score_kernel    -> Scores = (Q @ K^T) * scale                 (Score-Space output)
    2. matmul_proj_kernel     -> Out = Probs @ V                            (Projection-Space output)
    3. softmax_fused_forward  -> causal softmax in-place over Scores
    4. softmax_fused_backward -> VJP-form softmax backward (no Jacobian)
    5. matmul_grad_q_kernel   -> dQ = dScores @ K
       matmul_grad_k_kernel   -> dK = dScores^T @ Q (virtual transpose only)
    6. matmul_grad_v          -> dV = probs^T @ dOut (virtual transpose only)
    7. matmul_qkv_fused       -> Fused[H, M, 3D] = X[H, M, Din] @ W_qkv[Din, 3D]
    8. split_qkv_kernel       -> materializes contiguous Q/K/V[H, M, D] views out of
                                 Fused[H, M, 3D], so kernels 1-6 keep their existing
                                 contiguous Projection-Space stride assumptions instead
                                 of becoming offset/stride-aware (that is a deliberately
                                 deferred, separate optimization phase).
"""

MHA_KERNELS_STRING = r"""
extern "C" {

// ============================================================================
// KERNEL A: SCORE-SPACE MATMUL (Q @ K^T)
// Output:    Scores[H, M, M]   (Score-Space)
// Reduction: k = D             (Projection-Space depth)
// Index:     h*M*M + i*M + j
// ============================================================================
__global__ void matmul_score_kernel(
    const float* __restrict__ Q,       // [H, M, D]
    const float* __restrict__ K,       // [H, M, D]
    float* __restrict__ Scores,        // [H, M, M]
    const int H, const int M, const int D,
    const float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * M;
    if (idx >= total) return;

    int h = idx / (M * M);
    int rem = idx - h * (M * M);
    int i = rem / M;
    int j = rem - i * M;

    int q_base = h * M * D + i * D;
    int k_base = h * M * D + j * D;

    float acc = 0.0f;
    for (int k = 0; k < D; ++k) {
        acc += Q[q_base + k] * K[k_base + k];
    }

    Scores[h * M * M + i * M + j] = acc * scale;
}

// ============================================================================
// KERNEL B: PROJECTION-SPACE MATMUL (Probs @ V)
// Output:    Out[H, M, D]      (Projection-Space)
// Reduction: j = M             (Score-Space width)
// Index:     h*M*D + i*D + d
// ============================================================================
__global__ void matmul_proj_kernel(
    const float* __restrict__ A,       // [H, M, M] (Score-Space)
    const float* __restrict__ B,       // [H, M, D] (Projection-Space)
    float* __restrict__ Out,           // [H, M, D]
    const int H, const int M, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int i = rem / D;
    int d = rem - i * D;

    int a_row_base = h * M * M + i * M;
    int b_col_base = h * M * D + d;

    float acc = 0.0f;
    for (int j = 0; j < M; ++j) {
        acc += A[a_row_base + j] * B[b_col_base + j * D];
    }

    Out[h * M * D + i * D + d] = acc;
}

// ============================================================================
// KERNEL: FUSED CAUSAL SOFTMAX FORWARD (Score-Space, in-place)
// One block per (h, i) row. Reduction over j = M (causal: j <= i).
// Meta-Space index: h*M + i
// ============================================================================
__global__ void softmax_fused_forward(
    float* __restrict__ Scores,        // [H, M, M] in-place
    float* __restrict__ row_max,       // [H, M]
    float* __restrict__ row_sum,       // [H, M]
    const int H, const int M
) {
    extern __shared__ float shared_buf[];

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    int row_base = h * M * M + i * M;
    int valid_cols = i + 1; // causal mask: j <= i

    float thread_max = -1e30f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        float v = Scores[row_base + j];
        if (v > thread_max) thread_max = v;
    }
    shared_buf[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_buf[tid] = fmaxf(shared_buf[tid], shared_buf[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_buf[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        float exp_v = expf(Scores[row_base + j] - max_val);
        Scores[row_base + j] = exp_v;
        thread_sum += exp_v;
    }
    shared_buf[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_buf[tid] += shared_buf[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_buf[0];
    __syncthreads();

    if (tid == 0) {
        row_max[h * M + i] = max_val;
        row_sum[h * M + i] = sum_val;
    }

    float inv_sum = 1.0f / sum_val;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        Scores[row_base + j] *= inv_sum;
    }
    for (int j = valid_cols + tid; j < M; j += blockDim.x) {
        Scores[row_base + j] = 0.0f;
    }
}

// ============================================================================
// KERNEL: FUSED SOFTMAX BACKWARD (Score-Space, VJP form)
// dProbs[h,i,j] = probs[h,i,j] * (dScores[h,i,j] - row_sum[h,i])
// row_sum[h,i] = sum_k(dScores[h,i,k] * probs[h,i,k])  (computed in-kernel)
// One block per (h, i) row.
// ============================================================================
__global__ void softmax_fused_backward(
    const float* __restrict__ dScores, // [H, M, M]  (upstream: dL/dProbs_postsoftmax)
    const float* __restrict__ probs,   // [H, M, M]
    float* __restrict__ row_sum,       // [H, M]  (scratch + output)
    float* __restrict__ dProbs,        // [H, M, M]  (output: dL/dScores_presoftmax)
    const int H, const int M
) {
    extern __shared__ float shared_buf[];

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    int row_base = h * M * M + i * M;
    int valid_cols = i + 1; // causal mask: j <= i

    float thread_dot = 0.0f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        thread_dot += dScores[row_base + j] * probs[row_base + j];
    }
    shared_buf[tid] = thread_dot;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_buf[tid] += shared_buf[tid + stride];
        }
        __syncthreads();
    }
    float dot_val = shared_buf[0];
    __syncthreads();

    if (tid == 0) {
        row_sum[h * M + i] = dot_val;
    }

    for (int j = tid; j < valid_cols; j += blockDim.x) {
        dProbs[row_base + j] = probs[row_base + j] * (dScores[row_base + j] - dot_val);
    }
    for (int j = valid_cols + tid; j < M; j += blockDim.x) {
        dProbs[row_base + j] = 0.0f;
    }
}

// ============================================================================
// KERNEL: GRADIENT W.R.T. Q   (dQ = dScores_presoftmax @ K)
// Output:    dQ[H, M, D]      (Projection-Space)
// Reduction: j = M            (Score-Space width)
// Index:     h*M*D + i*D + d
// ============================================================================
__global__ void matmul_grad_q_kernel(
    const float* __restrict__ dProbs,  // [H, M, M] (Score-Space, dL/dScores_presoftmax)
    const float* __restrict__ K,       // [H, M, D] (Projection-Space)
    float* __restrict__ dQ,            // [H, M, D]
    const int H, const int M, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int i = rem / D;
    int d = rem - i * D;

    int row_base = h * M * M + i * M;
    int k_col_base = h * M * D + d;

    float acc = 0.0f;
    for (int j = 0; j < M; ++j) {
        acc += dProbs[row_base + j] * K[k_col_base + j * D];
    }

    dQ[h * M * D + i * D + d] = acc;
}

// ============================================================================
// KERNEL: GRADIENT W.R.T. K   (dK = dScores_presoftmax^T @ Q, virtual transpose only)
// Output:    dK[H, M, D]      (Projection-Space)
// Reduction: i = M            (Score-Space height, accessed transposed)
// Index:     h*M*D + j*D + d
// ============================================================================
__global__ void matmul_grad_k_kernel(
    const float* __restrict__ dProbs,  // [H, M, M] (Score-Space, dL/dScores_presoftmax)
    const float* __restrict__ Q,       // [H, M, D] (Projection-Space)
    float* __restrict__ dK,            // [H, M, D]
    const int H, const int M, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int j = rem / D;
    int d = rem - j * D;

    int q_col_base = h * M * D + d;
    int scores_base = h * M * M;

    float acc = 0.0f;
    for (int i = 0; i < M; ++i) {
        acc += dProbs[scores_base + i * M + j] * Q[q_col_base + i * D];
    }

    dK[h * M * D + j * D + d] = acc;
}

// ============================================================================
// KERNEL: GRADIENT W.R.T. V   (dV = probs^T @ dOut, virtual transpose only)
// Output:    dV[H, M, D]      (Projection-Space)
// Reduction: i = M            (Score-Space height, accessed transposed)
// Index:     h*M*D + j*D + d
// ============================================================================
__global__ void matmul_grad_v(
    const float* __restrict__ probs,   // [H, M, M] (Score-Space)
    const float* __restrict__ dOut,    // [H, M, D] (Projection-Space)
    float* __restrict__ dV,            // [H, M, D]
    const int H, const int M, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int j = rem / D;
    int d = rem - j * D;

    int dout_col_base = h * M * D + d;
    int probs_base = h * M * M;

    float acc = 0.0f;
    for (int i = 0; i < M; ++i) {
        acc += probs[probs_base + i * M + j] * dOut[dout_col_base + i * D];
    }

    dV[h * M * D + j * D + d] = acc;
}

// ============================================================================
// KERNEL: FUSED QKV PROJECTION
// Output:    Fused[H, M, 3D]   (Score-Space-agnostic projection buffer)
// Reduction: d = Din           (input feature depth)
// Index:     h*M*3D + i*3D + k    (k in [0, 3D))
// Shared weight matrix W_qkv[Din, 3D] is broadcast across all heads.
// ============================================================================
__global__ void matmul_qkv_fused(
    const float* __restrict__ X,       // [H, M, Din]
    const float* __restrict__ W_qkv,   // [Din, 3D]
    float* __restrict__ Fused,         // [H, M, 3D]
    const int H, const int M, const int Din, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int three_d = 3 * D;
    int total = H * M * three_d;
    if (idx >= total) return;

    int h = idx / (M * three_d);
    int rem = idx - h * (M * three_d);
    int i = rem / three_d;
    int k = rem - i * three_d;

    int x_base = h * M * Din + i * Din;

    float acc = 0.0f;
    for (int d = 0; d < Din; ++d) {
        acc += X[x_base + d] * W_qkv[d * three_d + k];
    }

    Fused[h * M * three_d + i * three_d + k] = acc;
}

// ============================================================================
// KERNEL: SPLIT FUSED QKV INTO CONTIGUOUS PROJECTION-SPACE TENSORS
// Materializes Q/K/V[H, M, D] (each individually contiguous, matching the
// stride assumptions of every other kernel in this file) out of the packed
// Fused[H, M, 3D] projection buffer. This is the explicit "logical tensor"
// boundary: nothing downstream needs to know fusion ever happened.
// ============================================================================
__global__ void split_qkv_kernel(
    const float* __restrict__ Fused,   // [H, M, 3D]
    float* __restrict__ Q,             // [H, M, D]
    float* __restrict__ K,             // [H, M, D]
    float* __restrict__ V,             // [H, M, D]
    const int H, const int M, const int D
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * M * D;
    if (idx >= total) return;

    int h = idx / (M * D);
    int rem = idx - h * (M * D);
    int i = rem / D;
    int d = rem - i * D;

    int three_d = 3 * D;
    int fused_row_base = h * M * three_d + i * three_d;
    int out_idx = h * M * D + i * D + d;

    Q[out_idx] = Fused[fused_row_base + d];
    K[out_idx] = Fused[fused_row_base + D + d];
    V[out_idx] = Fused[fused_row_base + 2 * D + d];
}

}
"""
