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
    9. matmul_score_softmax_fused_forward -> Phase 2A: fuses kernels 1 and 3 into a
                                 single SMEM-resident kernel, removing the raw-score
                                 HBM round trip between QKt and softmax. Numerically
                                 identical to running kernel 1 then kernel 3.
    10. softmax_pv_fused_kernel -> Phase 2B: row-residency fusion of the normalized
                                 probs row into shared memory, reused across all D
                                 output columns for the PV projection. Numerically
                                 identical to matmul_proj_kernel (kernel 2).
    11. split_heads_kernel / merge_heads_kernel -> Phase 3 Integration: GPU-side
                                 layout adapters between the real model's single
                                 shared c_attn_w/c_proj_w projections ([B*T,3C] /
                                 [B*T,C]) and the H=B*NH,M=T,D=HD convention that
                                 kernels 9/10 expect. Replace CPU-side numpy
                                 split/reshape/transpose round trips in
                                 model/gpt.py's MultiHeadAttention.forward().
    12. fused_attention_forward_kernel -> Phase 2C: merges kernels 9 and 10 into a
                                 single launch. The normalized probs row computed by
                                 the softmax step never leaves shared memory before
                                 being consumed by the PV reduction; it is still
                                 written to Scores[H,M,M] in global memory (write-only,
                                 never re-read) purely so MultiHeadAttention.backward()'s
                                 existing CPU-side cache_attn_weights requirement keeps
                                 working unchanged. Numerically identical to running
                                 kernel 9 then kernel 10.
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
// KERNEL: FUSED QKt SCORE MATMUL + CAUSAL SOFTMAX FORWARD (Score-Space, SMEM-resident)
// One block per (h, i) row. Eliminates the raw-score HBM round trip between
// matmul_score_kernel and softmax_fused_forward: scores are computed directly
// into shared memory and reduced there, with only the final normalized probs
// (plus row_max/row_sum) ever written to global memory. Math and causal
// semantics are identical to matmul_score_kernel -> softmax_fused_forward.
// ============================================================================
__global__ void matmul_score_softmax_fused_forward(
    const float* __restrict__ Q,       // [H, M, D]
    const float* __restrict__ K,       // [H, M, D]
    float* __restrict__ Scores,        // [H, M, M] (output: normalized probs)
    float* __restrict__ row_max,       // [H, M]
    float* __restrict__ row_sum,       // [H, M]
    const int H, const int M, const int D,
    const float scale
) {
    extern __shared__ float smem[];
    float* q_row = smem;            // [D]
    float* row_scores = smem + D;   // [M]
    float* shared_buf = smem + D + M; // [blockDim.x]

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    int valid_cols = i + 1; // causal mask: j <= i

    int q_base = h * M * D + i * D;
    for (int d = tid; d < D; d += blockDim.x) {
        q_row[d] = Q[q_base + d];
    }
    __syncthreads();

    for (int j = tid; j < valid_cols; j += blockDim.x) {
        int k_base = h * M * D + j * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += q_row[d] * K[k_base + d];
        }
        row_scores[j] = acc * scale;
    }
    __syncthreads();

    float thread_max = -1e30f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        float v = row_scores[j];
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
        float exp_v = expf(row_scores[j] - max_val);
        row_scores[j] = exp_v;
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
    int row_base = h * M * M + i * M;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        Scores[row_base + j] = row_scores[j] * inv_sum;
    }
    for (int j = valid_cols + tid; j < M; j += blockDim.x) {
        Scores[row_base + j] = 0.0f;
    }
}

// ============================================================================
// KERNEL: FUSED SOFTMAX-PROBS + PV PROJECTION (Phase 2B, row-residency)
// One block per (h, i) row. Loads the normalized probs row for (h,i) into
// shared memory once, then reuses it across all D output columns instead of
// matmul_proj_kernel's per-(h,i,d) global reads of the same probs row.
// Mathematically identical to matmul_proj_kernel: Out[h,i,d] = sum_j P[h,i,j] * V[h,j,d].
// ============================================================================
__global__ void softmax_pv_fused_kernel(
    const float* __restrict__ QKV_probs,   // [H, M, M] (normalized probs, from Phase 2A)
    const float* __restrict__ V,           // [H, M, D]
    float* __restrict__ Out,               // [H, M, D]
    const int H,
    const int M,
    const int D
) {
    extern __shared__ float smem[];

    float* row_probs = smem;   // [M]
    float* scratch   = smem + M; // [blockDim.x]

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;

    int row_base = h * M * M + i * M;
    int valid = i + 1;

    for (int j = tid; j < valid; j += blockDim.x) {
        row_probs[j] = QKV_probs[row_base + j];
    }
    __syncthreads();

    for (int j = valid + tid; j < M; j += blockDim.x) {
        row_probs[j] = 0.0f;
    }
    __syncthreads();

    for (int d = 0; d < D; d++) {
        float acc = 0.0f;

        for (int j = tid; j < valid; j += blockDim.x) {
            float p = row_probs[j];
            float v = V[h * M * D + j * D + d];
            acc += p * v;
        }

        scratch[tid] = acc;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                scratch[tid] += scratch[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            Out[h * M * D + i * D + d] = scratch[0];
        }
        __syncthreads();
    }
}

// ============================================================================
// KERNEL: FUSED ATTENTION FORWARD (QKt + Softmax + PV, single launch, Phase 2C)
// One block per (h, i) row. Merges matmul_score_softmax_fused_forward and
// softmax_pv_fused_kernel: the normalized probs row computed by the softmax
// step is consumed directly by the PV reduction from shared memory --
// it is never re-read from global memory. Scores[H,M,M] is still written
// (write-only) purely to preserve the existing cache_attn_weights contract
// that MultiHeadAttention.backward() depends on. Math is identical to
// running matmul_score_softmax_fused_forward then softmax_pv_fused_kernel.
// ============================================================================
__global__ void fused_attention_forward_kernel(
    const float* __restrict__ Q,       // [H, M, D]
    const float* __restrict__ K,       // [H, M, D]
    const float* __restrict__ V,       // [H, M, D]
    float* __restrict__ Scores,        // [H, M, M] (output: normalized probs, cache-only)
    float* __restrict__ Out,           // [H, M, D] (output)
    float* __restrict__ row_max,       // [H, M]
    float* __restrict__ row_sum,       // [H, M]
    const int H, const int M, const int D,
    const float scale
) {
    extern __shared__ float smem[];
    float* q_row = smem;              // [D]
    float* row_scores = smem + D;     // [M]  (raw -> exp -> normalized probs, in place)
    float* scratch = smem + D + M;    // [blockDim.x]  (reused: softmax reduction, then PV reduction)

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;
    int valid_cols = i + 1; // causal mask: j <= i

    // --- QKt (Score-Space), computed directly into shared memory ---
    int q_base = h * M * D + i * D;
    for (int d = tid; d < D; d += blockDim.x) {
        q_row[d] = Q[q_base + d];
    }
    __syncthreads();

    for (int j = tid; j < valid_cols; j += blockDim.x) {
        int k_base = h * M * D + j * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) {
            acc += q_row[d] * K[k_base + d];
        }
        row_scores[j] = acc * scale;
    }
    __syncthreads();

    // --- Causal softmax (max-reduce, exp, sum-reduce, normalize), in shared memory ---
    float thread_max = -1e30f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        float v = row_scores[j];
        if (v > thread_max) thread_max = v;
    }
    scratch[tid] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = scratch[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        float exp_v = expf(row_scores[j] - max_val);
        row_scores[j] = exp_v;
        thread_sum += exp_v;
    }
    scratch[tid] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            scratch[tid] += scratch[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = scratch[0];
    __syncthreads();

    if (tid == 0) {
        row_max[h * M + i] = max_val;
        row_sum[h * M + i] = sum_val;
    }

    float inv_sum = 1.0f / sum_val;
    for (int j = tid; j < valid_cols; j += blockDim.x) {
        row_scores[j] *= inv_sum;
    }
    for (int j = valid_cols + tid; j < M; j += blockDim.x) {
        row_scores[j] = 0.0f;
    }
    __syncthreads();

    // Write normalized probs to global -- cache-only (backward's cache_attn_weights),
    // never re-read by this kernel.
    int row_base = h * M * M + i * M;
    for (int j = tid; j < M; j += blockDim.x) {
        Scores[row_base + j] = row_scores[j];
    }
    __syncthreads();

    // --- PV reduction, reading row_scores directly from shared memory (no global re-read) ---
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = tid; j < valid_cols; j += blockDim.x) {
            acc += row_scores[j] * V[h * M * D + j * D + d];
        }
        scratch[tid] = acc;
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                scratch[tid] += scratch[tid + s];
            }
            __syncthreads();
        }

        if (tid == 0) {
            Out[h * M * D + i * D + d] = scratch[0];
        }
        __syncthreads();
    }
}

// ============================================================================
// KERNEL: SPLIT HEADS (real-model layout adapter, Phase 3 Integration)
// Replaces the CPU np.split + reshape + transpose(0,2,1,3) in
// model/gpt.py's MultiHeadAttention.forward(). Consumes the single shared
// c_attn_w projection output QKV[B*T, 3*C] (C = NH*HD, columns ordered
// [Q(C) | K(C) | V(C)], each C-block itself ordered as NH consecutive HD
// chunks) and produces contiguous Q/K/V[B*NH, T, HD] -- exactly the
// H=B*NH, M=T, D=HD layout matmul_score_softmax_fused_forward and
// softmax_pv_fused_kernel expect. No transpose is needed for K: those
// kernels consume K untransposed.
// ============================================================================
__global__ void split_heads_kernel(
    const float* __restrict__ QKV,     // [B*T, 3*C]
    float* __restrict__ Q,             // [B*NH, T, HD]
    float* __restrict__ K,             // [B*NH, T, HD]
    float* __restrict__ V,             // [B*NH, T, HD]
    const int B, const int T, const int NH, const int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int C = NH * HD;
    int total = B * T * NH * HD;
    if (idx >= total) return;

    int hd = idx % HD;
    int nh = (idx / HD) % NH;
    int t  = (idx / (HD * NH)) % T;
    int b  = idx / (HD * NH * T);

    int qkv_row = b * T + t;
    int col = nh * HD + hd;
    int out_idx = (b * NH + nh) * T * HD + t * HD + hd;

    Q[out_idx] = QKV[qkv_row * 3 * C + col];
    K[out_idx] = QKV[qkv_row * 3 * C + C + col];
    V[out_idx] = QKV[qkv_row * 3 * C + 2 * C + col];
}

// ============================================================================
// KERNEL: MERGE HEADS (real-model layout adapter, Phase 3 Integration)
// Replaces the CPU reshape + transpose(0,2,1,3) + reshape merge step in
// model/gpt.py's MultiHeadAttention.forward(). Consumes the per-head
// context output ContextHeads[B*NH, T, HD] (from softmax_pv_fused_kernel)
// and produces Context[B*T, C] (C = NH*HD), ready for the c_proj_w matmul.
// ============================================================================
__global__ void merge_heads_kernel(
    const float* __restrict__ ContextHeads,  // [B*NH, T, HD]
    float* __restrict__ Context,             // [B*T, C]
    const int B, const int T, const int NH, const int HD
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * T * NH * HD;
    if (idx >= total) return;

    int hd = idx % HD;
    int nh = (idx / HD) % NH;
    int t  = (idx / (HD * NH)) % T;
    int b  = idx / (HD * NH * T);

    int in_idx = (b * NH + nh) * T * HD + t * HD + hd;
    int out_idx = (b * T + t) * (NH * HD) + nh * HD + hd;
    Context[out_idx] = ContextHeads[in_idx];
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
