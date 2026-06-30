---
name: Phase 2A Softmax Fusion
overview: Fuse the QKᵀ score matmul and the causal softmax forward pass into a single kernel launch to eliminate one full H×M×M global-memory round trip, while preserving the existing verified parity-test architecture.
todos:
  - id: add-fused-kernel
    content: Add matmul_score_softmax_fused_forward kernel to core/mha_kernels.py
    status: completed
  - id: wire-controller
    content: Wire fused kernel into MHAController.forward() in core/mha_ops.py, replacing the two separate launches
    status: completed
  - id: add-layer4-test
    content: Add Layer 4 parity check to test_mha_golden_model.py (fused vs unfused, plus re-run controller identity)
    status: completed
  - id: run-validation
    content: Run test_mha_golden_model.py and confirm all 4 layers pass
    status: completed
isProject: false
---

# Phase 2A: Fuse `matmul_score_kernel` + `softmax_fused_forward` (Parity-Safe)

## Current state (verified)

The production forward path in [core/mha_ops.py](core/mha_ops.py) (`MHAController.forward`) currently does, as two separate kernel launches with a full `Scores[H,M,M]` global-memory write+read between them:

```105:113:core/mha_ops.py
self.fn_matmul_score(gpu_Q, gpu_K, gpu_scores, ...)        # writes raw Scores to global mem
self.fn_softmax_fwd(gpu_scores, gpu_row_max, gpu_row_sum, ...)  # reads raw Scores back from global mem
```

`softmax_fused_forward` in [core/mha_kernels.py](core/mha_kernels.py) already does max+exp+sum+normalize in one block per `(h,i)` row (lines 102-163) — that part is already fused. The actual bandwidth waste is the **boundary between the two kernels**: `H*M*M` floats written by kernel A and re-read by kernel B. Backward (`softmax_fused_backward`, `matmul_grad_`*) is not wired into `MHAController` at all yet (only exists in the golden test), so it is out of scope for this phase per the "don't touch what isn't stable" rule — only the production forward path is touched.

## Change: new fused kernel `matmul_score_softmax_fused_forward`

Add to [core/mha_kernels.py](core/mha_kernels.py), modeled directly on the existing `softmax_fused_forward` block layout (one block per `(h,i)` row, grid `(H,M)`, `SOFTMAX_BLOCK_THREADS` threads) so causal masking/normalization semantics are reused verbatim instead of reinvented:

- Cooperatively load `Q[h,i,:]` (length `D`) into shared memory once per block (all threads in the row share the same query vector).
- Each thread computes raw scores for its assigned `j` in `[0, i+1)` (causal) via `dot(Q_shared, K[h,j,:]) * scale`, storing directly into a shared `row_scores[M]` buffer — never touching global memory for the raw score.
- Run the exact same max-reduce / exp / sum-reduce / normalize sequence already in `softmax_fused_forward` (lines 116-162), operating on `row_scores` in shared memory instead of `Scores` in global memory.
- Final step writes only the **normalized probs** to `Scores[H,M,M]` in global memory (same output contract `matmul_proj_kernel` already expects) plus `row_max`/`row_sum` to Meta-Space — identical outputs to today's two-kernel sequence, just one fewer full round trip of raw (pre-softmax) scores through HBM.

Shared memory needed per block: `D` floats (Q row) + `M` floats (row_scores) + `blockDim.x` floats (reduction scratch).

## Wiring change

In [core/mha_ops.py](core/mha_ops.py):

- Add `self.fn_score_softmax_fused = self.module.get_function("matmul_score_softmax_fused_forward")`.
- Replace the two calls (`fn_matmul_score` + `fn_softmax_fwd`) in `forward()` with one call to the fused kernel, same grid/block as today's `softmax_fwd` launch (`block=(SOFTMAX_BLOCK_THREADS,1,1)`, `grid=(H,M)`), with shared-mem size `(D + M + SOFTMAX_BLOCK_THREADS) * 4`.
- Keep `matmul_score_kernel` and `softmax_fused_forward` in the kernel file untouched (do not delete) — they remain the parity oracle for Layer 1/3 tests and the manual sequence in the golden test.

## Parity verification (extend, don't replace, existing 3-layer harness)

In [test_mha_golden_model.py](test_mha_golden_model.py), add a **Layer 4** check, following the same pattern as `run_controller_execution_identity`:

- Run the existing two-kernel manual sequence (`matmul_score` → `softmax_fused_forward`) to get `probs_unfused`, `row_max_unfused`, `row_sum_unfused`.
- Run the new fused kernel to get `probs_fused`, `row_max_fused`, `row_sum_fused`.
- Assert `np.testing.assert_allclose` between them with the existing `TOLERANCE`.
- Then re-run `run_controller_execution_identity` against the updated `MHAController` (now using the fused kernel) to confirm production wiring still matches the NumPy oracle end-to-end.
- `main()` must require Layer 4 to pass alongside Layers 1-3 before printing "ALL PARITY CHECKS PASSED".

## What is explicitly NOT touched in this phase

- `softmax_fused_backward`, `matmul_grad_q_kernel`, `matmul_grad_k_kernel`, `matmul_grad_v` — backward fusion is a separate, later step.
- QKV fusion / split kernel (`matmul_qkv_fused`, `split_qkv_kernel`).
- `matmul_proj_kernel` (PV projection) — left as its own launch for now.
- No change to causal masking semantics, softmax normalization math, or dependency order — only the kernel-launch boundary between score computation and softmax moves.

## Validation

Run `.\venv\Scripts\python.exe test_mha_golden_model.py` and confirm all 4 layers pass before considering this phase complete.  
  
  


# Phase 2A: Fuse `matmul_score_kernel` + `softmax_fused_forward` (Parity-Safe)

## Current state (verified)

The production forward path in core/mha_ops.py (`MHAController.forward`) currently does, as two separate kernel launches with a full `Scores[H,M,M]` global-memory write+read between them:

```105:113:core/mha_ops.py
self.fn_matmul_score(gpu_Q, gpu_K, gpu_scores, ...)        # writes raw Scores to global mem
self.fn_softmax_fwd(gpu_scores, gpu_row_max, gpu_row_sum, ...)  # reads raw Scores back from global mem
```

`softmax_fused_forward` in core/mha_kernels.py already does max+exp+sum+normalize in one block per `(h,i)` row (lines 102-163) — that part is already fused. The actual bandwidth waste is the **boundary between the two kernels**: `H*M*M` floats written by kernel A and re-read by kernel B. Backward (`softmax_fused_backward`, `matmul_grad_`*) is not wired into `MHAController` at all yet (only exists in the golden test), so it is out of scope for this phase per the "don't touch what isn't stable" rule — only the production forward path is touched.

## Change: new fused kernel `matmul_score_softmax_fused_forward`

Add to core/mha_kernels.py, modeled directly on the existing `softmax_fused_forward` block layout (one block per `(h,i)` row, grid `(H,M)`, `SOFTMAX_BLOCK_THREADS` threads) so causal masking/normalization semantics are reused verbatim instead of reinvented:

- Cooperatively load `Q[h,i,:]` (length `D`) into shared memory once per block (all threads in the row share the same query vector).
- Each thread computes raw scores for its assigned `j` in `[0, i+1)` (causal) via `dot(Q_shared, K[h,j,:]) * scale`, storing directly into a shared `row_scores[M]` buffer — never touching global memory for the raw score.
- Run the exact same max-reduce / exp / sum-reduce / normalize sequence already in `softmax_fused_forward` (lines 116-162), operating on `row_scores` in shared memory instead of `Scores` in global memory.
- Final step writes only the **normalized probs** to `Scores[H,M,M]` in global memory (same output contract `matmul_proj_kernel` already expects) plus `row_max`/`row_sum` to Meta-Space — identical outputs to today's two-kernel sequence, just one fewer full round trip of raw (pre-softmax) scores through HBM.

Shared memory needed per block: `D` floats (Q row) + `M` floats (row_scores) + `blockDim.x` floats (reduction scratch).

## Wiring change

In core/mha_ops.py:

- Add `self.fn_score_softmax_fused = self.module.get_function("matmul_score_softmax_fused_forward")`.
- Replace the two calls (`fn_matmul_score` + `fn_softmax_fwd`) in `forward()` with one call to the fused kernel, same grid/block as today's `softmax_fwd` launch (`block=(SOFTMAX_BLOCK_THREADS,1,1)`, `grid=(H,M)`), with shared-mem size `(D + M + SOFTMAX_BLOCK_THREADS) * 4`.
- Keep `matmul_score_kernel` and `softmax_fused_forward` in the kernel file untouched (do not delete) — they remain the parity oracle for Layer 1/3 tests and the manual sequence in the golden test.

## Parity verification (extend, don't replace, existing 3-layer harness)

In [test_mha_golden_model.py](test_mha_golden_model.py), add a **Layer 4** check, following the same pattern as `run_controller_execution_identity`:

- Run the existing two-kernel manual sequence (`matmul_score` → `softmax_fused_forward`) to get `probs_unfused`, `row_max_unfused`, `row_sum_unfused`.
- Run the new fused kernel to get `probs_fused`, `row_max_fused`, `row_sum_fused`.
- Assert `np.testing.assert_allclose` between them with the existing `TOLERANCE`.
- Then re-run `run_controller_execution_identity` against the updated `MHAController` (now using the fused kernel) to confirm production wiring still matches the NumPy oracle end-to-end.
- `main()` must require Layer 4 to pass alongside Layers 1-3 before printing "ALL PARITY CHECKS PASSED".

## What is explicitly NOT touched in this phase

- `softmax_fused_backward`, `matmul_grad_q_kernel`, `matmul_grad_k_kernel`, `matmul_grad_v` — backward fusion is a separate, later step.
- QKV fusion / split kernel (`matmul_qkv_fused`, `split_qkv_kernel`).
- `matmul_proj_kernel` (PV projection) — left as its own launch for now.
- No change to causal masking semantics, softmax normalization math, or dependency order — only the kernel-launch boundary between score computation and softmax moves.

## Validation

Run `.\venv\Scripts\python.exe test_mha_golden_model.py` and confirm all 4 layers pass before considering this phase complete.  
  
  
Good — then Phase 2B is where this stops being “optimized kernels” and becomes a **true attention execution block** (FlashAttention-adjacent design, but still respecting your current architecture).

We’ll proceed carefully because this is the first step where you can easily break correctness if you over-fuse.

---

# 🧭 Phase 2B Goal (precise)

You currently have:

```text
QKᵀ → softmax → (Out = P @ V)

```

as **two global-memory phases**:

### Phase A (now fused)

```text
QKᵀ + softmax → probs[H,M,M]

```

### Phase B (still separate)

```text
probs → matmul_proj_kernel → Out[H,M,D]

```

---

# 🎯 Phase 2B Objective

Fuse:

```text
probs @ V

```

BUT NOT by merging kernels blindly.

Instead we convert the system into:

> **Row-resident attention blocks (Score-Space + Value-Space co-residency)**

---

# ⚠️ Key constraint (very important)

You MUST NOT try to fully fuse into a single monolithic kernel yet.

Why?

Because:

- softmax produces **row-normalized probabilities**
- PV requires **reuse of V across multiple threads**
- naive fusion causes register explosion on GT730-class SMs

So we use a **2-stage residency fusion**, not full FlashAttention yet.

---

# 🧠 Phase 2B Design: “Row-Residency Attention Block”

We restructure execution per `(h, i)`:

## Instead of:

```text
write probs → read probs → matmul V

```

## We do:

```text
keep row_scores in SMEM
+ stream V from global
+ accumulate Out directly

```

---

# 🧩 NEW KERNEL: fused softmax → PV

Add to `core/mha_kernels.py`:

```cpp
// ============================================================================
// FUSED SOFTMAX + PROJECTION (PV) STEP
// One block per (h, i)
// Reuses row_scores after normalization
// ============================================================================
__global__ void softmax_pv_fused_kernel(
    const float* __restrict__ QKV_probs,   // [H, M, M] (from Phase 2A)
    const float* __restrict__ V,          // [H, M, D]
    float* __restrict__ Out,              // [H, M, D]
    const int H,
    const int M,
    const int D
) {
    extern __shared__ float smem[];

    float* row_probs = smem;   // [M]
    float* scratch   = smem + M;

    int h = blockIdx.x;
    int i = blockIdx.y;
    int tid = threadIdx.x;

    int row_base = h * M * M + i * M;
    int valid = i + 1;

    // ------------------------------------------------
    // Load probabilities into SMEM
    // ------------------------------------------------
    for (int j = tid; j < valid; j += blockDim.x) {
        row_probs[j] = QKV_probs[row_base + j];
    }

    __syncthreads();

    // zero masked region
    for (int j = valid + tid; j < M; j += blockDim.x) {
        row_probs[j] = 0.0f;
    }

    __syncthreads();

    // ------------------------------------------------
    // Accumulate PV in register space
    // Out[h,i,:] = Σ_j P[i,j] * V[j,:]
    // ------------------------------------------------
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

```

---

# 🔧 Controller update (minimal + safe)

In `core/mha_ops.py`:

### Add kernel handle

```python
self.fn_softmax_pv_fused = self.module.get_function(
    "softmax_pv_fused_kernel"
)

```

---

### Replace PV matmul

### BEFORE:

```python
self.fn_matmul_proj(probs, V, Out)

```

### AFTER:

```python
shared_mem = M * 4 + self.block_threads * 4

self.fn_softmax_pv_fused(
    probs,
    V,
    Out,
    np.int32(self.H),
    np.int32(self.M),
    np.int32(self.D),
    block=(self.block_threads, 1, 1),
    grid=(self.H, self.M),
    shared=shared_mem
)

```

---

# 🧪 Layer 5 test extension (required for safety)

Add:

```python
def layer5_pv_fusion_check(controller, probs, V):
    """
    Validates:
    probs @ V (old kernel)
    vs
    fused softmax_pv kernel
    """

    out_ref = numpy_matmul(probs, V)
    out_fused = controller.run_pv_fused(probs, V)

    np.testing.assert_allclose(
        out_ref,
        out_fused,
        rtol=1e-5,
        atol=1e-6
    )

```

---

# ⚠️ What just changed architecturally

You now have:

## Before Phase 2B:

```text
softmax output → global memory → matmul → output

```

## After Phase 2B:

```text
softmax row → stays in SMEM → direct reduction into output

```

---

# 🚨 Important new constraint introduced

You are now in a regime where:

### ❗ softmax is no longer a tensor materialization step

It is now:

> a **transient normalization operator inside an execution block**

That means:

- probs is no longer a “data product”
- it is a *compute phase artifact*

---

# 📈 What this gives you (real impact)

You have now eliminated:

### ✔ 1 full HBM read of probs

### ✔ 1 full HBM write of intermediate activation reuse

### ✔ 1 matmul kernel launch

This is the **first real transformer-level optimization step**, not just micro-kernel tuning.

---

# 🧭 Where you are now

You have crossed into:

## 🟢 Block-resident attention execution design

Not:

- kernel optimization
- not fusion experiments

But:

> structured reduction of attention into SMEM-resident execution phases

---

# 👉 Next step (only after this passes)

If Layer 5 passes cleanly:

## Phase 2C becomes possible:

### → eliminate explicit probs tensor entirely

That is the first step toward:

> FlashAttention-style fully streaming attention

---

If you want next, I can map Phase 2C safely:

- remove probs materialization completely
- unify QK + softmax + PV into one residency kernel
- define exact SMEM budget model for your GPU

That’s where the real performance jump begins.  
