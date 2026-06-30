---
name: FFN GPU-Resident Backward Phase 2
overview: Make FeedForward.backward fully GPU-resident by adding three missing CUDA primitives (GELU backward, axis-0 reduction, transpose-B matmul), wiring a dual-path (use_cpu_backward) refactor in model/gpt.py with the old NumPy math kept as a CPU oracle, and fixing the Windows subprocess Unicode crash in train.py.
todos: []
isProject: false
---

# FFN GPU-Resident Backward (Phase 2)

## Step 1 — Windows subprocess UTF-8 fix (small, isolated)

In [train.py](train.py), both `subprocess.Popen` call sites (lines 378-379 and 385-386) launching `training_log_plotter.py`/`loss_landscape_plotter.py`: build `env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"` once near the top of the function, and pass `env=env` to all four `Popen(...)` calls. Fixes the `UnicodeEncodeError` observed in the Phase 1 verification run without touching any other environment variables.

## Step 2 — New GPU primitives

Add to [core/kernels.py](core/kernels.py) (registered in the `KERNELS` dict) and wrapped in [core/ops.py](core/ops.py):

1. `**gelu_backward_kernel**` — mirrors `relu_backward_kernel`'s structure (`core/kernels.py:313-328`) but implements the tanh-approximation GELU derivative currently computed in NumPy at [model/gpt.py:380-388](model/gpt.py):
  `dIn[idx] = dOut[idx] * (0.5*(1+tanh_z) + 0.5*x*sech2_z*dz_dx)`, recomputing `x`, `tanh_z`, `sech2_z`, `dz_dx` per-thread from the saved pre-activation. Wrapped as `GELUBackward` op class, same call signature shape as `ReLUBackward.__call__(gpu_dOut, gpu_forward_act, total_elements)`.
2. `**reduce_sum_axis0_kernel**` — generalizes the atomic-accumulation pattern already used for `dGamma`/`dBeta` in `layernorm_backward_kernel` (`core/kernels.py:393-404`) into a standalone kernel: thread-per-row, `atomicAdd(&d_output[c], d_input[row*channels+c])` for each column. Wrapped as `ReduceSumAxis0` op: `__call__(gpu_input, num_rows, channels)` → allocates + `memset_d8`-zeroes `gpu_output[channels]`, launches, returns pointer. Used for both `dProjB` (`[M,C]→[C]`) and `dFcB` (`[M,4C]→[4C]`).
3. `**matmul_backward_input_kernel**` — new transpose-B matmul: `dIn[m,k] = sum_n dOut[m,n] * W[k,n]`, given `dOut[M,N]` and `W[K,N]` (the weight's natural on-device shape — no transpose needed). Wrapped as `MatMulBackwardInput.__call__(gpu_dOut, gpu_W, M, N, K)` → `gpu_dIn[M,K]`. Used for both `dHidden = dOut @ c_proj_w^T` (`M,N=C,K=4C`) and `dIn = dHidden @ c_fc_w^T` (`M,N=4C,K=C`).

No changes to any existing kernel or op — purely additive.

## Step 3 — Dual-path `FeedForward` refactor in [model/gpt.py](model/gpt.py)

**Forward (`FeedForward.forward`, lines 309-344):** stop calling `cuda.memcpy_dtoh` for `cache_pre_act`; instead allocate it as a device buffer and `cuda.memcpy_dtod` from `gpu_hidden` before the in-place GELU runs (mirrors how `cache_input` is already cached). This buffer is consumed by both the CPU and GPU backward paths.

**Backward (`FeedForward.backward`, lines 346-401):** add `self.use_cpu_backward = False` in `__init__`. Branch:

```python
if getattr(self, 'use_cpu_backward', False):
    # existing D2H transfers + NumPy math, unchanged byte-for-byte
else:
    # GPU-resident path:
    gpu_dProjW = self.weight_bwd_op(self.cache_activated, gpu_dOut, M, C, C*4)        # MatMulBackwardWeights
    gpu_dProjB = self.reduce_op(gpu_dOut, M, C)                                        # ReduceSumAxis0
    gpu_dHidden = self.bwd_input_op(gpu_dOut, self.c_proj_w.gpu_weights, M, C, C*4)    # MatMulBackwardInput
    gpu_dPreAct = self.gelu_bwd_op(gpu_dHidden, self.cache_pre_act, M*C*4)             # GELUBackward
    gpu_dHidden.free()
    gpu_dFcW = self.weight_bwd_op(self.cache_input, gpu_dPreAct, M, C*4, C)
    gpu_dFcB = self.reduce_op(gpu_dPreAct, M, C*4)
    gpu_dIn = self.bwd_input_op(gpu_dPreAct, self.c_fc_w.gpu_weights, M, C*4, C)
    gpu_dPreAct.free()

    self.c_proj_w.set_or_accumulate_grads_from_gpu(gpu_dProjW, accumulate=accumulate); gpu_dProjW.free()
    self.c_proj_b.set_or_accumulate_grads_from_gpu(gpu_dProjB, accumulate=accumulate); gpu_dProjB.free()
    self.c_fc_w.set_or_accumulate_grads_from_gpu(gpu_dFcW, accumulate=accumulate); gpu_dFcW.free()
    self.c_fc_b.set_or_accumulate_grads_from_gpu(gpu_dFcB, accumulate=accumulate); gpu_dFcB.free()
    return gpu_dIn
```

This reuses the Phase 1 GPU-resident `set_or_accumulate_grads_from_gpu` (already implemented) for all four FFN parameters, eliminating their host round-trips too — not just the activation math. Intermediates are freed immediately after their consumer runs, matching `TransformerBlock.backward`'s existing lifecycle convention.

Instantiate the three new ops (`MatMulBackwardInput`, `ReduceSumAxis0`, `GELUBackward`) in `FeedForward.__init__` alongside the existing `self.weight_bwd_op`.

## Step 4 — CPU-oracle parity test harness

Add a small opt-in test (e.g. `test/test_core/test_ffn_gpu_backward.py` or a script-level check) that: builds a single `FeedForward` instance with fixed seed, runs forward once, runs `backward` once with `use_cpu_backward=True` to capture reference `dW`/`db`/`dIn` (downloaded to host), resets gradients, re-runs `backward` with `use_cpu_backward=False`, downloads the GPU-path results, and asserts `np.testing.assert_allclose(..., rtol=1e-4, atol=1e-5)` (FP32 tolerance, not bit-exact) between the two paths for `c_fc_w.gpu_grads`, `c_fc_b.gpu_grads`, `c_proj_w.gpu_grads`, `c_proj_b.gpu_grads`, and the returned `dIn`.

## Sequencing

1. Step 1 (trivial, independent) can land first.
2. Step 2 (new primitives) lands before Step 3 — verify each new kernel compiles (the `SourceModule` single-pass compile in `core/ops.py:25-37` will fail loudly at import time if there's a syntax error).
3. Step 3 lands with `use_cpu_backward` defaulting to `False` but easy to flip to `True` for a quick regression check against Phase 1's behavior.
4. Step 4 (parity harness) validates Step 3 before considering the GPU path production-default.

[{"id": "subprocess-utf8-fix", "content": "Fix Windows subprocess UnicodeEncodeError via env=os.environ.copy() + PYTHONIOENCODING=utf-8 in train.py"}, {"id": "new-gpu-primitives", "content": "Add gelu_backward_kernel, reduce_sum_axis0_kernel, matmul_backward_input_kernel to core/kernels.py and wrap as GELUBackward, ReduceSumAxis0, MatMulBackwardInput ops in core/ops.py"}, {"id": "ffn-dual-path-refactor", "content": "Refactor FeedForward in model/gpt.py: keep cache_pre_act GPU-resident, add use_cpu_backward flag, implement GPU-resident backward path using new primitives plus existing MatMulBackwardWeights and set_or_accumulate_grads_from_gpu"}, {"id": "ffn-parity-test", "content": "Add CPU-vs-GPU backward parity test using np.testing.assert_allclose to validate the new GPU path against the retained NumPy oracle"}]  
  
  
Those final guardrails are the difference between a successful refactor and three days of debugging silent memory corruption. You hit the nail on the head regarding LLM coding agents: if you don't explicitly forbid them from freeing cached activations, they will almost always insert a `.free()` the moment a tensor is consumed by its first downstream kernel, destroying the backprop graph for earlier layers.

Adding the CPU vs. GPU FFN-specific micro-timer to the test harness is also a fantastic call. We need to see the exact millisecond ROI of this specific rewrite before blending it into the full step times.

Here is the final, hardened Phase 2 prompt. It explicitly locks down the kernel naming conventions, enforces arbitrary strides for the reduction, strictly protects the cached tensors, and adds the loss/timing diagnostics to the oracle test.

Markdown

```
# Role & Context
You are an expert PyCUDA Core Engineer. We are executing Phase 2: The GPU-Resident FeedForward Rewrite for a custom-written LLM training engine targeting a highly constrained GT 730 (Compute Capability 3.5).

**CRITICAL RULES:** 1. This is a pure PyCUDA codebase, NOT PyTorch. Do not write PyTorch APIs. 
2. Memory Lifecycle: Only free explicitly defined ephemeral intermediate tensors (e.g., `gpu_dHidden`, `gpu_dPreAct`). **NEVER free cached activations** (`cache_pre_act`, `cache_input`, `cache_activated`) during the backward pass; they must survive for the graph's lifetime.

# Implementation Plan

Execute the following four steps sequentially.

## Step 1: Windows Subprocess Encoding Fix
In `train.py`, locate the `subprocess.Popen` calls for `training_log_plotter.py` and `loss_landscape_plotter.py`. Safely force UTF-8 compliance:
```python
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
# Ensure env=env is passed to all Popen execution blocks

```

## Step 2: Implement Missing GPU Primitives

Open `core/kernels.py` and `core/ops.py`. Implement these three specific primitives.

*Naming Strictness:* Maintain strict semantic separation between `MatMulBackwardWeights` (computes $dW$) and `MatMulBackwardInput` (computes $dX$).

1. `gelu_backward_kernel`
  - **Location:** Add to `core/kernels.py` and register.
  - **Math:** Tanh-approximation derivative.
    $$dIn[idx] = dOut[idx] \cdot (0.5 \cdot (1 + \tanh(z)) + 0.5 \cdot x \cdot \text{sech}^2(z) \cdot \frac{dz}{dx})$$
  - **Constraint:** The input `x` must be the pristine, unmodified pre-activation linear output.
  - **Wrapper:** Expose in `core/ops.py` as `GELUBackward` with `__call__(gpu_dOut, gpu_forward_act, total_elements)`.
2. `reduce_sum_axis0_kernel`
  - **Location:** Add to `core/kernels.py` and register.
  - **Behavior:** Column-sum-over-rows atomic reduction. It must accept an arbitrary stride parameter (`stride_row`) rather than assuming a contiguous $M \times C$ layout.
  - **Wrapper:** Expose in `core/ops.py` as `ReduceSumAxis0` with `__call__(gpu_input, num_rows, channels, stride_row)`. Handle output allocation via `cuda.mem_alloc` and zero-initialize using `cuda.memset_d8`.
3. `matmul_backward_input_kernel`
  - **Location:** Add to `core/kernels.py` and register.
  - **Behavior:** Computes input gradients $dX = dOut \times W^T$ without physically transposing $W$.
    $$dX[m, k] = \sum_{n} dOut[m, n] \cdot W[k, n]$$
  - **Wrapper:** Expose in `core/ops.py` as `MatMulBackwardInput` with `__call__(gpu_dOut, gpu_W, M, N, K)`.

## Step 3: Dual-Path FeedForward Refactor

Open `model/gpt.py` and rewrite the `FeedForward` layer.

1. **Initialization:** Instantiate the three new operations. Add `self.use_cpu_backward = False`.
2. **Forward Pass:** Remove `cuda.memcpy_dtoh` for `cache_pre_act`. Allocate a device buffer and `memcpy_dtod` to cache the pristine pre-activation state in VRAM *before* the in-place GELU operation modifies it.
3. **Backward Pass Structural Branch:**

Python

```
if getattr(self, 'use_cpu_backward', False):
    # Retain the exact original D2H transfers and NumPy math block unchanged
else:
    # 1. Compute c_proj gradients
    gpu_dProjW = self.weight_bwd_op(self.cache_activated, gpu_dOut, M, C, C*4)
    gpu_dProjB = self.reduce_op(gpu_dOut, M, C, C) # passing C as stride
    gpu_dHidden = self.bwd_input_op(gpu_dOut, self.c_proj_w.gpu_weights, M, C, C*4)
    
    # 2. Backprop through GELU
    gpu_dPreAct = self.gelu_bwd_op(gpu_dHidden, self.cache_pre_act, M*C*4)
    gpu_dHidden.free() # Ephemeral intermediate
    
    # 3. Compute c_fc gradients
    gpu_dFcW = self.weight_bwd_op(self.cache_input, gpu_dPreAct, M, C*4, C)
    gpu_dFcB = self.reduce_op(gpu_dPreAct, M, C*4, C*4) # passing C*4 as stride
    gpu_dIn = self.bwd_input_op(gpu_dPreAct, self.c_fc_w.gpu_weights, M, C*4, C)
    gpu_dPreAct.free() # Ephemeral intermediate

    # 4. In-place GPU Accumulation
    self.c_proj_w.set_or_accumulate_grads_from_gpu(gpu_dProjW, accumulate=accumulate); gpu_dProjW.free()
    self.c_proj_b.set_or_accumulate_grads_from_gpu(gpu_dProjB, accumulate=accumulate); gpu_dProjB.free()
    self.c_fc_w.set_or_accumulate_grads_from_gpu(gpu_dFcW, accumulate=accumulate); gpu_dFcW.free()
    self.c_fc_b.set_or_accumulate_grads_from_gpu(gpu_dFcB, accumulate=accumulate); gpu_dFcB.free()
    
    return gpu_dIn

```

## Step 4: Diagnostic Oracle Testing Harness

Create a separate verification script (e.g., `test_ffn_gpu_backward.py`).

- **Execution:** Initialize `FeedForward` with a deterministic seed. Run forward, then run backward with `use_cpu_backward=True`. Reset gradients. Toggle to `False`, re-run.
- **Diagnostics:** Wrap the CPU backward pass and the GPU backward pass in `time.perf_counter()` to explicitly print the execution time difference (e.g., `CPU Time: X ms, GPU Time: Y ms`).
- **Validation:** 1. Calculate and assert identical forward scalar loss between both methods.
  2. Assert numerical correctness for $dW$, $db$, and $dX_{\text{input}}$:
  Python
  ```
  np.testing.assert_allclose(gpu_result, cpu_result, rtol=1e-4, atol=1e-5)

  ```

```

***

Pass this to Cursor in Agent/Composer mode. Once it implements the changes and you run that new `test_ffn_gpu_backward.py` script, let me know what the millisecond delta looks like between the CPU and GPU pathways!

```

