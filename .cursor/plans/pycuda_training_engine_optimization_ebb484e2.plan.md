---
name: PyCUDA Training Engine Optimization
overview: Implement the 4-phase optimization sequence (profiling baseline, GPU-resident gradient accumulation, decoupled validation scheduling, pinned/async token transfers) for the custom PyCUDA GT730 training engine, in strict isolation order to avoid compounding bugs.
todos:
  - id: step1-profiling
    content: Add profile_gpu_timing/profile_memcpy flags to RunConfig, create gpu_timing.py with GpuProfiler/_ProfileZone, and instrument train.py step/forward/backward/optimizer zones plus VRAM sampling
    status: completed
  - id: step2-grad-accum
    content: Replace host NumPy add in set_or_accumulate_grads_from_gpu with ElementwiseAdd GPU kernel
    status: completed
  - id: step3-val-schedule
    content: Add val_interval to RunConfig and decouple validation cadence from logging cadence in train.py
    status: completed
  - id: step4-pinned-async
    content: Switch token batch transfer to pagelocked_empty + memcpy_htod_async on a dedicated stream
    status: completed
isProject: false
---

# PyCUDA Training Engine Optimization — Phase 1

## Context

The engine in [model/gpt.py](model/gpt.py), [core/ops.py](core/ops.py), and [train.py](train.py) is a hand-written PyCUDA stack (no PyTorch). The audit identified the highest-risk-to-reward fixes; we execute them in isolation, one at a time, each independently verifiable against numerical parity before moving to the next.

## Step 1 — Profiling Instrumentation (Baseline)

Locked-in implementation (finalized, copy-exact):

### 1a. Config flags

In [run_config.py](run_config.py), add to `RunConfig`:
* `profile_gpu_timing: bool = False`
* `profile_memcpy: bool = False`

### 1b. New `gpu_timing.py` (root directory)

A zero-allocation, deferred-synchronization `GpuProfiler`:
* Pre-allocates one `(start_evt, end_evt)` `cuda.Event` pair per tracked key at construction time (keys e.g. `step_gpu`, `forward`, `backward`, `optimizer`, `loss`, optionally `memcpy`).
* Exposes `.zone(key)` returning a `_ProfileZone` context manager that calls `start_evt.record()` on enter and `end_evt.record()` on exit, tracking which keys were touched this step in `active_keys_this_step`.
* `synchronize_and_accumulate()` records a single dedicated `sync_event`, synchronizes only on that event (never `Context.synchronize()`), then reads `start_evt.time_till(end_evt)` for each active key and accumulates into `metrics`.
* `get_averages_and_reset()` returns per-key averages over `accumulation_steps` and resets accumulators — intended to be called only on logged steps.

Full source (to be created verbatim):

```python
import pycuda.driver as cuda

class GpuProfiler:
    """
    Zero-allocation CUDA event profiler.
    Pre-allocates events to measure GPU execution time without altering the execution schedule.
    """
    def __init__(self, keys):
        self.events = {key: (cuda.Event(), cuda.Event()) for key in keys}
        self.metrics = {key: 0.0 for key in keys}
        self.active_keys_this_step = set()
        self.accumulation_steps = 0
        self.sync_event = cuda.Event()  # Dedicated event to wait for tracked work

    def zone(self, key):
        return _ProfileZone(self, key)

    def synchronize_and_accumulate(self):
        if not self.active_keys_this_step:
            return

        self.sync_event.record()
        self.sync_event.synchronize()

        for key in self.active_keys_this_step:
            start_evt, end_evt = self.events[key]
            elapsed_sec = start_evt.time_till(end_evt) / 1000.0
            self.metrics[key] += elapsed_sec

        self.active_keys_this_step.clear()
        self.accumulation_steps += 1

    def get_averages_and_reset(self):
        """Returns the average times per step and resets the accumulators."""
        if self.accumulation_steps == 0:
            return {k: 0.0 for k in self.metrics}

        avgs = {k: v / self.accumulation_steps for k, v in self.metrics.items()}
        for k in self.metrics:
            self.metrics[k] = 0.0
        self.accumulation_steps = 0
        return avgs


class _ProfileZone:
    def __init__(self, profiler, key):
        self.profiler = profiler
        self.key = key
        self.start_evt, self.end_evt = profiler.events[key]

    def __enter__(self):
        self.start_evt.record()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_evt.record()
        self.profiler.active_keys_this_step.add(self.key)
```

### 1c. Instrument [train.py](train.py)

* Instantiate `GpuProfiler` once, outside the loop, only `if run_config.profile_gpu_timing:` — keys `["step_gpu", "forward", "backward", "optimizer", "loss"]`, plus `"memcpy"` if `run_config.profile_memcpy`.
* Wrap the full step in a `step_gpu` zone; wrap `model.forward()`/loss inside the `micro_step` loop in `forward`/`loss` zones, `model.backward()` in a `backward` zone, and the `update_weights` call in an `optimizer` zone. If `profile_memcpy` is enabled, also wrap the existing `cuda.memcpy_htod` token-upload calls in a `memcpy` zone.
* Sample `cuda.mem_get_info()` at step start, immediately after the forward pass, and immediately after the backward pass (per micro-step granularity), to report `vram_base` / `vram_post_fw` / `vram_post_bw`.
* On `metrics.should_log_step(step)`: call `profiler.get_averages_and_reset()`, compute `total_gpu_time`, `tracked_gpu_parts` (sum of forward/backward/optimizer/loss), `untracked_gpu_time` (their difference), and `non_gpu_wall_time` (wall time minus total GPU time), and print a baseline summary line including VRAM deltas (see exact print format already drafted with the user — `[Step {step} Baseline] Wall: ... | Total GPU: ... | Non-GPU Wall Time: ...` etc.).

### Guardrails (must hold)

* `Context.synchronize()` is never called anywhere — only `profiler.sync_event.synchronize()` inside `synchronize_and_accumulate()`.
* When `run_config.profile_gpu_timing` is `False`, the loop must execute exactly as it does today, with zero added overhead (no event creation, no zones entered).
* Do not change forward/backward/gradient math, call order, or accumulation semantics in this step — purely additive instrumentation wrapped around existing calls.

## Step 2 — GPU-Resident Gradient Accumulation

In [model/gpt.py](model/gpt.py), `Parameter.set_or_accumulate_grads_from_gpu` (lines 134-143): replace the host download/add/upload with the existing `ElementwiseAdd` kernel from [core/ops.py](core/ops.py), operating in-place on `self.gpu_grads` using the already-GPU-resident `gpu_grads_ptr`.

- Add a shared `ElementwiseAdd()` instance to `GPTModel.__init__` and thread it down to `Parameter` calls (via a method parameter or a module-level singleton, matching existing patterns like `self.add_op` in other classes).
- `set_or_accumulate_grads` (host-array variant, used by `TokenEmbedding.backward`) is out of scope for this step (its inputs originate from host loops anyway — left for a later phase).
- Verify numerically: run a short fixed-seed training session before/after, confirm identical loss trajectory (same accumulation order matters for FP32 reproducibility — `ElementwiseAdd` does straightforward `target[idx] += source[idx]`, matching NumPy's `existing + host_new` elementwise order).

## Step 3 — Validation Scheduling

- Add `val_interval: int = 100` (or similar) field to `RunConfig` in [run_config.py](run_config.py).
- In [train.py](train.py), change the validation block (currently gated by `metrics.should_log_step(step)`, lines 254-260) to its own independent condition, e.g. `step % run_config.val_interval == 0 or step == total_steps`.
- Keep `grad_norm` computation tied to logging cadence as today (separate concern), but validation now runs on its own cadence so logging frequency doesn't force a validation forward pass every step.
- Update `TrainingMetrics.step_end` call so `val_loss`/`val_ppl` are simply `None` on non-validation steps (already supported via existing `Optional` handling in `_format_optional`).

## Step 4 — Pinned Memory & Async Transfers

- In [train.py](train.py), replace the plain NumPy host staging arrays for `input_tokens_sample`/`target_tokens_sample` with `cuda.pagelocked_empty(shape, dtype=np.int32)` buffers, filled in-place by `sample_token_batch` (or copied into post-hoc).
- Create a dedicated `cuda.Stream()` for H2D transfers; switch `cuda.memcpy_htod` calls for token batches to `cuda.memcpy_htod_async(..., stream=stream)`.
- Ensure the forward pass kernel launches either run on the same stream (default behavior is fine since PyCUDA kernel calls default to stream 0 unless given a `stream=` kwarg) or add an explicit `stream.synchronize()` before consuming the transferred buffer if kernels are issued on a different stream.
- This step has the highest hardware-dependent risk (stream ordering bugs are easy to introduce silently) — validate by diffing first-N-step loss values against a baseline run with identical seeds.

## Sequencing & Validation Notes

- Each step lands as its own isolated change; re-run a short fixed-seed training session (e.g. 50-100 steps on the existing minimal/tinystories config) and diff loss/perplexity trajectories against a pre-change baseline before proceeding to the next step.
- Step 1 (instrumentation) should land first and stay in place throughout, so Steps 2-4 each get an empirical before/after timing comparison using the same harness.