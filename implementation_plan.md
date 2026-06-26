# PyCUDA GPT Inference Optimization for GT 730

## Problem Statement

The current `GenerationSession.generate()` loop has severe per-token overhead:

1. **Full forward pass per token** — runs *all* T tokens through the entire model even though only the last-position logits are used. With ctx=128, by token 100 it's doing 100× the minimal work.
2. **Reallocates `gpu_input` every iteration** — `cuda.mem_alloc()` + `.free()` per step.
3. **H2D + D2H copies every step** — full `host_input` pushed to GPU, full `(T, vocab_size)` logits pulled back (even though only the last row is needed).
4. **`free_forward_caches()` called every iteration** — destroys reusable intermediate VRAM (TransformerBlock pre-LN caches, FFN input/activation caches, attention caches), forcing reallocation on the next step.
5. **No instrumentation** — no visibility into per-token latency, transfer sizes, or kernel time.

### Architecture Context

- **Model**: 1-layer GPT, `embedding_dim=128`, `num_heads=4`, `attention_impl=strided`, `ctx=128`
- **Hardware**: GT 730 (Kepler GK208, Compute 3.5), 1 GB VRAM, PCIe 2.0 x8
- **Attention**: When `attention_impl=strided`, full multi-head attention runs on GPU (QKV projection → strided matmul → causal softmax → output projection). This is the expensive path.
- **Identity fallback**: When `attention_impl=identity`, attention is just a memcpy (no QKV/softmax). The checkpoint uses `strided`.

### Why Not KV Cache?

A true KV cache (caching K,V from prior positions to avoid recomputation on each step) would be the standard optimization. However, this model's attention path does heavy CPU↔GPU bouncing:

- `MultiHeadAttention.forward()` (lines 456-533 of gpt.py): After the GPU QKV matmul, it copies the *entire* QKV result back to host with `cuda.memcpy_dtoh`, does numpy reshaping/transposing on CPU, then copies Q/K/V back to GPU for the strided matmul. This is fundamentally a hybrid CPU/GPU implementation.
- Implementing a true incremental KV cache would require refactoring `MultiHeadAttention.forward()` to keep K,V tensors persistent on GPU and only compute Q for the new token — a significant architectural change that risks correctness.

**Decision**: We focus on safe, correctness-first overhead reductions that don't require modifying the model internals.

## Proposed Changes

### Phase 1: Reduce Per-Token Memory Churn in `generate.py`

#### [MODIFY] [generate.py](file:///c:/dev/llm%20gpu%205/generate.py)

**1a. Pre-allocate `gpu_input` and reuse across iterations**

Currently (line 535):
```python
gpu_input = cuda.mem_alloc(host_input.nbytes)
```
is called every token, paired with `gpu_input.free()` on line 539.

Change: Allocate a single `gpu_input` buffer of size `max_context * 4` bytes before the loop, reuse it with `cuda.memcpy_htod()` each step, and free it once after the loop.

**1b. Only copy back the last row of logits**

Currently (lines 541-543):
```python
host_logits_matrix = np.empty((current_t_length, vocab_size), dtype=np.float32)
cuda.memcpy_dtoh(host_logits_matrix, gpu_logits)
```
copies `T × vocab_size × 4` bytes every step (e.g., 128×vocab×4), but only `host_logits_matrix[-1, :]` is used.

Change: Use `cuda.memcpy_dtoh()` with a byte offset to copy only the last row:
```python
last_row_offset = (current_t_length - 1) * vocab_size * 4
host_last_logits = np.empty(vocab_size, dtype=np.float32)
cuda.memcpy_dtoh(host_last_logits, int(gpu_logits) + last_row_offset)
```

This reduces D2H transfer from `T × V × 4` bytes to just `V × 4` bytes per step (e.g., from ~260 KB to ~2 KB for a 500-token vocab at T=128).

**1c. Stop freeing `gpu_logits` inside the loop — defer to after**

The `gpu_logits` allocation returned by `model.forward()` is of size `T × V × 4` bytes. Currently freed (line 543) and reallocated by the model on every step. Since the model's `forward()` always allocates a fresh result via `matmul_op`, we must free the previous iteration's result. But we can at least avoid the extra `gpu_input` alloc/free.

**1d. `free_forward_caches()` — still required each step**

After analysis: `free_forward_caches()` releases VRAM occupied by intermediate activations (pre-LN caches in TransformerBlocks, FFN cache_input/cache_activated, etc.). On a 1 GB GT 730, failing to free these would cause VRAM exhaustion. We **must** keep calling `free_forward_caches()` each step. However, we can remove the `gpu_input.free()` call since we're reusing the buffer.

> [!IMPORTANT]
> `free_forward_caches()` cannot be removed — the TransformerBlock allocates `cache_input_ln1` and `cache_input_ln2` on each forward call (lines 679, 690 of gpt.py), and the FFN allocates `cache_input` and `cache_activated`. These would accumulate across steps without freeing.

---

### Phase 2: Add Performance Instrumentation

#### [MODIFY] [generate.py](file:///c:/dev/llm%20gpu%205/generate.py)

Add a `GenerationMetrics` dataclass/dict that collects:
- **Per-token wall-clock latency** (using `time.perf_counter()`)
- **H2D bytes per step** (input token IDs: `T * 4` bytes)
- **D2H bytes per step** (last logits row: `vocab_size * 4` bytes, was `T * vocab_size * 4`)
- **Total kernel time** (using `cuda.Event` pairs around `model.forward()`)
- **Cache reuse flag** (whether `gpu_input` buffer was reused vs reallocated)

Print a summary after generation completes:
```
[PERF] Generated 1024 tokens in X.XXs (Y.Y tok/s)
[PERF] Avg per-token: Z.Z ms (kernel: K.K ms, H2D: A.A KB, D2H: B.B KB)
[PERF] gpu_input buffer reused: yes
```

---

### Phase 3: Correctness Verification Checks

#### [MODIFY] [generate.py](file:///c:/dev/llm%20gpu%205/generate.py)

**3a. Tokenizer/checkpoint vocab mismatch guard**

`build_generation_model()` already checks `tokenizer.vocab_size != config.vocab_size` (line 344). This is correct and sufficient.

**3b. Causal mask verification**

The causal mask is correct. The `causal_softmax_kernel` (kernels.py line 192) uses `row_idx % T` as the current token position and masks out `t > current_token_pos`. This correctly implements a lower-triangular mask within each attention head's score matrix. No bug here.

**3c. No hidden CPU fallback in model.forward()**

When `attention_impl=strided`, the `MultiHeadAttention.forward()` does execute GPU kernels (QKV projection via `matmul_op`, strided matmul, causal softmax). However, it round-trips through CPU for reshape/transpose operations (lines 456-470, 500-527). This is the designed behavior for this Kepler-targeted implementation — not a "hidden fallback" but intentional CPU-assisted layout transformations. No change needed, but we'll document this in the instrumentation output.

---

### Phase 4: Kepler-Specific Micro-Optimizations

#### [MODIFY] [generate.py](file:///c:/dev/llm%20gpu%205/generate.py)

**4a. Allocate `gpu_zero_bias` once for the `lm_head` projection**

In `GPTModel.forward()` (gpt.py line 826), a zero-bias buffer is allocated and freed every forward call:
```python
gpu_zero_bias = cuda.mem_alloc(self.config.vocab_size * 4)
cuda.memset_d8(gpu_zero_bias, 0, self.config.vocab_size * 4)
...
gpu_zero_bias.free()
```

#### [MODIFY] [model/gpt.py](file:///c:/dev/llm%20gpu%205/model/gpt.py)

Move to a persistent allocation in `GPTModel.__init__()`:
```python
self._gpu_zero_bias = cuda.mem_alloc(config.vocab_size * 4)
cuda.memset_d8(self._gpu_zero_bias, 0, config.vocab_size * 4)
```
And use `self._gpu_zero_bias` in `forward()`, removing the per-call alloc/free. Free it in the model cleanup.

**4b. Pre-allocate position index buffer in TokenEmbedding**

In `TokenEmbedding.forward()` (gpt.py line 194), position indices are allocated, H2D-copied, and freed every call. For inference with fixed `T ≤ max_len`, we can pre-allocate and cache this.

> [!NOTE]
> This optimization saves one `cuda.mem_alloc()` + `cuda.memcpy_htod()` + `.free()` per forward call — small but measurable on PCIe 2.0.

---

### Phase 5: Apply to Probe Functions Too

#### [MODIFY] [generate.py](file:///c:/dev/llm%20gpu%205/generate.py)

The one-shot logits extraction in `run_generation_probes()` (lines 308-318) has the same pattern: full logits copy when only the last row is needed. Apply the same last-row-only optimization there.

---

## Summary of All Changes

| File | Change | Impact |
|------|--------|--------|
| [generate.py](file:///c:/dev/llm%20gpu%205/generate.py) | Pre-allocate `gpu_input` buffer, reuse across loop | Eliminates N `mem_alloc`+`free` calls |
| [generate.py](file:///c:/dev/llm%20gpu%205/generate.py) | Copy only last logits row (D2H) | Reduces D2H by ~T× per step |
| [generate.py](file:///c:/dev/llm%20gpu%205/generate.py) | Add `GenerationMetrics` instrumentation | Visibility into per-token overhead |
| [generate.py](file:///c:/dev/llm%20gpu%205/generate.py) | Optimize probe function logits copy | Consistency |
| [model/gpt.py](file:///c:/dev/llm%20gpu%205/model/gpt.py) | Persistent `_gpu_zero_bias` in GPTModel | Eliminates per-forward alloc/free |
| [model/gpt.py](file:///c:/dev/llm%20gpu%205/model/gpt.py) | Cache position indices in TokenEmbedding | Eliminates per-forward alloc/H2D/free |
| [generate.py](file:///c:/dev/llm%20gpu%205/generate.py) | Free persistent buffers in `free_model()` | Clean VRAM teardown |

## What We Are NOT Changing (and Why)

1. **No KV cache** — Would require rewriting `MultiHeadAttention.forward()` to persist K,V on GPU and only compute new-position Q. The current implementation round-trips through CPU for reshaping, making incremental caching architecturally infeasible without a major rewrite that risks correctness.
2. **Sampling stays on CPU** — `sample_next_token()` uses numpy. Moving to GPU would require a custom CUDA sampling kernel (temperature scaling + top-k/top-p filtering + multinomial draw). Not trivial, and the CPU overhead is small relative to the forward pass.
3. **No `forward()` internals changed** — All attention/FFN/LayerNorm paths stay identical. We only optimize the bookkeeping around them.
4. **`free_forward_caches()` stays per-step** — VRAM would accumulate otherwise on 1 GB.

## Verification Plan

### Automated Tests
```bash
python generate.py --checkpoint .\output\checkpoints\gpt_250000steps_9p0e-05lr_ctx128_base_128d_1l_20260620_102909.best.npz --max_new_tokens 40 --temperature 0.0
```
Run greedy decode (temperature=0.0) before and after changes. Output must be **identical** since we're not changing any math, only memory management and transfer sizes.

### Manual Verification
1. Compare greedy decode output before/after for identical character sequences
2. Verify `[PERF]` summary appears in output with sane values
3. Check that D2H bytes per step = `vocab_size * 4` (not `T * vocab_size * 4`)
4. Monitor VRAM with `nvidia-smi` to confirm no leaks during 1024-token generation
