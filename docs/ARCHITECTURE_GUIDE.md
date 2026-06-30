# Architecture Guide

Last updated: 2026-06-30

Current state: GPU-resident FeedForward backward and Phase 3 Multi-Head Attention CUDA kernels have landed (see "GPU kernel layers" below), gated behind NumPy golden-model parity harnesses. Training uses sampled corpus windows, a held-out validation slice, and generation reuses the same tokenizer/corpus assumptions.

## High-level modules

- core/kernels.py: CUDA kernels for forward, backward, and optimizer updates
- core/ops.py: PyCUDA kernel wrappers and compilation/runtime glue (includes GPU-resident FFN backward: `GELUBackward`, `ReduceSumAxis0`, `MatMulBackwardInput`)
- core/mha_kernels.py: Phase 3 Multi-Head Attention CUDA kernel string (causal score matmul, fused softmax forward/backward, `dQ`/`dK`/`dV` gradients, fused QKV projection, QKV split)
- core/mha_ops.py: `MHAController`, the orchestration layer that wires the MHA kernels into a single forward pass
- core/loss.py: fused softmax cross-entropy
- model/gpt.py: GPTConfig, layers, model forward/backward/update/checkpoint logic
- tokenizer/tokenizer.py: CharacterGPTTokenizer compatibility wrapper around the hybrid tokenizer implementation
- corpus_utils.py: shared dataset loading, tokenizer construction, token cache (tqdm progress on file load)
- gpu_memory.py: pooled allocator support
- gpu_timing.py: lightweight step-level GPU timing instrumentation
- training_metrics.py: compact per-step and final metrics logging
- gitpush.py: status -> add -> commit -> push helper for publishing to `origin/main`

## GPU kernel correctness gates

Two standalone harnesses validate kernel correctness against independent NumPy
references before any kernel is wired into the live training path or before any
performance optimization (tiling, fusion) is attempted:

- `test_ffn_gpu_backward.py`: compares the GPU-resident FeedForward backward against
  the legacy CPU/NumPy backward oracle (`FeedForward.use_cpu_backward` toggle), and
  reports the wall-clock speedup.
- `test_mha_golden_model.py`: three-layer MHA gate --
  1. kernel-level parity (raw kernels vs. NumPy causal-attention forward/backward),
  2. fused-QKV representation equivalence (fused projection + split kernel vs. NumPy
     triple projection),
  3. controller execution-path parity (`MHAController.forward()` vs. an independently
     hand-assembled kernel call sequence), which catches "correct kernels, wrong
     wiring" bugs that kernel-only tests cannot see.

`core/mha_ops.py`'s `MHAController` is not yet wired into `model/gpt.py`'s attention
block; it currently exists as a standalone, fully-tested orchestration layer pending
integration.

## Runtime data flow

1. Text corpus loaded from dataset source
2. Tokenizer built from selected docs using a hybrid word/piece vocabulary with character fallback
3. Token matrix created or loaded from output/cache/tokenizer
4. Input and target slices staged to GPU
5. Model forward produces logits
6. Loss kernel computes loss and dLogits
7. Backward computes gradients
8. AdamW updates weights
9. Held-out validation slice is evaluated and logged
10. Checkpoint and logs written to output

## Checkpoint metadata behavior

generate.py infers model config from checkpoint arrays and optional metadata fields. It can also use output/last_run_config.json to resolve missing metadata such as num_heads/attention_impl.
