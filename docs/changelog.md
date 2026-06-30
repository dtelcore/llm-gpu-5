# Changelog

## 2026-07-01

- **Phase 2C fused attention:** merged QKt + causal softmax + PV into `fused_attention_forward_kernel`; wired into `MHAController.forward()` and production `MultiHeadAttention` in `model/gpt.py`. Layer 6 parity test in `test_mha_golden_model.py`.
- **MHA integration:** GPU-resident forward with `split_heads_kernel` / `merge_heads_kernel`; integration oracle in `test_mha_integration_parity.py`; smoke test `smoke_train_mha_integration.py`.
- **Label smoothing:** `RunConfig.label_smoothing` (default 0.1); fused loss kernel in `core/loss.py`; threaded through `auto_train.py` and `train.py`; tests in `test_label_smoothing_loss.py`.
- **Embedding scale fix:** `small` preset and `last_run_config.json` bumped to `embedding_dim=64` for 4096-token vocab boundary stability.
- **Regime Controller:** `regime_monitor.py` (BIS/TTR/RCI/Phi + bounded controller); wired into `auto_train.py` train loop; telemetry in `output/regime_metrics_latest.jsonl`; docs in `docs/REGIME_CONTROLLER.md`.
- **AutoTrain cost cards:** per-preset VRAM/tok/s/collapse-risk estimates in model selection menu.
- **Trajectory scoring:** `regime_policy_optimizer.py` + `test_regime_policy_optimizer.py` for offline policy evaluation.
- **Docs refresh:** updated index, workflow, architecture, quick reference, logging guides for all of the above.

## 2026-06-30

- Added GPU-resident FeedForward backward pass (`core/ops.py`: `GELUBackward`, `ReduceSumAxis0`, `MatMulBackwardInput`), removing the CPU/NumPy round-trip from the FFN gradient path. Legacy CPU backward kept as a togglable oracle (`use_cpu_backward`) for parity verification via `test_ffn_gpu_backward.py`.
- Added Phase 3 Multi-Head Attention CUDA kernels (`core/mha_kernels.py`): causal score matmul, fused causal softmax forward/backward (VJP form), projection-space `dQ`/`dK`/`dV` gradient kernels, a fused QKV projection kernel, and a split kernel that materializes contiguous Q/K/V tensors out of the fused buffer.
- Added `core/mha_ops.py` (`MHAController`): the production orchestration layer wiring fused QKV projection -> split -> causal attention scores -> fused softmax -> output projection into a single `forward()` call.
- Added `test_mha_golden_model.py`: a three-layer correctness gate -- (1) kernel-level parity against an independent NumPy reference, (2) fused-QKV representation equivalence against a NumPy triple-projection reference, (3) controller execution-path parity (hand-assembled kernel sequence vs. `MHAController.forward()`) -- required to pass before any tiling/fusion optimization work.
- Added a cosine warmup learning-rate scheduler and centralized GPU kernel compilation into a single shared `SourceModule` pass.
- Added `tqdm` progress bars to the tokenizer build path (`tokenizer/hybrid_tokenizer.py`) and corpus file loading (`corpus_utils.py`).
- Added `gpu_timing.py` for lightweight step-level GPU timing instrumentation.
- Added `gitpush.py`: a status -> add -> commit -> push helper for `origin/main`.

## 2026-05-24

- Added reusable generation probes for tokenizer roundtrip, greedy decode, memorization-prefix, and first-step logits.
- Fixed the shared corpus path to preserve normalized newline structure instead of stripping it away.
- Changed training to sample batch windows across the corpus instead of reusing one fixed slice.
- Wired auto_train.py and train.py to emit probe reports after checkpoint saves.
- Updated auto_train.py to trigger milestone checkpoint + probe runs at 25%, 50%, 75%, and 100% of configured steps.
- Updated auto_train.py artifact naming to include model config token in log/checkpoint names.
- Raised the default context length to 128 when VRAM allows.
- Added a hybrid tokenizer vocabulary with character fallback, plus a larger default vocab cap.
- Added a held-out validation slice and validation loss/perplexity logging.
- Added sampled generation probes using top-p sampling and a mild repetition penalty.
- Performed docs-wide refresh for all Markdown files in docs.
- Updated commands and examples to current script names and output paths.
- Standardized checkpoint/log/cache paths under output.
- Replaced stale references to old artifact roots and obsolete workflow claims.
- Clarified training entry points: auto_train.py, train.py, pipeline.py, generate.py.
- Updated logging docs to reflect training_metrics compact line format.
- Updated FineWeb notes to match shared loader path data/fineweb_100mb.txt.

## Historical note

Older changelog details from previous milestones were condensed to keep this file aligned with current maintainable documentation goals.
