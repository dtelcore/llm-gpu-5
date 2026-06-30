# GT730 LLM Project

This repository is a Kepler/GT730-focused GPT training and generation project. The current baseline is the repaired data path: training samples across the corpus, the shared corpus loader preserves normalized newline structure, and generation uses the same tokenizer assumptions as training.

## Current State

- Training entry points: `auto_train.py`, `train.py`, `pipeline.py`
- Inference entry point: `generate.py`
- Tokenizer: hybrid word/piece tokenizer with character fallback, shared through `corpus_utils.py`
- Tokenizer OOV safety: UTF-8 byte fallback window (256 reserved IDs) so unseen validation/prompt Unicode no longer crashes encoding
- Token cache safety: tokenizer version is now part of the cache key to prevent stale matrix reuse after tokenizer layout changes
- Validation: shared probes for tokenizer roundtrip, greedy decode, sampled top-p decode, memorization-prefix continuation, and first-step logits
- Training metrics: held-out validation loss is logged alongside training loss
- Artifacts: `output/checkpoints`, `output/logs`, `output/cache/tokenizer`
- auto_train milestones: checkpoint + probes at 25%, 50%, 75%, and 100% of total steps
- Artifact naming: includes model token (`<preset>_<embedding>d_<layers>l`)
- Default training context: 128 tokens when VRAM allows
- auto_train UX: summary screen now appears first, with explicit confirmation before expensive tokenizer/VRAM preflight and training start

- auto_train UX: preset menu shows VRAM/speed/collapse-risk cost cards; Regime Controller probes language quality every 100 steps
- Regime telemetry: `output/regime_metrics_latest.jsonl`; offline trajectory scoring via `regime_policy_optimizer.py`
- Label smoothing: default 0.1 in `RunConfig`; fused cross-entropy kernel in `core/loss.py`
- Default `small` preset: `embedding_dim=64` (aligned with 4096-token vocab)

## Latest Progress (July 2026)

- **Phase 2C fused attention:** single-kernel `fused_attention_forward_kernel` (QKt + softmax + PV) wired into production `model/gpt.py` — one launch instead of two, probs row stays in shared memory between softmax and PV.
- **GPU-resident MHA integration:** `MultiHeadAttention` uses GPU layout adapters + fused forward; validated by `test_mha_integration_parity.py` and `smoke_train_mha_integration.py`.
- **Label smoothing:** `RunConfig.label_smoothing` threaded through `auto_train.py` and `train.py`; kernel-level tests in `test_label_smoothing_loss.py`.
- **Regime Controller:** BIS/TTR/RCI/Phi telemetry + bounded mid-run control (`label_smoothing`, `lr_regime_multiplier`) in `auto_train.py`; see [docs/REGIME_CONTROLLER.md](docs/REGIME_CONTROLLER.md).
- **Trajectory scoring:** `regime_policy_optimizer.py` scores past runs for faster semantic emergence, stable dwell, and lower fragmentation.

## Latest Progress (June 2026)

- Added GPU-resident FeedForward backward pass, eliminating the CPU/NumPy round-trip from FFN gradients; legacy CPU path kept as a togglable correctness oracle (`use_cpu_backward`).
- Added Phase 3 Multi-Head Attention CUDA kernels (`core/mha_kernels.py`) with a strict Score-Space / Projection-Space / Meta-Space memory model: causal score matmul, fused causal softmax forward/backward (VJP form), and projection-space `dQ`/`dK`/`dV` gradient kernels.
- Added a NumPy golden-model validation harness (`test_mha_golden_model.py`) gating MHA kernel correctness before any tiling/fusion optimization work, plus an FFN backward parity harness (`test_ffn_gpu_backward.py`).
- Centralized all GPU kernels into a single shared `SourceModule` compilation pass and added a cosine warmup learning-rate scheduler.
- Added live `tqdm` progress bars to tokenizer building and corpus file loading so large multi-million-document runs no longer appear to hang.

## Latest Progress (May 2026)

- Hardened leak-free validation behavior: tokenizer can be built from the training split while still safely encoding unseen validation symbols through byte fallback.
- Added decode-side byte reassembly so fallback tokens roundtrip back to their original Unicode text.
- Added tokenizer regression tests for unseen Unicode encode/decode fallback behavior.
- Purged tokenizer cache after vocabulary-layout changes (`output/cache/tokenizer`) and validated clean rebuild behavior.

## Recommended Workflow

1. Start a fresh run with `auto_train.py`.
2. Let the run save milestone checkpoints at 25%, 50%, 75%, and 100%.
3. Compare greedy vs sampled probe output across those milestone checkpoints.
4. Use the same prompt and memorization prefix across runs to compare quality cleanly.

## Useful Commands

```powershell
cd "C:\dev\llm gpu 5"
.\venv\Scripts\Activate.ps1
python .\auto_train.py
python .\regime_policy_optimizer.py output\regime_metrics_latest.jsonl
python .\test_regime_monitor.py
python .\generate.py --checkpoint output/checkpoints/<checkpoint>.npz --prompt "the" --max_new_tokens 40
python .\training_log_plotter.py --select
python .\training_log_plotter.py --select --no-show --export-csv output/logs/metrics_export.csv
```

### Publishing Changes

`gitpush.py` runs the repo's standard status -> add -> commit -> push sequence against `origin/main` in one step (PowerShell-safe; no `&&`/heredoc quoting issues):

```powershell
python .\gitpush.py "your commit message"
python .\gitpush.py    # no args: shows pending files and prompts for a message
```

If tokenizer behavior changes, clear cache before the next run:

```powershell
Remove-Item -Path "output\cache\tokenizer\*" -Recurse -Force
```

For the latest operational notes, start with [docs/index.md](docs/index.md), then open [docs/REGIME_CONTROLLER.md](docs/REGIME_CONTROLLER.md) for language-quality telemetry.