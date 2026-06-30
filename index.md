# Project Index

This is the canonical starting point for the current state of the project.

## Latest Snapshot (2026-06-30)

- Added GPU-resident FeedForward backward pass (`core/ops.py`), removing the CPU/NumPy round-trip from FFN gradients.
- Added Phase 3 Multi-Head Attention CUDA kernels (`core/mha_kernels.py`) and an orchestration layer (`core/mha_ops.py: MHAController`) with fused QKV projection.
- Added a three-layer MHA correctness gate (`test_mha_golden_model.py`): kernel-level parity, fused-QKV representation equivalence, and controller execution-path parity, each against an independent NumPy reference.
- Added an FFN backward parity/timing harness (`test_ffn_gpu_backward.py`).
- Added `tqdm` progress bars to tokenizer building and corpus file loading so large multi-million-document runs no longer appear frozen.
- Added `gitpush.py`: a one-shot status -> add -> commit -> push helper for publishing changes to `origin/main`.

## Previous Snapshot (2026-05-24)

- Data path repaired and newline-safe corpus normalization is active.
- Training uses sampled windows plus held-out validation metrics.
- Probe/checkpoint milestones are percentage-based at 25/50/75/100.
- Hybrid tokenizer now includes UTF-8 byte fallback for unseen Unicode.
- Token cache keys include tokenizer versioning to avoid stale matrix reuse.
- `auto_train.py` now shows summary first and asks for explicit start confirmation before heavy preflight.

## Start Here

1. [readme.md](readme.md)
2. [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)

## Key Topics

- [docs/TRAINING_WORKFLOW.md](docs/TRAINING_WORKFLOW.md)
- [docs/TRAINING_MODES.md](docs/TRAINING_MODES.md)
- [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md)
- [docs/FINEWEB_SETUP.md](docs/FINEWEB_SETUP.md)
- [docs/LOGGING.md](docs/LOGGING.md)
- [docs/LOGS_GUIDE.md](docs/LOGS_GUIDE.md)
- [docs/PROJECT_COMPLETION.md](docs/PROJECT_COMPLETION.md)

## Current Baseline

- Repaired data path with newline-preserving corpus loading
- Sampled training windows instead of fixed-slice reuse
- Shared probe checks in `generate.py`
- Byte-fallback tokenizer path for OOV Unicode safety
- Versioned tokenizer cache keying in `corpus_utils.py`
- Fresh-run validation recommended before judging checkpoint quality