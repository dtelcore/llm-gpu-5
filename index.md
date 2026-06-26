# Project Index

This is the canonical starting point for the current state of the project.

## Latest Snapshot (2026-05-24)

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