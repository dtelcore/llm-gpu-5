# Changelog

## 2026-05-24

- Repaired the shared corpus path so FineWeb and custom corpora preserve normalized newline structure instead of stripping it away.
- Changed training to sample batch windows across the corpus instead of reusing one fixed slice.
- Added reusable generation probes for tokenizer roundtrip, greedy decode, memorization-prefix continuation, and first-step logits.
- Wired both `auto_train.py` and `train.py` to emit probe reports after checkpoint saves.
- Updated `auto_train.py` milestone behavior to percentage-based triggers (25/50/75/100) with checkpoint + probe at each milestone.
- Updated `auto_train.py` artifact naming to include model config token in log/checkpoint filenames.
- Raised the default interactive and fixed trainer context length to 128 when VRAM allows.
- Switched the shared tokenizer to a hybrid word/piece vocabulary with character fallback and a larger default cap.
- Added a small held-out validation slice and step-level validation loss reporting.
- Added sampled generation probes using top-p sampling plus a mild repetition penalty alongside greedy decode.
- Moved older logs and checkpoints into `output/logs/old` and `output/checkpoints/old` to keep new runs clean.
- Added fresh root-level documentation entry points for the current project state.
- Refactored `auto_train.py` UX flow to print the summary first, then require explicit user confirmation before tokenizer/VRAM preflight and training start.
- Added tokenizer UTF-8 byte fallback with a reserved 256-token window to prevent hard OOV crashes on unseen validation/prompt Unicode.
- Added byte-token decode reassembly so fallback IDs roundtrip back to original Unicode text.
- Added tokenizer regression tests for unseen-Unicode byte fallback encoding and decode roundtrip.
- Added tokenizer versioning to token-matrix cache keys so stale caches are invalidated automatically after tokenizer layout changes.
- Cleared tokenizer cache directory and validated rebuild behavior with train-only tokenizer + unseen validation Unicode smoke check.

## Notes

This changelog reflects the repaired flow as the baseline. Older checkpoint quality issues were traced to the data path, not the loader.