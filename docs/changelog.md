# Changelog

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
