# Documentation and Project TODO

Last updated: 2026-05-24

Current state: the next near-term task is to extend the fresh repaired run and compare the shared probe output at 500, 1000, and 2000 steps.

## High priority

- Validate docs examples against a fresh end-to-end run (auto_train -> generate).
- Add a single canonical benchmark doc for GT730 runs (throughput, VRAM, convergence patterns).
- Add an explicit troubleshooting matrix for common CUDA and checkpoint compatibility failures.

## Medium priority

- Consolidate overlapping setup guidance between pipeline.py and auto_train.py docs.
- Add versioned checkpoint naming conventions and retention policy notes.
- Expand docs for tokenizer cache invalidation behavior.

## Low priority

- Add architecture diagram assets (PNG/SVG) alongside markdown summaries.
- Add command snippets for scripted experiment sweeps.
