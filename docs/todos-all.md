# Consolidated TODOs

Last updated: 2026-05-24

Current state: keep the repaired data-path and probe automation as the baseline, and use the remaining worklist to improve training quality rather than re-debugging the loader.

## Runtime validation

- Re-run test suite and capture current pass/fail snapshot in docs.
- Add a small reproducibility checklist for training runs (seeding, config capture, checkpoint naming).

## Documentation quality

- Keep docs examples synced with actual argparse flags after script changes.
- Add explicit links between docs files for easier navigation.
- Add one-page glossary for terms used across training logs.

## Reliability and ops

- Add known limitations section for GT730 memory behavior.
- Add failure recovery section for interrupted training/checkpoint resume.

## Developer ergonomics

- Provide a single script or task to open latest log and latest checkpoint quickly.
- Add a short command reference for common inspection operations.
