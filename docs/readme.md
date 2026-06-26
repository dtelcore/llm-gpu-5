# GT730 LLM Project Documentation

Last updated: 2026-05-24

Current state: the repaired data path is the baseline. Training samples across the corpus, generation uses the same shared tokenizer path, and fresh checkpoints are validated with the shared probe set.

This docs folder describes the current codebase state for the GT730-focused PyCUDA GPT training project.

## What this project is

- Custom GPT-style training and inference pipeline built on PyCUDA.
- Primary hardware target: NVIDIA GeForce GT 730 (Kepler, sm_35).
- Character-level tokenizer by default.
- Checkpoint format: compressed NPZ files in output/checkpoints.

## Current primary scripts

- auto_train.py: interactive training flow with dataset/model/hyperparameter prompts.
- train.py: non-interactive training engine with fixed in-file defaults.
- generate.py: checkpoint-based text generation.
- pipeline.py: setup walkthrough and high-level orchestration helper.
- main.py: end-to-end demo pipeline for token staging and forward pass.

## Common output locations

- output/checkpoints: model checkpoints
- output/logs: training and run logs
- output/cache/tokenizer: cached token matrices
- output/last_run_config.json: latest auto_train configuration snapshot

## Documentation map

- GETTING_STARTED.md: fastest first run
- QUICK_REFERENCE.md: command cheat sheet
- TRAINING_WORKFLOW.md: detailed training flow
- TRAINING_MODES.md: when to use each script
- ARCHITECTURE_GUIDE.md: module and data flow overview
- LOGGING.md and LOGS_GUIDE.md: logging behavior and log inspection
- FINEWEB_SETUP.md: FineWeb dataset notes
- changelog.md and stepchangelog.md: recent documentation and project updates

## Notes

Historically, many docs in this folder reflected older artifact paths and older model presets. This folder now tracks the current code state as of the date above.
