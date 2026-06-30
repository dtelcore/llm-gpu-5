# GT730 LLM Project Documentation

Last updated: 2026-07-01

Current state: GPU-resident MHA (Phase 2C fused attention) in production `model/gpt.py`. Training uses label smoothing, AutoTrain preset cost cards, and a BIS/Phi Regime Controller in `auto_train.py`. Offline trajectory scoring via `regime_policy_optimizer.py`.

This docs folder describes the GT730-focused PyCUDA GPT training project.

## What this project is

- Custom GPT-style training and inference pipeline built on PyCUDA
- Primary hardware target: NVIDIA GeForce GT 730 (Kepler, sm_35)
- Hybrid word/piece tokenizer with character/byte fallback
- Checkpoint format: compressed NPZ files in `output/checkpoints`

## Start here

1. [index.md](index.md) — documentation map
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md) — commands
3. [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) — full pipeline
4. [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md) — language-quality telemetry and control

## Current primary scripts

- `auto_train.py` — interactive training + Regime Controller
- `train.py` — fixed-config training (no regime loop)
- `generate.py` — checkpoint-based text generation
- `regime_policy_optimizer.py` — offline trajectory scoring from JSONL

## Key artifacts

- Checkpoints: `output/checkpoints/`
- Logs: `output/logs/`
- Regime telemetry: `output/regime_metrics_latest.jsonl`
- Run config: `output/last_run_config.json`
- Token cache: `output/cache/tokenizer/`
