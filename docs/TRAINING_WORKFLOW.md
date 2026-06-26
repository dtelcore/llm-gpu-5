# Training Workflow

Last updated: 2026-05-24

Current state: training now samples batch windows across the corpus, and fresh checkpoints should be validated with the shared probe set before judging generation quality.
The default interactive context length is 128 when VRAM allows, and the shared tokenizer now builds a larger hybrid word/piece vocabulary with character fallback.

## Recommended path: auto_train.py

1. Dataset selection
- Minimal, FineWeb, or custom text file from data/*.txt
- Large datasets can be limited interactively (default cap prompt around 5000 docs)

2. Model architecture selection
- Preset menu from micro to giant
- Attention implementation defaults to strided
- Sequence length and architecture can be adopted from init checkpoint metadata

3. Hyperparameter selection
- Learning rate
- Step count
- Sequence length

4. Logging and checkpoint naming
- Log name and checkpoint name are auto-generated from steps/LR/context/model-config/timestamp
- Config persisted to output/last_run_config.json

5. Optional generation test
- Prompt and max tokens can be entered for post-training generation

## Core training loop behavior

For each step:
- Forward pass
- Fused softmax cross-entropy loss
- Backward pass
- AdamW update
- Cache cleanup and pooled memory stats logging

Milestone checkpoint and probe behavior in auto_train.py:
- At 25%, 50%, 75%, and 100% of configured total steps
- Save checkpoint at each milestone
- Run generation probes (tokenizer roundtrip, greedy decode, sampled top-p decode, memorization-prefix, first-step logits) at each milestone

Validation behavior in both trainers:
- A small held-out validation slice is encoded separately from the training slice
- Step logs include both training loss and held-out validation loss/perplexity

Metrics are emitted through training_metrics.py with per-step fields including:
- loss, avg_loss, ppl, lr, grad_norm
- elapsed, eta, step_ms, tok/s
- pool_used_mb, pool_total_mb, device_used_mb

## Artifacts

- Main checkpoint: output/checkpoints/<name>.npz
- Goal checkpoint (if thresholds met): output/checkpoints/<name>.best.npz
- Logs: output/logs/<name>.log
- Token cache: output/cache/tokenizer/*.npz
