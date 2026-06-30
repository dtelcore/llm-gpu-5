# Training Workflow

Last updated: 2026-07-01

Current state: `auto_train.py` runs the full interactive pipeline including Regime Controller probes (BIS/Phi every 100 steps), label smoothing, preset cost cards, and milestone generation probes. `train.py` is the fixed-config non-interactive path (label smoothing yes, regime controller no).

## Recommended path: auto_train.py

1. Dataset selection
- Minimal, FineWeb, or custom text file from `data/*.txt`
- Large datasets can be limited interactively (default cap prompt around 5000 docs)

2. Model architecture selection
- Preset menu from micro to giant (tiered by complexity)
- Each preset shows a **cost card**: estimated VRAM, tok/s, embedding/vocab ratio, collapse risk
- If a prior run detected boundary collapse, a **regime hint** may recommend a larger `embedding_dim`
- Attention implementation defaults to strided
- Sequence length and architecture can be adopted from init checkpoint metadata
- Default `small` preset uses `embedding_dim=64` for 4096-token vocab stability

3. Hyperparameter selection
- Learning rate (LR suggestion scales with approximate parameter count)
- Step count
- Sequence length

4. Logging and checkpoint naming
- Log name and checkpoint name are auto-generated from steps/LR/context/model-config/timestamp
- Config persisted to `output/last_run_config.json` (includes `label_smoothing`, `recommended_next_embedding_dim`)

5. Optional generation test
- Prompt and max tokens can be entered for post-training generation

## Core training loop behavior (auto_train.py)

For each step:
- Forward pass (GPU-resident MHA with Phase 2C fused attention forward)
- Fused softmax cross-entropy loss with `label_smoothing`
- Backward pass
- AdamW update with `current_lr = scheduler.get_lr(step) * lr_regime_multiplier`
- Cache cleanup and pooled memory stats logging

Every 100 steps (after step 50):
- **Regime probe:** greedy decode from live weights, compute BIS/TTR/RCI/Phi
- **Regime Controller:** may adjust `label_smoothing`, `lr_regime_multiplier`, save awakening checkpoint, or recommend next-run `embedding_dim`
- Telemetry appended to `output/regime_metrics_latest.jsonl`

Milestone checkpoint and probe behavior:
- At 25%, 50%, 75%, and 100% of configured total steps
- Save checkpoint at each milestone
- Run full generation probes (tokenizer roundtrip, greedy, sampled, memorization, logits)

Validation behavior:
- Held-out validation slice encoded separately
- Step logs include training loss and validation loss/perplexity

Metrics emitted through `training_metrics.py`:
- loss, avg_loss, val_loss, ppl, lr, grad_norm
- elapsed, eta, step_ms, tok/s
- pool_used_mb, pool_total_mb, device_used_mb

Regime log lines (`[REGIME]`):
- bis, ttr, rci, phi, regime, ema_phi, trend, actions

## train.py (fixed defaults)

Same core loop as above except:
- No Regime Controller probes
- Loads config from `output/last_run_config.json`
- Uses `label_smoothing` from `RunConfig`

## Artifacts

- Main checkpoint: `output/checkpoints/<name>.npz`
- Goal checkpoint (if thresholds met): `output/checkpoints/<name>.best.npz`
- Awakening checkpoint (regime pivot): `output/checkpoints/<name>.awakening.step<N>.npz`
- Logs: `output/logs/<name>.log`
- Regime telemetry: `output/regime_metrics_latest.jsonl`
- Token cache: `output/cache/tokenizer/*.npz`
- Training metrics: `output/training_metrics_latest.csv`, `.jsonl`

## Post-run analysis

Score the language-quality trajectory of a completed run:

```powershell
python .\regime_policy_optimizer.py output\regime_metrics_latest.jsonl
```

See [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md) for metric definitions and controller actions.
