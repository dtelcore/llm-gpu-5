# Logging Configuration

Last updated: 2026-07-01

Current state: training logs include per-step metrics, validation loss, milestone probe reports, and Regime Controller `[REGIME]` lines.

## Source

`logging_config.py` configures a global logger named KepleGPT with:
- Console handler (safe Unicode degradation)
- File handler (UTF-8)

## Log file location

- `output/logs`

Default filename if no custom run name is provided:
- `training_YYYYMMDD_HHMMSS.log`

auto_train.py creates custom run log names via `setup_logging(log_filename=<name>)`.

## Runtime usage pattern

Most modules import:

```python
from logging_config import logger
```

## Training metric line format

`training_metrics.py` emits compact progress lines:

```
[train][cuda] step=... loss=... avg_loss=... val_loss=... val_ppl=... lr=... ppl=... grad_norm=... elapsed=... eta=... step_ms=... tok/s=... pool_used_mb=... device_used_mb=...
```

## Regime Controller line format

Every 100 steps (after step 50) in `auto_train.py`:

```
[REGIME] step=200 bis=0.412 ttr=0.350 rci=0.680 phi=0.212 regime=syntactic_drift ema_phi=0.245 trend=falling actions=EMBEDDING_EXPANSION_RECOMMENDED
```

Controller action lines may also appear as `[REGIME]` warnings (label smoothing bump, embedding recommendation).

## Regime JSONL telemetry

Parallel structured log: `output/regime_metrics_latest.jsonl`

One JSON object per probe (step, bis, ttr, rci, phi, regime, ema_phi, trend, actions, probe_text).

See [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md).

## Training CSV/JSONL metrics

`output/training_metrics_latest.csv` and `.jsonl` — per-step scalars from `TrainingMetrics`.
