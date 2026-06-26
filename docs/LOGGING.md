# Logging Configuration

Last updated: 2026-05-24

Current state: training logs now include the repaired pipeline's probe reports after checkpoint saves, alongside the per-step loss and VRAM metrics.

## Source

logging_config.py configures a global logger named KepleGPT with:
- Console handler (safe Unicode degradation)
- File handler (UTF-8)

## Log file location

- output/logs

Default filename if no custom run name is provided:
- training_YYYYMMDD_HHMMSS.log

## Runtime usage pattern

Most modules import:

```python
from logging_config import logger
```

auto_train.py can create custom run log names by calling setup_logging(log_filename=<name>).

## Training metric line format

training_metrics.py emits compact progress lines with backend tag:
- [train][cuda] step=... loss=... ppl=... tok/s=... pool_used_mb=... device_used_mb=...

This is the authoritative per-step metric format for current training runs.
