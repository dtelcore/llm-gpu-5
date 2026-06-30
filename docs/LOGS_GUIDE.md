# Logs Guide

Last updated: 2026-07-01

When comparing runs, inspect scalar metrics, `[REGIME]` language-quality lines, milestone probe output, and offline trajectory scores.

## List recent logs

```powershell
Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object Name, LastWriteTime
```

## Show latest log tail

```powershell
$latest = Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) { Get-Content $latest.FullName -Tail 120 }
```

## Follow latest log live

```powershell
$latest = Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) { Get-Content $latest.FullName -Wait }
```

## Filter training metrics

```powershell
Get-ChildItem output\logs\*.log | Select-String "\[train\]\[cuda\]|loss=|ppl=|tok/s=|device_used_mb="
```

## Filter Regime Controller lines

```powershell
Get-ChildItem output\logs\*.log | Select-String "\[REGIME\]"
```

## Inspect regime JSONL telemetry

```powershell
Get-Content output\regime_metrics_latest.jsonl
python .\regime_policy_optimizer.py output\regime_metrics_latest.jsonl
```

## Interactive plotting (matplotlib)

```powershell
python -m pip install matplotlib
python .\training_log_plotter.py --select
```

Top chart: train + validation loss. Bottom: lr, tok/s, grad_norm, step_ms, device_used_mb, ppl.

## Export for external dashboards

```powershell
python .\training_log_plotter.py --select --no-show --export-csv output\logs\metrics_export.csv --export-json output\logs\metrics_export.json
```

## What to compare across runs

1. **Loss curve** — prediction accuracy (can improve while language quality degrades)
2. **`[REGIME]` phi/regime trend** — representational integrity
3. **Milestone probe text** — greedy vs sampled at 25/50/75/100%
4. **Trajectory score J** — single scalar from `regime_policy_optimizer.py`

See [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md) for metric definitions.
