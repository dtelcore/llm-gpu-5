# Logs Guide

Last updated: 2026-05-24

Current state: when comparing runs, inspect the saved probe output for tokenizer roundtrip, greedy decode, memorization-prefix, and first-step logits alongside the scalar metrics.

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

## Filter by key fields

```powershell
Get-ChildItem output\logs\*.log | Select-String "\[train\]\[cuda\]|loss=|ppl=|tok/s=|device_used_mb="
```

## Interactive plotting (matplotlib)

Install matplotlib once:

```powershell
python -m pip install matplotlib
```

Launch interactive selector (pick one or multiple logs):

```powershell
python .\training_log_plotter.py --select
```

The plotter opens:
- Top chart: train loss and validation loss
- Bottom chart: switch metric with the radio controls (lr, tok/s, grad_norm, step_ms, device_used_mb, ppl)

## Export for Grafana or external dashboards

Export parsed rows to CSV and JSON:

```powershell
python .\training_log_plotter.py --select --no-show --export-csv output\logs\metrics_export.csv --export-json output\logs\metrics_export.json
```

Each row includes run name, timestamp, step, and all parsed scalar fields from the compact [train][backend] lines.
