# Quick Reference

Last updated: 2026-07-01

Current state: use `auto_train.py` for interactive runs with Regime Controller probes. Score language-quality trajectories offline with `regime_policy_optimizer.py`.

## Environment

```powershell
cd "C:\dev\llm gpu 5"
.\venv\Scripts\Activate.ps1
```

## Train (interactive, recommended)

```powershell
python .\auto_train.py
```

Features:
- Preset cost cards (VRAM, tok/s, collapse risk)
- Regime probes every 100 steps (BIS/Phi)
- Label smoothing (default 0.1)
- Milestone checkpoints + full probes at 25/50/75/100%

## Train (fixed defaults)

```powershell
python .\train.py
```

Uses `output/last_run_config.json`. Label smoothing yes; Regime Controller no.

## Score regime trajectory (offline)

```powershell
python .\regime_policy_optimizer.py output\regime_metrics_latest.jsonl
```

## Run tests (no training)

```powershell
python .\test_regime_monitor.py
python .\test_regime_policy_optimizer.py
python .\test_mha_golden_model.py
python .\test_label_smoothing_loss.py
```

## Generate text

```powershell
python .\generate.py --checkpoint output/checkpoints/gpt_model_latest.npz --prompt "cuda " --max_new_tokens 40 --temperature 0.6
```

Omit `--checkpoint` to auto-pick best from `output/last_run_config.json` or newest `*.best.npz`.

## Interactive model tester

```powershell
python .\interactive_model_tester.py
```

## View latest logs

```powershell
Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 3
$latest = Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) { Get-Content $latest.FullName -Tail 80 }
```

Filter regime lines:

```powershell
Get-ChildItem output\logs\*.log | Select-String "\[REGIME\]"
```

## List checkpoints

```powershell
Get-ChildItem output\checkpoints\*.npz | Sort-Object LastWriteTime -Descending
```

## Important defaults in code

- Shared FineWeb path: `data/fineweb_100mb.txt`
- Shared tokenizer cache: `output/cache/tokenizer`
- Run config: `output/last_run_config.json`
- Regime telemetry: `output/regime_metrics_latest.jsonl`
- Default `label_smoothing`: 0.1 (`RunConfig`)
- Default `embedding_dim` (small preset): 64
- Regime probe interval: every 100 steps after step 50 (`auto_train.py`)

## Documentation

- [docs/REGIME_CONTROLLER.md](REGIME_CONTROLLER.md) — BIS/TTR/RCI/Phi and controller actions
- [docs/TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md) — full training pipeline
- [docs/ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) — module map and MHA path
