# Quick Reference

Last updated: 2026-05-24

Current state: the fastest validation path is a fresh training run followed by the shared probe report. Use the same prompt and memorization prefix across checkpoints to compare quality consistently.

## Environment

```powershell
cd "C:\dev\llm gpu 5"
.\venv\Scripts\Activate.ps1
```

## Train (interactive)

```powershell
python .\auto_train.py
```

`auto_train.py` now auto-saves milestone checkpoints and runs probe diagnostics at 25%, 50%, 75%, and 100% of configured steps.
Those probes now include both greedy decode and a fixed sampled decode using top-p sampling plus a mild repetition penalty.

## Train (fixed defaults)

```powershell
python .\train.py
```

## Generate text

```powershell
python .\generate.py --checkpoint output/checkpoints/gpt_model_latest.npz --prompt "cuda " --max_new_tokens 40 --temperature 0.6
```

If you want the script to auto-pick the current best checkpoint from `output/last_run_config.json` or the newest `*.best.npz`, omit `--checkpoint`.

## Interactive model tester

```powershell
python .\interactive_model_tester.py
```

Or pin a specific checkpoint explicitly:

```powershell
python .\interactive_model_tester.py --checkpoint output/checkpoints/gpt_500steps_1p5e-06lr_ctx128_20260524_191056.best.npz --max_new_tokens 60 --temperature 0.8
```

## View latest logs

```powershell
Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 3
$latest = Get-ChildItem output\logs\*.log | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($latest) { Get-Content $latest.FullName -Tail 80 }
```

## List checkpoints

```powershell
Get-ChildItem output\checkpoints\*.npz | Sort-Object LastWriteTime -Descending
```

## Important defaults in code

- Shared FineWeb path: data/fineweb_100mb.txt
- Shared training corpus fallback: corpus_utils.TRAINING_CORPUS
- Shared tokenizer cache dir: output/cache/tokenizer
- generate.py default checkpoint: output/checkpoints/gpt_model_latest.npz
- interactive_model_tester.py default checkpoint: best checkpoint from output/last_run_config.json, else newest *.best.npz
- auto_train.py artifact names include model token (`<name>_<embedding>d_<layers>l`) in log/checkpoint filenames
- Training now reports a held-out validation loss in the step logs
