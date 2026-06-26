# Getting Started

Last updated: 2026-05-24

Current state: use this guide for the repaired flow. The corpus loader preserves newline structure, training samples across the dataset, and the generation scripts can now run the shared probe set on fresh checkpoints.

## Prerequisites

- Windows PowerShell
- Project virtual environment at .\venv
- CUDA-capable NVIDIA GPU (targeted to GT730)

## 5-minute run

```powershell
cd "C:\dev\llm gpu 5"
.\venv\Scripts\Activate.ps1
python .\auto_train.py
```

In auto_train.py:
- Choose dataset (default is FineWeb if available)
- Choose model preset
- Choose LR, steps, and sequence length
- Confirm summary to begin training

During training, auto_train.py will now save checkpoints and run probes at 25%, 50%, 75%, and 100% of your configured steps. It also logs a small held-out validation loss so training loss alone is not the only quality signal.

The default sequence length is now 128 when the VRAM estimate allows it.

## Quick generation test

```powershell
python .\generate.py --checkpoint output/checkpoints/<your_checkpoint>.npz --prompt "the" --max_new_tokens 40
```

If you want a persistent prompt loop without reloading the model each turn:

```powershell
python .\interactive_model_tester.py
```

The tester automatically prefers the latest best checkpoint recorded in `output/last_run_config.json`. You can still override it with `--checkpoint`.

## If you want the fixed-script trainer

```powershell
python .\train.py
```

This uses train.py defaults and writes latest checkpoints/logs to output.

## Where files go

- Checkpoints: output/checkpoints
- Logs: output/logs
- Token cache: output/cache/tokenizer
- Last auto_train config: output/last_run_config.json
