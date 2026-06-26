# Training Modes

Last updated: 2026-05-24

Current state: the interactive and fixed trainers both use the repaired corpus path and now emit the shared probe diagnostics after checkpoint saves.
Both trainers also report held-out validation loss, and the default context length is 128 when VRAM allows.

## 1) auto_train.py (recommended)

Use when you want interactive control over:
- Dataset
- Model preset and dimensions
- Learning rate, steps, sequence length
- Optional init checkpoint compatibility/adoption
- Optional generation after training

Command:

```powershell
python .\auto_train.py
```

## 2) train.py (fixed in-script defaults)

Use when you want a consistent non-interactive run path for repeatability.

Command:

```powershell
python .\train.py
```

## 3) pipeline.py (setup-oriented walkthrough)

Use when you want a guided flow for configuration understanding and project orientation.

Command:

```powershell
python .\pipeline.py
```

## 4) generate.py (inference)

Use after a checkpoint exists.

Command:

```powershell
python .\generate.py --checkpoint output/checkpoints/<checkpoint>.npz --prompt "the" --max_new_tokens 40
```

The generation path now supports greedy decoding, top-p sampling, and a mild repetition penalty so sampled probes can show whether distribution quality improved before greedy decode fully opens up.
