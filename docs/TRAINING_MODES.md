# Training Modes

Last updated: 2026-07-01

## 1) auto_train.py (recommended)

Interactive control over dataset, model preset (with cost cards), hyperparameters, init checkpoint, and post-training generation.

**Includes:**
- Regime Controller (BIS/Phi probes every 100 steps)
- Label smoothing (adjustable by controller mid-run)
- Preset collapse-risk hints
- Milestone checkpoints + full generation probes

```powershell
python .\auto_train.py
```

## 2) train.py (fixed in-script defaults)

Non-interactive repeatability. Loads `output/last_run_config.json`. Label smoothing yes; **no Regime Controller**.

```powershell
python .\train.py
```

## 3) pipeline.py (setup-oriented walkthrough)

Guided flow for configuration understanding.

```powershell
python .\pipeline.py
```

## 4) generate.py (inference)

After a checkpoint exists:

```powershell
python .\generate.py --checkpoint output/checkpoints/<checkpoint>.npz --prompt "the" --max_new_tokens 40
```

## 5) regime_policy_optimizer.py (offline analysis)

Score a completed run's language-quality trajectory from JSONL:

```powershell
python .\regime_policy_optimizer.py output\regime_metrics_latest.jsonl
```

Not a training entry point — post-run analysis only.
