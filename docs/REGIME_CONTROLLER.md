# Regime Controller (BIS / TTR / RCI / Phi)

Last updated: 2026-07-01

The Regime Controller is an **active control layer** in `auto_train.py` that measures linguistic coherence during training and nudges safe hyperparameters mid-run. It decouples **loss (prediction accuracy)** from **representational integrity (word-boundary stability)**.

Implementation: [`regime_monitor.py`](../regime_monitor.py) (metrics + controller logic), wired in [`auto_train.py`](../auto_train.py) (`_run_regime_probe_and_apply`). Tests: [`test_regime_monitor.py`](../test_regime_monitor.py).

Trajectory scoring (offline policy evaluation): [`regime_policy_optimizer.py`](../regime_policy_optimizer.py).

---

## Why this exists

Training loss can improve while generated text degrades (token-boundary fusion: `"andthe"`, `"Heher"`, repetition loops). The controller probes the **live in-memory model** every 100 steps and classifies the run into a **language regime** before output quality becomes visually obvious.

---

## Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **BIS** (Boundary Integrity Score) | 0–1 | Word-boundary stability. Detects glued function-word artifacts (`andthe`, `tothe`, pronoun fusion). 1.0 = clean. |
| **TTR** (Type-Token Ratio) | 0–1 | Vocabulary richness (`unique_tokens / total_tokens`). |
| **RCI** (Repetition Collapse Index) | 0–1 | Attractor-basin tendency (`0.5 * dominant_ratio + 0.5 * immediate_repeat_ratio`). |
| **Phi** (Regime Transition Function) | 0+ | `(BIS * TTR) / max(RCI, 0.05)` |

### Regime classification (from Phi)

| Phi | Regime | Typical behavior |
|-----|--------|------------------|
| `< 0.2` | `attractor_collapse` | Loops, low BIS, loss may still look OK |
| `0.2 – 0.6` | `syntactic_drift` | Structure without content; boundary artifacts |
| `0.6 – 1.2` | `semantic_emergence` | Vocabulary used properly; BIS recovering |
| `> 1.2` | `stable_generator` | Robust high-entropy language manifold |

---

## Probe schedule

Constants in `auto_train.py`:

- `REGIME_PROBE_INTERVAL_STEPS = 100` — probe every 100 training steps
- `REGIME_MIN_STEP = 50` — no probes before step 50
- `REGIME_PROBE_MAX_TOKENS = 40` — greedy decode length
- `REGIME_PROBE_PROMPT = "Once upon a "` — fixed prompt for comparability

The probe uses `lightweight_greedy_probe()`: greedy decode against the **current training weights** (no checkpoint reload).

---

## Controller actions (bounded, cooldown-gated)

Every action type has a **300-step cooldown** to prevent oscillation.

| Action | Trigger | Live effect |
|--------|---------|-------------|
| `EMBEDDING_EXPANSION_RECOMMENDED` | BIS `< 0.5` | **Next-run only:** sets `recommended_next_embedding_dim = 2 × current` in config. Never resizes the model mid-run. |
| `INCREASE_LABEL_SMOOTHING` | RCI `> 0.6` and TTR `< 0.4` | `label_smoothing += 0.02`, clamped to `[0.0, 0.3]` |
| `DECAY_LEARNING_RATE` | BIS `> 0.75` and Phi trend `rising` | `lr_regime_multiplier *= 0.9`, floored at `0.5` (only ever decreases) |
| `SAVE_AWAKENING_CHECKPOINT` | Regime crosses from collapse/drift → emergence/stable | Saves `{checkpoint}.awakening.step{N}.npz` |

Effective learning rate:

```
current_lr = scheduler.get_lr(step) * lr_regime_multiplier
```

---

## Artifacts and logs

### Training log lines

```
[REGIME] step=200 bis=0.412 ttr=0.350 rci=0.680 phi=0.212 regime=syntactic_drift ema_phi=0.245 trend=falling actions=EMBEDDING_EXPANSION_RECOMMENDED
```

### JSONL telemetry

`output/regime_metrics_latest.jsonl` — one JSON object per probe:

```json
{
  "step": 200,
  "bis": 0.412,
  "ttr": 0.350,
  "rci": 0.680,
  "phi": 0.212,
  "regime": "syntactic_drift",
  "ema_phi": 0.245,
  "trend": "falling",
  "actions": ["EMBEDDING_EXPANSION_RECOMMENDED"],
  "probe_text": "Once upon a ..."
}
```

### Persisted config (`output/last_run_config.json`)

- `label_smoothing` — final value after controller adjustments
- `recommended_next_embedding_dim` — surfaced as a hint at the next AutoTrain model-config prompt

---

## AutoTrain preset cost cards

When selecting a model preset, the menu shows deterministic estimates at `vocab≈4096`, `ctx=128`:

- Estimated VRAM and % of GT730 budget
- Estimated tok/s (relative heuristic)
- Embedding/vocab ratio (`embed_dim / log2(vocab_size)`)
- Collapse risk label (HIGH / MODERATE / LOW)

See `build_preset_cost_card()` in `auto_train.py`.

---

## Trajectory shaping (policy optimization)

The rule-based controller is **observability + safe reactive control**. **Optimal trajectory shaping** treats training as a controlled dynamical system:

**State** `s_t`: (BIS, TTR, RCI, Phi, EMA(Phi), trend, regime, step, loss, …)

**Actions** `a_t`: bounded changes to `label_smoothing`, `lr_regime_multiplier`, checkpoint saves, next-run embed recommendation.

**Objectives:**

1. **Faster semantic emergence** — minimize steps until first `Phi ≥ 0.6`
2. **Longer stable dwell** — maximize time in `semantic_emergence` + `stable_generator`
3. **Lower fragmentation** — minimize fraction of probes with `BIS < 0.5`

**Trajectory score J** (implemented in `regime_policy_optimizer.py`):

```
J = α / T_emerge + β * T_dwell - γ * P_frag
```

Score past runs from JSONL to compare threshold schedules or tune controller constants offline — no GPU required.

```powershell
python .\regime_policy_optimizer.py output\regime_metrics_latest.jsonl
```

---

## Scope limits

- **Only `auto_train.py`** — `train.py` does not run the regime loop (fixed-config path).
- **No live architecture mutation** — `embedding_dim`, `num_heads`, `num_layers` are frozen after `GPTModel` construction.
- **LR multiplier never increases** once lowered by the controller.

---

## Related tests

```powershell
python .\test_regime_monitor.py
python .\test_regime_policy_optimizer.py
```
