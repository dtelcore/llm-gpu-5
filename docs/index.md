# Documentation Index

Last updated: 2026-07-01

Current state: GPU-resident MHA is integrated into production `model/gpt.py` (Phase 2C single-kernel fused attention forward). Training uses label smoothing, AutoTrain preset cost cards, and a BIS/Phi Regime Controller that probes language quality every 100 steps. Offline trajectory scoring is available via `regime_policy_optimizer.py`.

## Start here

1. [GETTING_STARTED.md](GETTING_STARTED.md)
2. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
3. [TRAINING_WORKFLOW.md](TRAINING_WORKFLOW.md)
4. [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md) — BIS/TTR/RCI/Phi metrics and active training control

## By topic

- Project overview: [readme.md](../readme.md)
- Architecture: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md)
- Training entry points: [TRAINING_MODES.md](TRAINING_MODES.md)
- Regime Controller and trajectory scoring: [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md)
- Dataset setup: [FINEWEB_SETUP.md](FINEWEB_SETUP.md)
- Logging system: [LOGGING.md](LOGGING.md)
- Log inspection commands: [LOGS_GUIDE.md](LOGS_GUIDE.md)
- CUDA and MSVC background: [MSVC_FIX_REPORT.md](MSVC_FIX_REPORT.md)
- Context scaling notes: [SCALING_T64_CHANGES.md](SCALING_T64_CHANGES.md)
- Progress and history: [changelog.md](changelog.md), [stepchangelog.md](stepchangelog.md)
- Current backlog: [todo.md](todo.md), [todos-all.md](todos-all.md)
- Completion snapshot: [PROJECT_COMPLETION.md](PROJECT_COMPLETION.md)
- GPU kernel internals: [ARCHITECTURE_GUIDE.md](ARCHITECTURE_GUIDE.md) (FFN backward, MHA kernels, golden-model tests)
- Publishing changes: `gitpush.py` (repo root) runs status -> add -> commit -> push to `origin/main`

## Test harnesses

| Script | Purpose |
|--------|---------|
| `test_mha_golden_model.py` | 6-layer MHA kernel + controller parity |
| `test_mha_integration_parity.py` | Production `MultiHeadAttention` vs oracle |
| `test_label_smoothing_loss.py` | Fused loss kernel with label smoothing |
| `test_regime_monitor.py` | BIS/Phi metrics and RegimeController (no GPU) |
| `test_regime_policy_optimizer.py` | Offline trajectory score J (no GPU) |
| `smoke_train_mha_integration.py` | Full training loop smoke test |
