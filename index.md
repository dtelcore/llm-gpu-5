# Project Index

This is the canonical starting point for the current state of the project.

## Latest Snapshot (2026-07-01)

- **Phase 2C fused attention** integrated into production `model/gpt.py` (single-kernel QKt + softmax + PV).
- **Label smoothing** (default 0.1) in fused cross-entropy loss; threaded through `auto_train.py` and `train.py`.
- **Regime Controller** in `auto_train.py`: BIS/TTR/RCI/Phi probes every 100 steps; bounded mid-run control of `label_smoothing` and `lr_regime_multiplier`.
- **AutoTrain cost cards:** VRAM, tok/s, collapse-risk estimates per preset.
- **Trajectory scoring:** `regime_policy_optimizer.py` for offline policy evaluation.
- **`embedding_dim=64`** for `small` preset (4096-token vocab stability).
- Full docs: [docs/index.md](docs/index.md), [docs/REGIME_CONTROLLER.md](docs/REGIME_CONTROLLER.md).

## Previous Snapshot (2026-06-30)

- GPU-resident FeedForward backward pass (`core/ops.py`).
- Phase 3 MHA CUDA kernels (`core/mha_kernels.py`) and `MHAController` sandbox.
- Three-layer MHA correctness gate (`test_mha_golden_model.py`).
- FFN backward parity harness (`test_ffn_gpu_backward.py`).
- Cosine warmup scheduler; shared kernel compilation; tqdm progress bars.

## Previous Snapshot (2026-05-24)

- Repaired corpus path (newline preservation).
- Sampled batch windows; held-out validation loss.
- Milestone checkpoints + generation probes at 25/50/75/100%.
- Hybrid tokenizer with byte fallback.

## Quick links

| Task | Command / Doc |
|------|----------------|
| Train interactively | `python auto_train.py` |
| Score language trajectory | `python regime_policy_optimizer.py` |
| Architecture overview | [docs/ARCHITECTURE_GUIDE.md](docs/ARCHITECTURE_GUIDE.md) |
| Regime metrics | [docs/REGIME_CONTROLLER.md](docs/REGIME_CONTROLLER.md) |
| All docs | [docs/index.md](docs/index.md) |
