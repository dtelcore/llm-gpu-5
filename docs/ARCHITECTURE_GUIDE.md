# Architecture Guide

Last updated: 2026-07-01

Current state: GPU-resident FeedForward backward and Phase 2C fused Multi-Head Attention are integrated into production `model/gpt.py`. Training uses label smoothing, sampled corpus windows, held-out validation, Regime Controller telemetry in `auto_train.py`, and shared generation probes.

## High-level modules

- `core/kernels.py`: CUDA kernels for matmul, layernorm, GELU, AdamW
- `core/ops.py`: PyCUDA kernel wrappers (includes GPU-resident FFN backward)
- `core/mha_kernels.py`: MHA CUDA kernels — Phase 2A/2B/2C fusion, layout adapters (`split_heads_kernel`, `merge_heads_kernel`), golden-model oracles kept for parity
- `core/mha_ops.py`: `MHAController` sandbox + `get_shared_mha_module()` singleton used by `model/gpt.py`
- `core/loss.py`: fused softmax cross-entropy with optional `label_smoothing`
- `model/gpt.py`: GPTConfig, layers, forward/backward; `MultiHeadAttention` uses GPU-resident fused attention forward
- `regime_monitor.py`: BIS/TTR/RCI/Phi metrics, `RegimeTracker`, `RegimeController` (pure Python, CUDA-free except lazy import in probe)
- `regime_policy_optimizer.py`: offline trajectory score J from regime JSONL
- `tokenizer/`: hybrid word/piece tokenizer with character/byte fallback
- `corpus_utils.py`: dataset loading, tokenizer build, token cache
- `training_metrics.py`: per-step metrics + VRAM estimation helpers
- `run_config.py`: `RunConfig` dataclass (`label_smoothing`, `recommended_next_embedding_dim`, presets)
- `auto_train.py`: interactive training + Regime Controller loop
- `train.py`: fixed-config training (no regime loop)

## GPU kernel correctness gates

Standalone harnesses validate kernel correctness before production wiring:

| Test | Validates |
|------|-----------|
| `test_ffn_gpu_backward.py` | GPU FFN backward vs CPU oracle |
| `test_mha_golden_model.py` | 6 layers: kernels, fused QKV, controller, Phase 2A/2B/2C fusion |
| `test_mha_integration_parity.py` | Production `MultiHeadAttention` vs pre-integration oracle |
| `test_label_smoothing_loss.py` | Fused loss kernel with smoothing |
| `smoke_train_mha_integration.py` | Full GPT training loop smoke test |

## MHA production path (model/gpt.py)

`MultiHeadAttention.forward()`:

1. `split_heads_kernel` — GPU layout adapter: `[B*T, 3C]` → `Q/K/V [B*NH, T, HD]`
2. `fused_attention_forward_kernel` — Phase 2C: QKt + causal softmax + PV in one launch; probs written to global memory for backward cache only
3. `merge_heads_kernel` — GPU layout adapter: context heads → `[B*T, C]`

Backward pass still uses CPU-side cached attention weights (unchanged contract).

`MHAController` in `core/mha_ops.py` remains a standalone R&D/sandbox path using the same kernel module; production uses direct kernel handles in `MultiHeadAttention`.

## Regime Controller (auto_train.py only)

Every 100 steps after step 50:

1. `lightweight_greedy_probe()` — decode from live training weights
2. Compute BIS, TTR, RCI, Phi; classify regime
3. `RegimeController.decide()` — bounded actions with 300-step cooldowns
4. Log `[REGIME]` line; append to `output/regime_metrics_latest.jsonl`

Live knobs: `label_smoothing`, `lr_regime_multiplier`. Architecture changes are next-run recommendations only.

See [REGIME_CONTROLLER.md](REGIME_CONTROLLER.md).

## Runtime data flow

1. Text corpus loaded from dataset source
2. Tokenizer built from training split (hybrid vocab + byte fallback)
3. Token matrix created or loaded from `output/cache/tokenizer`
4. Input and target slices staged to GPU
5. Model forward (GPU-resident MHA + FFN)
6. Loss kernel (with label smoothing) computes loss and dLogits
7. Backward computes gradients
8. AdamW updates weights (LR × regime multiplier in auto_train)
9. Regime probe (auto_train only, every 100 steps)
10. Held-out validation evaluated on log steps
11. Checkpoint and logs written to `output/`

## Checkpoint metadata behavior

`generate.py` infers model config from checkpoint arrays and optional metadata. It can use `output/last_run_config.json` to resolve missing fields such as `num_heads` / `attention_impl` / `vocab_size`.
