# Architecture Guide

Last updated: 2026-05-24

Current state: the architecture is unchanged, but the effective data path is now repaired. Training uses sampled corpus windows, a held-out validation slice, and generation reuses the same tokenizer/corpus assumptions.

## High-level modules

- core/kernels.py: CUDA kernels for forward, backward, and optimizer updates
- core/ops.py: PyCUDA kernel wrappers and compilation/runtime glue
- core/loss.py: fused softmax cross-entropy
- model/gpt.py: GPTConfig, layers, model forward/backward/update/checkpoint logic
- tokenizer/tokenizer.py: CharacterGPTTokenizer compatibility wrapper around the hybrid tokenizer implementation
- corpus_utils.py: shared dataset loading, tokenizer construction, token cache
- gpu_memory.py: pooled allocator support
- training_metrics.py: compact per-step and final metrics logging

## Runtime data flow

1. Text corpus loaded from dataset source
2. Tokenizer built from selected docs using a hybrid word/piece vocabulary with character fallback
3. Token matrix created or loaded from output/cache/tokenizer
4. Input and target slices staged to GPU
5. Model forward produces logits
6. Loss kernel computes loss and dLogits
7. Backward computes gradients
8. AdamW updates weights
9. Held-out validation slice is evaluated and logged
10. Checkpoint and logs written to output

## Checkpoint metadata behavior

generate.py infers model config from checkpoint arrays and optional metadata fields. It can also use output/last_run_config.json to resolve missing metadata such as num_heads/attention_impl.
