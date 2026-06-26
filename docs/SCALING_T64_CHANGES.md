# Context Scaling Notes (T=64)

Last updated: 2026-05-24

Current state: the T=64 setup now samples across the corpus and keeps the same newline-aware tokenization assumptions used by generation.

## Current state

- train.py default sequence length is 64.
- auto_train.py default sequence length prompt is 64.
- generate.py derives max context from checkpoint config when checkpoint exists.

## Why this file exists

Earlier project stages focused on smaller context windows. Current training flows and config handling now support practical T=64 usage as a normal baseline.

## Compatibility reminder

When using checkpoints, generation uses checkpoint-derived config (vocab, context, embedding, heads, layers, attention impl). Keep tokenizer/checkpoint pairing consistent.
