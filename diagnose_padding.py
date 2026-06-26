#!/usr/bin/env python3
"""Diagnostic: Check padding ratio in encoded token matrix."""

import numpy as np
import os
from corpus_utils import build_shared_tokenizer, load_dataset_corpus, load_or_build_token_matrix
from tokenizer.hybrid_tokenizer import CharacterGPTTokenizer

# Load tiny stories
corpus, source = load_dataset_corpus("tiny_stories")
print(f"Loaded corpus: {len(corpus)} documents from {source}\n")

# Build tokenizer
print("Building tokenizer...")
tokenizer, _, _ = build_shared_tokenizer("tiny_stories", source_docs=corpus[:5000])
print(f"PAD_ID: {tokenizer.PAD_ID}")
print(f"BOS_ID: {tokenizer.BOS_ID}")
print(f"Vocab size: {tokenizer.vocab_size}\n")

# Show sample documents
print("Sample document lengths:")
for i, doc in enumerate(corpus[:20]):
    tokens, ids, _ = tokenizer.encode(doc)
    print(f"  Doc {i}: {len(ids):4} tokens | {len(doc):4} chars | Preview: {doc[:50]}...")

# Check encoding stats
print("\nDocuments encoded:")
all_ids = []
doc_lengths = []
for i, doc in enumerate(corpus[:1000]):
    if (i+1) % 200 == 0:
        print(f"  {i+1:4} / 1000")
    _, ids, _ = tokenizer.encode(doc)
    all_ids.extend(ids)
    doc_lengths.append(len(ids))

doc_lengths = np.array(doc_lengths)
print(f"\nDocument token counts:")
print(f"  Mean:   {doc_lengths.mean():.0f}")
print(f"  Median: {np.median(doc_lengths):.0f}")
print(f"  Min:    {doc_lengths.min()}")
print(f"  Max:    {doc_lengths.max()}")
print(f"  Std:    {doc_lengths.std():.1f}")

# Load actual token matrix
print("\nLoading token matrix (1024 context window)...")
token_matrix = load_or_build_token_matrix(
    tokenizer,
    corpus[:5000],
    max_sequence_length=1025,
    cache_namespace="test_diag",
    dataset_name="tiny_stories"
)

print(f"Token matrix shape: {token_matrix.shape}")

# Analyze padding
print("\nPadding analysis:")
rows_examined = min(1000, token_matrix.shape[0])
padding_counts = []
for i in range(rows_examined):
    row = token_matrix[i]
    pad_count = np.sum(row == tokenizer.PAD_ID)
    padding_counts.append(pad_count)

padding_counts = np.array(padding_counts)
padding_pct = padding_counts / 1025 * 100

print(f"Padding tokens per row (PAD_ID={tokenizer.PAD_ID}):")
print(f"  Mean:   {padding_counts.mean():.0f} tokens ({padding_pct.mean():.1f}%)")
print(f"  Median: {np.median(padding_counts):.0f} tokens ({np.median(padding_pct):.1f}%)")
print(f"  Min:    {padding_counts.min()} tokens ({padding_pct.min():.1f}%)")
print(f"  Max:    {padding_counts.max()} tokens ({padding_pct.max():.1f}%)")
print(f"  Rows with >50% padding: {np.sum(padding_pct > 50)} / {rows_examined} ({np.sum(padding_pct > 50) / rows_examined * 100:.1f}%)")

print("\n⚠️  CRITICAL ISSUE:")
print(f"   The model is spending {padding_pct.mean():.1f}% of its loss calculation")
print(f"   predicting PAD tokens (trivial task: PAD→PAD).")
print(f"   This artificially suppresses loss and makes PPL meaningless.")
print(f"\n   Solution: Add PAD token masking to the loss function.")
