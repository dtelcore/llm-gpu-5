"""Shared corpus loading and tokenizer construction utilities."""

import hashlib
import json
import os
import unicodedata

import numpy as np

from logging_config import logger

from tokenizer.tokenizer import CharacterGPTTokenizer


TRAINING_CORPUS = [
    "cuda training operational step sequence setup complete.",
    "gpu acceleration optimizes matrix transformations.",
    "manual backpropagation gradient loops updating parameter histories.",
]

FINEWEB_DATASET_NAME = "tinystories"
FINEWEB_PATH = "data/tinystories_500mb.txt"
DEFAULT_MAX_TRAINING_DOCS = 5000
TOKEN_CACHE_DIR = os.path.join("output", "cache", "tokenizer")


def normalize_corpus_text(text, preserve_trailing_newline=True):
    """Normalize corpus text while preserving the structural separators used for training."""
    normalized = unicodedata.normalize('NFKC', str(text))
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = normalized.replace('\u00a0', ' ')

    if preserve_trailing_newline:
        has_trailing_newline = normalized.endswith('\n')
        normalized = normalized.rstrip('\n')
        if normalized and has_trailing_newline:
            normalized = f"{normalized}\n"
    else:
        normalized = normalized.rstrip('\n')

    return normalized


def _read_text_corpus(file_path):
    documents = []
    with open(file_path, 'r', encoding='utf-8') as handle:
        for raw_line in handle:
            normalized = normalize_corpus_text(raw_line, preserve_trailing_newline=True)
            if normalized.strip():
                if not normalized.endswith('\n'):
                    normalized = f"{normalized}\n"
                documents.append(normalized)
    return documents


def _hash_text_items(items):
    digest = hashlib.sha256()
    for item in items:
        encoded_item = str(item).encode('utf-8')
        digest.update(len(encoded_item).to_bytes(8, byteorder='little', signed=False))
        digest.update(encoded_item)
    return digest.hexdigest()


def _token_cache_key(tokenizer, corpus, max_sequence_length, cache_namespace, dataset_name):
    key_payload = {
        "tokenizer_version": int(getattr(tokenizer, "TOKENIZER_VERSION", 1)),
        "cache_namespace": str(cache_namespace or "train"),
        "dataset_name": str(dataset_name or "custom"),
        "max_sequence_length": int(max_sequence_length),
        "vocab_size": int(tokenizer.vocab_size),
        "tokenizer_chars_sha256": _hash_text_items(tokenizer.uchars),
        "corpus_docs_sha256": _hash_text_items(corpus),
        "corpus_doc_count": int(len(corpus)),
    }
    payload = json.dumps(key_payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:24]


def load_or_build_token_matrix(tokenizer, corpus, max_sequence_length, cache_namespace="train", dataset_name=None):
    """Load a cached aligned token matrix or build and persist it on first use."""
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)

    cache_key = _token_cache_key(tokenizer, corpus, max_sequence_length, cache_namespace, dataset_name)
    cache_path = os.path.join(TOKEN_CACHE_DIR, f"{cache_namespace}_{cache_key}.npz")

    if os.path.exists(cache_path):
        with np.load(cache_path, allow_pickle=False) as cache_data:
            cached_matrix = cache_data["matrix"]
        logger.info(f"[CACHE] Loaded token matrix from {cache_path}")
        return cached_matrix

    logger.info(f"[CACHE] Building token matrix cache at {cache_path}")
    token_matrix = tokenizer.encode_batch_gpu_aligned(corpus, max_sequence_length=max_sequence_length)
    np.savez_compressed(cache_path, matrix=token_matrix.astype(np.int32, copy=False))
    return token_matrix


def split_corpus_for_validation(corpus, validation_fraction=0.05, minimum_validation_docs=16):
    """Split a corpus into train and held-out validation slices."""
    corpus = list(corpus)
    if not corpus:
        return [], []
    if len(corpus) == 1:
        return corpus, list(corpus)

    validation_count = max(int(round(len(corpus) * float(validation_fraction))), int(minimum_validation_docs))
    validation_count = min(validation_count, len(corpus) - 1)
    if validation_count <= 0:
        return corpus, []

    split_index = len(corpus) - validation_count
    return corpus[:split_index], corpus[split_index:]


def sample_token_batch(token_matrix, batch_size, seq_len, rng=None):
    """Sample a random batch of aligned training windows from the cached token matrix."""
    if token_matrix.ndim != 2:
        raise ValueError(f"Expected token_matrix rank 2, got shape {token_matrix.shape}")
    if token_matrix.shape[1] < seq_len + 1:
        raise ValueError(
            f"token_matrix width {token_matrix.shape[1]} is too small for seq_len={seq_len}"
        )

    rng = rng if rng is not None else np.random.default_rng()
    row_indices = rng.integers(0, token_matrix.shape[0], size=int(batch_size))
    sampled_rows = token_matrix[row_indices]
    input_tokens = sampled_rows[:, :seq_len].astype(np.int32, copy=True)
    target_tokens = sampled_rows[:, 1:seq_len + 1].astype(np.int32, copy=True)
    return input_tokens, target_tokens, row_indices


def resolve_dataset_path(dataset_name):
    """Resolve a dataset name or file reference to a concrete text file path."""
    dataset_name = dataset_name or FINEWEB_DATASET_NAME

    if str(dataset_name).startswith("minimal"):
        return None
    if dataset_name in {FINEWEB_DATASET_NAME, os.path.basename(FINEWEB_PATH)}:
        return FINEWEB_PATH
    if os.path.exists(str(dataset_name)):
        return str(dataset_name)
    if os.path.isabs(str(dataset_name)) and os.path.exists(str(dataset_name)):
        return str(dataset_name)

    candidate_path = os.path.join("data", str(dataset_name))
    if os.path.exists(candidate_path):
        return candidate_path
    return None


def load_dataset_corpus(dataset_name=FINEWEB_DATASET_NAME, limit=None):
    """Load a corpus from the shared dataset locations with minimal fallback."""
    dataset_path = resolve_dataset_path(dataset_name)

    if dataset_path is None:
        corpus = list(TRAINING_CORPUS)
        source = "minimal"
    else:
        corpus = _read_text_corpus(dataset_path)
        source = os.path.basename(dataset_path)

    if limit is not None:
        corpus = corpus[:int(limit)]
    return corpus, source


def build_shared_tokenizer(dataset_name=FINEWEB_DATASET_NAME, source_docs=None, limit=None, fallback_docs=None,
                           max_vocab_size=4096, min_piece_frequency=2):
    """Build a tokenizer from a shared vocabulary corpus for the selected dataset."""
    if source_docs is None:
        vocab_docs, source = load_dataset_corpus(dataset_name, limit=limit)
    else:
        vocab_docs = list(source_docs)
        if limit is not None:
            vocab_docs = vocab_docs[:int(limit)]
        source = str(dataset_name or "provided")

    if not vocab_docs:
        vocab_docs = list(fallback_docs) if fallback_docs else list(TRAINING_CORPUS)
        source = "fallback"

    return CharacterGPTTokenizer(
        vocab_docs,
        max_vocab_size=max_vocab_size,
        min_piece_frequency=min_piece_frequency,
    ), vocab_docs, source