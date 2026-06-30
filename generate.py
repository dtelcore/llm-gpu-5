"""
Autoregressive Text Generation Engine for Kepler GT 730 GPT Model.

Loads a trained model checkpoint and generates text character-by-character
using temperature-scaled sampling. Demonstrates inference-time execution
of the custom CUDA transformer without training overhead.

Target: NVIDIA GeForce GT 730 with trained model weights
"""

from corpus_utils import FINEWEB_DATASET_NAME
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
import numpy as np
import pycuda.driver as cuda

# Force initial system driver context initialization mappings
import env_config
import pycuda.autoinit

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from logging_config import logger
from corpus_utils import build_shared_tokenizer, load_dataset_corpus
from tokenizer.tokenizer import CharacterGPTTokenizer
from model.gpt import GPTConfig, GPTModel


DEFAULT_CHECKPOINT_PATH = None

LAST_RUN_CONFIG_PATH = "output/last_run_config.json"
CHECKPOINT_DIR = os.path.join("output", "checkpoints")


# Training corpus (same as train.py)
TRAINING_CORPUS = [
    "cuda training operational step sequence setup complete.",
    "gpu acceleration optimizes matrix transformations.",
    "manual backpropagation gradient loops updating parameter histories."
]


def _copy_last_logits_row(gpu_logits, t_length, vocab_size):
    """Copy only the last row of a (T, vocab_size) logits matrix from GPU to host."""
    host_last_logits = np.empty(vocab_size, dtype=np.float32)
    last_row_offset = (t_length - 1) * vocab_size * 4
    cuda.memcpy_dtoh(host_last_logits, int(gpu_logits) + last_row_offset)
    return host_last_logits


@dataclass
class GenerationMetrics:
    """Collect per-token generation performance statistics."""
    token_count: int = 0
    total_wall_time_s: float = 0.0
    total_kernel_time_ms: float = 0.0
    total_h2d_bytes: int = 0
    total_d2h_bytes: int = 0
    per_token_wall_ms: list = field(default_factory=list)
    per_token_kernel_ms: list = field(default_factory=list)
    gpu_input_reused: bool = False

    def record_step(self, wall_ms, kernel_ms, h2d_bytes, d2h_bytes):
        self.token_count += 1
        self.total_wall_time_s += wall_ms / 1000.0
        self.total_kernel_time_ms += kernel_ms
        self.total_h2d_bytes += h2d_bytes
        self.total_d2h_bytes += d2h_bytes
        self.per_token_wall_ms.append(wall_ms)
        self.per_token_kernel_ms.append(kernel_ms)

    def print_summary(self):
        if self.token_count == 0:
            return
        avg_wall = sum(self.per_token_wall_ms) / self.token_count
        avg_kernel = self.total_kernel_time_ms / self.token_count
        avg_h2d_kb = (self.total_h2d_bytes / self.token_count) / 1024
        avg_d2h_kb = (self.total_d2h_bytes / self.token_count) / 1024
        tok_per_s = self.token_count / self.total_wall_time_s if self.total_wall_time_s > 0 else 0.0
        logger.info(f"[PERF] Generated {self.token_count} tokens in {self.total_wall_time_s:.2f}s ({tok_per_s:.1f} tok/s)")
        logger.info(
            f"[PERF] Avg per-token: {avg_wall:.1f} ms "
            f"(kernel: {avg_kernel:.1f} ms, H2D: {avg_h2d_kb:.1f} KB, D2H: {avg_d2h_kb:.1f} KB)"
        )
        logger.info(f"[PERF] gpu_input buffer reused: {'yes' if self.gpu_input_reused else 'no'}")


def normalize_path(path):
    """Normalize a path for reliable comparisons on Windows."""
    return os.path.normcase(os.path.abspath(path))


def resolve_recommended_checkpoint(explicit_path=None):
    """Choose the most suitable checkpoint for inference."""
    if explicit_path:
        candidate_path = os.path.normpath(str(explicit_path))
        if os.path.exists(candidate_path):
            return candidate_path
        raise FileNotFoundError(f"Checkpoint not found: {candidate_path}")

    run_config = load_matching_run_config(DEFAULT_CHECKPOINT_PATH)
    if run_config is None and os.path.exists(LAST_RUN_CONFIG_PATH):
        try:
            with open(LAST_RUN_CONFIG_PATH, 'r', encoding='utf-8') as handle:
                run_config = json.load(handle)
        except Exception as exc:
            logger.warning(f"[WARN] Failed to read {LAST_RUN_CONFIG_PATH}: {exc}")

    if run_config is not None:
        for key in ("best_checkpoint_path", "checkpoint_path"):
            candidate_path = run_config.get(key)
            if candidate_path and os.path.exists(candidate_path):
                return os.path.normpath(candidate_path)

    if not os.path.isdir(CHECKPOINT_DIR):
        raise FileNotFoundError(f"Checkpoint directory not found: {CHECKPOINT_DIR}")

    checkpoint_names = [
        name for name in os.listdir(CHECKPOINT_DIR)
        if name.endswith(".npz") and not name.endswith(".tmp")
    ]
    if not checkpoint_names:
        raise FileNotFoundError(f"No checkpoint .npz files found in {CHECKPOINT_DIR}")

    def sort_key(file_name):
        return os.path.getmtime(os.path.join(CHECKPOINT_DIR, file_name))

    best_names = sorted(
        [name for name in checkpoint_names if name.endswith(".best.npz")],
        key=sort_key,
        reverse=True,
    )
    if best_names:
        return os.path.join(CHECKPOINT_DIR, best_names[0])

    latest_names = sorted(checkpoint_names, key=sort_key, reverse=True)
    return os.path.join(CHECKPOINT_DIR, latest_names[0])


def load_matching_run_config(checkpoint_path):
    """Load the last saved run config only when it matches the requested checkpoint."""
    if not os.path.exists(LAST_RUN_CONFIG_PATH):
        return None

    try:
        from run_config import RunConfig
        run_config = RunConfig.load(LAST_RUN_CONFIG_PATH)
    except Exception as exc:
        logger.warning(f"[WARN] Failed to read {LAST_RUN_CONFIG_PATH}: {exc}")
        return None

    candidate_paths = []

    configured_checkpoint = f"output/checkpoints/{run_config.name}_checkpoint.npz"
    if configured_checkpoint:
        candidate_paths.append(configured_checkpoint)
        
    best_checkpoint_path = f"output/checkpoints/{run_config.name}.best.npz"
    if best_checkpoint_path:
        candidate_paths.append(best_checkpoint_path)

    if not candidate_paths:
        return None

    requested_name = os.path.basename(checkpoint_path) if checkpoint_path else ""

    for candidate_path in candidate_paths:
        if checkpoint_path and normalize_path(candidate_path) == normalize_path(checkpoint_path):
            return run_config
        if os.path.basename(candidate_path) == requested_name:
            return run_config
    return None


def resolve_checkpoint_config(checkpoint_path, num_heads=None):
    """Infer model dimensions from the checkpoint and fill in num_heads from run metadata."""
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint_data:
        vocab_size, embedding_dim = checkpoint_data['wte'].shape
        max_len, position_embedding_dim = checkpoint_data['wpe'].shape
        block_ids = sorted({int(name.split('_')[1]) for name in checkpoint_data.files if name.startswith('block_')})
        checkpoint_num_heads = None
        checkpoint_attention_impl = None
        if '__meta_num_heads' in checkpoint_data.files:
            checkpoint_num_heads = int(np.asarray(checkpoint_data['__meta_num_heads']).item())
        if '__meta_max_len' in checkpoint_data.files:
            max_len = int(np.asarray(checkpoint_data['__meta_max_len']).item())
        if '__meta_num_layers' in checkpoint_data.files:
            num_layers = int(np.asarray(checkpoint_data['__meta_num_layers']).item())
        else:
            num_layers = len(block_ids)
        if '__meta_attention_impl' in checkpoint_data.files:
            checkpoint_attention_impl = str(np.asarray(checkpoint_data['__meta_attention_impl']).item())

    if position_embedding_dim != embedding_dim:
        raise ValueError(
            f"Checkpoint position embedding width mismatch: wte={embedding_dim}, wpe={position_embedding_dim}"
        )

    if num_layers == 0:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain any transformer block weights")

    run_config = load_matching_run_config(checkpoint_path)
    if num_heads is None and checkpoint_num_heads is not None:
        num_heads = checkpoint_num_heads
    if num_heads is None and run_config is not None:
        num_heads = run_config.num_heads

    attention_impl = checkpoint_attention_impl
    if attention_impl is None and run_config is not None:
        attention_impl = run_config.attention_impl
    if attention_impl is None:
        attention_impl = 'strided'

    if num_heads is None:
        raise ValueError(
            "Unable to determine num_heads for this checkpoint. Pass --num_heads or make sure "
            f"{LAST_RUN_CONFIG_PATH} matches the checkpoint you are loading."
        )

    if embedding_dim % num_heads != 0:
        raise ValueError(
            f"Checkpoint embedding_dim={embedding_dim} is not divisible by num_heads={num_heads}"
        )

    return GPTConfig(
        vocab_size=vocab_size,
        max_len=max_len,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attention_impl=attention_impl,
        dropout_prob=0.0
    )


def encode_token_ids(tokenizer, text):
    """Extract the flat integer token IDs from the tokenizer's debug-friendly return structure."""
    encoded = tokenizer.encode(text)
    token_ids = encoded[1] if isinstance(encoded, tuple) else encoded
    return [int(token_id) for token_id in token_ids]


def decode_token_ids(tokenizer, token_ids):
    """Extract plain decoded text from the tokenizer's `(text, logs)` return structure."""
    decoded = tokenizer.decode(token_ids)
    return decoded[0] if isinstance(decoded, tuple) else decoded


def _top_k_logits(tokenizer, logits, top_k=10):
    """Return the highest-scoring token IDs and decoded pieces for a logits row."""
    top_ids = np.argsort(logits)[-int(top_k):][::-1]
    results = []
    for rank, token_id in enumerate(top_ids, start=1):
        token_id = int(token_id)
        results.append({
            "rank": rank,
            "token_id": token_id,
            "piece": decode_token_ids(tokenizer, [token_id]),
            "logit": float(logits[token_id]),
        })
    return results


def run_generation_probes(checkpoint_path, prompt="the", memorization_prefix=None, top_k=10,
                          max_new_tokens=40, temperature=0.0, sampled_temperature=0.6,
                          sampled_top_p=0.9, sampled_repetition_penalty=1.15,
                          num_heads=None, source_docs=None, dataset_path=None):
    """Run the four core generation probes against a checkpoint and return structured results."""
    resolved_checkpoint_path = resolve_recommended_checkpoint(checkpoint_path)

    with GenerationSession(resolved_checkpoint_path, num_heads=num_heads, source_docs=source_docs, dataset_path=dataset_path) as session:
        probe_results = {
            "checkpoint_path": resolved_checkpoint_path,
            "tokenizer_roundtrip": [],
        }

        roundtrip_texts = ["the", " the", ".", "hello world", "\n\nline1\nline2\n"]
        for text in roundtrip_texts:
            entry = {"text": text}
            try:
                token_ids = encode_token_ids(session.tokenizer, text)
                entry["token_ids"] = token_ids
                entry["decoded"] = decode_token_ids(session.tokenizer, token_ids)
            except Exception as exc:
                entry["error"] = repr(exc)
            probe_results["tokenizer_roundtrip"].append(entry)

        greedy_result = session.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=sampled_repetition_penalty,
            repetition_window=64,
            stream=False,
        )
        probe_results["greedy_decode"] = {
            "prompt": prompt,
            "repetition_penalty": sampled_repetition_penalty,
            "completion": greedy_result["completion"],
            "text": greedy_result["text"],
            "generated_token_ids": greedy_result["generated_token_ids"],
        }

        sampled_result = session.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=sampled_temperature,
            top_p=sampled_top_p,
            repetition_penalty=sampled_repetition_penalty,
            repetition_window=32,
            rng=np.random.default_rng(12345),
            stream=False,
        )
        probe_results["sampled_decode"] = {
            "prompt": prompt,
            "temperature": sampled_temperature,
            "top_p": sampled_top_p,
            "repetition_penalty": sampled_repetition_penalty,
            "completion": sampled_result["completion"],
            "text": sampled_result["text"],
            "generated_token_ids": sampled_result["generated_token_ids"],
        }

        memorization_prompt = memorization_prefix if memorization_prefix is not None else prompt
        memorization_result = session.generate(
            memorization_prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            repetition_penalty=sampled_repetition_penalty,
            repetition_window=64,
            stream=False,
        )
        probe_results["memorization_prefix"] = {
            "prompt": memorization_prompt,
            "repetition_penalty": sampled_repetition_penalty,
            "completion": memorization_result["completion"],
            "text": memorization_result["text"],
            "generated_token_ids": memorization_result["generated_token_ids"],
        }

        prompt_ids = encode_token_ids(session.tokenizer, prompt)
        context_ids = prompt_ids[-session.config.max_len:]
        host_input = np.array([context_ids], dtype=np.int32)
        gpu_input = cuda.mem_alloc(host_input.nbytes)
        cuda.memcpy_htod(gpu_input, host_input)
        gpu_logits = session.model.forward(gpu_input, B=1, T=len(context_ids))
        host_last_logits = _copy_last_logits_row(gpu_logits, len(context_ids), session.config.vocab_size)
        gpu_logits.free()
        gpu_input.free()
        session.model.free_forward_caches()
        probe_results["logits_step_1"] = _top_k_logits(session.tokenizer, host_last_logits, top_k=top_k)

        return probe_results


def build_generation_model(tokenizer, checkpoint_path, num_heads=None):
    """Construct a GPT model that matches the checkpoint architecture when possible."""
    if os.path.exists(checkpoint_path):
        config = resolve_checkpoint_config(checkpoint_path, num_heads=num_heads)
        logger.info(
            f"[CFG] Using checkpoint config: vocab={config.vocab_size}, ctx={config.max_len}, "
            f"embed={config.embedding_dim}, heads={config.num_heads}, layers={config.num_layers}, attention={config.attention_impl}"
        )
    else:
        logger.warning(f"[WARN] Checkpoint not found at {checkpoint_path}; falling back to default untrained config")
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            max_len=64,
            embedding_dim=64,
            num_heads=num_heads or 2,
            num_layers=1,
            attention_impl='identity',
            dropout_prob=0.0
        )

    if tokenizer.vocab_size != config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab_size={tokenizer.vocab_size} does not match checkpoint vocab_size={config.vocab_size}"
        )

    return config, GPTModel(config)


def build_tokenizer_for_checkpoint(checkpoint_path, expected_vocab_size, source_docs=None, dataset_path=None):
    """Build the shared tokenizer using the specified dataset, prioritizing static vocabulary."""
    import re
    from pathlib import Path
    cp_path = Path(checkpoint_path)
    # Strip step/progress suffixes (.step10000.p100, .best) before .npz so that
    # the derived vocab name matches the run-level vocab file (e.g. vocab_..._003916.json)
    # rather than a non-existent per-step copy (vocab_..._003916.step10000.p100.json).
    stem = cp_path.name
    stem = re.sub(r'\.step\d+\.p\d+\.npz$', '.npz', stem)
    stem = stem.replace('.best.npz', '.npz')
    vocab_name = stem.replace('.npz', '.json')
    if vocab_name.startswith('training_'):
        vocab_name = vocab_name.replace('training_', 'vocab_', 1)
    elif vocab_name.startswith('gpt_'):
        vocab_name = vocab_name.replace('gpt_', 'vocab_', 1)
    else:
        vocab_name = "vocab_" + vocab_name
        
    vocab_path = cp_path.parent / vocab_name
    if vocab_path.exists():
        tokenizer = CharacterGPTTokenizer.load_vocab(str(vocab_path))
        if tokenizer.vocab_size == expected_vocab_size:
            return tokenizer
        else:
            logger.warning(f"[WARN] Static vocab at {vocab_path} size ({tokenizer.vocab_size}) != expected ({expected_vocab_size})")
    else:
        logger.warning(f"[WARNING] Static vocab file not found at {vocab_path}!")
        logger.info("[INFO] Falling back to legacy corpus fitting (Expect scrambled outputs)...")

    run_config = load_matching_run_config(checkpoint_path)
    
    # Priority: 1. CLI explicit argument, 2. Config file 'dataset', 3. Global default
    dataset_name = dataset_path or (run_config.get('dataset', FINEWEB_DATASET_NAME) if run_config else FINEWEB_DATASET_NAME)
    
    tokenizer_limit = None
    if source_docs is None and run_config is not None and run_config.get('corpus_size'):
        tokenizer_limit = int(run_config['corpus_size'])

    tokenizer, vocab_docs, vocab_source = build_shared_tokenizer(
        dataset_name,
        source_docs=source_docs,
        limit=tokenizer_limit,
        max_vocab_size=expected_vocab_size,
    )
    logger.info(f"Loaded shared tokenizer corpus: {len(vocab_docs):,} documents ({vocab_source})")
    if tokenizer.vocab_size == expected_vocab_size:
        return tokenizer

    if run_config is not None and run_config.get('corpus_size'):
        limited_tokenizer, limited_docs, limited_source = build_shared_tokenizer(
            dataset_name,
            limit=run_config['corpus_size'],
            max_vocab_size=expected_vocab_size,
        )
        if limited_tokenizer.vocab_size == expected_vocab_size:
            logger.warning(
                f"[WARN] Falling back to {len(limited_docs):,}-document vocab ({limited_source}) for checkpoint compatibility"
            )
            return limited_tokenizer

    if source_docs is not None:
        explicit_tokenizer, explicit_docs, explicit_source = build_shared_tokenizer(
            dataset_name,
            source_docs=source_docs,
            limit=len(source_docs),
            max_vocab_size=expected_vocab_size,
        )
        if explicit_tokenizer.vocab_size == expected_vocab_size:
            logger.warning(
                f"[WARN] Using explicit source docs ({len(explicit_docs):,}) for checkpoint compatibility"
            )
            return explicit_tokenizer

    raise ValueError(
        f"Tokenizer vocab_size={tokenizer.vocab_size} does not match checkpoint vocab_size={expected_vocab_size}"
    )


def free_model(model):
    """Release all GPU allocations owned by the GPT model."""
    if model is None:
        return

    model.embedding.wte.free()
    model.embedding.wpe.free()
    for block in model.blocks:
        block.ln_1_gamma.free()
        block.ln_1_beta.free()
        block.ln_2_gamma.free()
        block.ln_2_beta.free()
        block.attn.c_attn_w.free()
        block.attn.c_attn_b.free()
        block.attn.c_proj_w.free()
        block.attn.c_proj_b.free()
        block.mlp.c_fc_w.free()
        block.mlp.c_fc_b.free()
        block.mlp.c_proj_w.free()
        block.mlp.c_proj_b.free()
    model.ln_f_gamma.free()
    model.ln_f_beta.free()
    model.lm_head_w.free()
    model.free_persistent_buffers()


def load_training_corpus():
    """Load training corpus, preferring FineWeb if available."""
    try:
        corpus, corpus_source = load_dataset_corpus(FINEWEB_DATASET_NAME)
        if corpus_source != "minimal":
            logger.info(f"Loaded FineWeb dataset: {len(corpus):,} documents")
            return corpus
    except Exception as e:
        logger.warning(f"Failed to load FineWeb: {e}")

    logger.info("Using default training corpus")
    return TRAINING_CORPUS


def sample_next_token(host_logits, temperature=0.7, top_k=None, top_p=None,
                      repetition_penalty=1.0, recent_token_ids=None, rng=None, bos_token_id=None):
    """Sample a token index with temperature, top-k/top-p filtering, and repetition penalty."""
    logits = np.array(host_logits, dtype=np.float32, copy=True)
    recent_token_ids = recent_token_ids or []
    rng = rng if rng is not None else np.random.default_rng()
    
    if bos_token_id is not None and 0 <= bos_token_id < len(logits):
        logits[bos_token_id] = -np.inf

    if repetition_penalty and repetition_penalty > 1.0:
        token_counts = {}
        for token_id in recent_token_ids:
            token_id = int(token_id)
            token_counts[token_id] = token_counts.get(token_id, 0) + 1

        for token_id, count in token_counts.items():
            if 0 <= token_id < logits.size:
                penalty_factor = float(repetition_penalty) ** float(min(count, 8))
                if logits[token_id] < 0:
                    logits[token_id] *= penalty_factor
                else:
                    logits[token_id] /= penalty_factor

    if temperature <= 1e-5:
        return int(np.argmax(logits))

    logits = logits / max(float(temperature), 1e-5)

    if top_k is not None:
        top_k = int(top_k)
        if 0 < top_k < logits.size:
            keep_ids = np.argpartition(logits, -top_k)[-top_k:]
            filtered_logits = np.full_like(logits, -np.inf)
            filtered_logits[keep_ids] = logits[keep_ids]
            logits = filtered_logits

    if top_p is not None:
        top_p = float(top_p)
        if 0.0 < top_p < 1.0:
            sorted_ids = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_ids]
            finite_mask = np.isfinite(sorted_logits)
            if not np.any(finite_mask):
                return int(np.argmax(host_logits))
            sorted_logits = sorted_logits[finite_mask]
            sorted_ids = sorted_ids[finite_mask]
            shifted = sorted_logits - np.max(sorted_logits)
            probs = np.exp(shifted)
            probs /= np.sum(probs)
            cumulative = np.cumsum(probs)
            cutoff = np.searchsorted(cumulative, top_p, side='left')
            keep_ids = sorted_ids[:max(1, cutoff + 1)]
            filtered_logits = np.full_like(logits, -np.inf)
            filtered_logits[keep_ids] = logits[keep_ids]
            logits = filtered_logits

    shifted_logits = logits - np.max(logits)
    exps = np.exp(shifted_logits)
    probs = exps / np.sum(exps)
    return int(rng.choice(len(probs), p=probs))


class GenerationSession:
    """Keep a checkpoint and tokenizer loaded for repeated prompt testing."""

    def __init__(self, checkpoint_path=None, num_heads=None, source_docs=None, dataset_path=None):
        self.checkpoint_path = resolve_recommended_checkpoint(checkpoint_path)
        self.num_heads = num_heads
        self.source_docs = source_docs
        self.model = None
        self.config = resolve_checkpoint_config(self.checkpoint_path, num_heads=num_heads)
        self.tokenizer = build_tokenizer_for_checkpoint(
            self.checkpoint_path,
            self.config.vocab_size,
            source_docs=source_docs,
            dataset_path=dataset_path
        )
        self.config, self.model = build_generation_model(self.tokenizer, self.checkpoint_path, num_heads=num_heads)

        if not self.model.load_checkpoint(self.checkpoint_path):
            raise RuntimeError(f"Failed to load checkpoint: {self.checkpoint_path}")

    def generate(self, prompt, max_new_tokens=40, temperature=0.6, top_k=None, top_p=None,
                 repetition_penalty=1.0, repetition_window=32, rng=None, stream=False,
                 collect_metrics=False):
        """Generate a continuation from the supplied prompt."""
        rng = rng if rng is not None else np.random.default_rng()
        current_tokens = encode_token_ids(self.tokenizer, prompt)
        generated_token_ids = []
        vocab_size = self.config.vocab_size
        max_context = self.config.max_len

        metrics = GenerationMetrics() if collect_metrics else None
        gpu_input = cuda.mem_alloc(max_context * 4)
        if metrics is not None:
            metrics.gpu_input_reused = True

        kernel_start = cuda.Event()
        kernel_end = cuda.Event()

        try:
            for _ in range(max_new_tokens):
                step_start = time.perf_counter()
                context_window = current_tokens[-max_context:]
                current_t_length = len(context_window)

                host_input = np.array([context_window], dtype=np.int32)
                h2d_bytes = host_input.nbytes
                cuda.memcpy_htod(gpu_input, host_input)

                kernel_start.record()
                gpu_logits = self.model.forward(gpu_input, B=1, T=current_t_length)
                kernel_end.record()
                kernel_end.synchronize()
                kernel_ms = kernel_start.time_till(kernel_end)

                host_last_logits = _copy_last_logits_row(gpu_logits, current_t_length, vocab_size)
                d2h_bytes = vocab_size * 4
                gpu_logits.free()
                self.model.free_forward_caches()

                if metrics is not None:
                    wall_ms = (time.perf_counter() - step_start) * 1000.0
                    metrics.record_step(wall_ms, kernel_ms, h2d_bytes, d2h_bytes)

                recent_token_ids = current_tokens[-int(repetition_window):] if repetition_window else current_tokens
                next_token_id = sample_next_token(
                    host_last_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    recent_token_ids=recent_token_ids,
                    rng=rng,
                    bos_token_id=self.tokenizer.BOS_ID,
                )
                current_tokens.append(next_token_id)
                generated_token_ids.append(next_token_id)

                if stream:
                    print(decode_token_ids(self.tokenizer, [next_token_id]), end="", flush=True)
        finally:
            gpu_input.free()

        if metrics is not None:
            metrics.print_summary()

        result = {
            "prompt": prompt,
            "completion": decode_token_ids(self.tokenizer, generated_token_ids),
            "text": decode_token_ids(self.tokenizer, current_tokens),
            "generated_token_ids": generated_token_ids,
        }
        if metrics is not None:
            result["metrics"] = metrics
        return result

    def close(self):
        """Release the loaded model from GPU memory."""
        free_model(self.model)
        self.model = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def execute_inference_engine(prompt_seed="cuda ", max_new_tokens=40, temperature=0.6,
                             checkpoint_path=DEFAULT_CHECKPOINT_PATH, num_heads=None,
                             top_k=None, top_p=0.9, repetition_penalty=1.15, dataset_path=None):
    """Main inference orchestrator."""
    logger.info("="*73)
    logger.info("[INIT] INITIALIZING AUTOREGRESSIVE TEXT GENERATION ENGINE")
    logger.info("="*73)
    
    resolved_checkpoint_path = resolve_recommended_checkpoint(checkpoint_path)

    with GenerationSession(resolved_checkpoint_path, num_heads=num_heads, dataset_path=dataset_path) as session:
        logger.info(f"Vocabulary Size: {session.config.vocab_size}")
        logger.info(f"Max Context Length: {session.config.max_len}")
        logger.info(f"Generation Temperature: {temperature}")
        logger.info(f"Checkpoint: {resolved_checkpoint_path}")
        logger.info("[OK] Model architecture initialized")

        seed_token_ids = encode_token_ids(session.tokenizer, prompt_seed)
        logger.info(f"Input Seed Prompt: '{prompt_seed}'")
        logger.info(f"Seed Token IDs: {seed_token_ids}")
        logger.info("="*73)

        print(f"\n✍️ Generating: {prompt_seed}", end="", flush=True)
        session.generate(
            prompt_seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stream=True,
            collect_metrics=True,
        )
        print("\n")

        logger.info("="*73)
        logger.info("[OK] Generation complete!")
        logger.info("="*73)
        logger.info("🧹 Initiating physical layer allocation scrubbing...")
        logger.info("[OK] VRAM allocations cleared down safely.")


def generate_from_seed(checkpoint_path, prompt, max_tokens=50, temperature=0.6, num_heads=None,
                       top_k=None, top_p=0.9, repetition_penalty=1.15, source_docs=None, dataset_path=None):
    """Wrapper function for auto_train.py compatibility."""
    try:
        resolved_checkpoint_path = resolve_recommended_checkpoint(checkpoint_path)
        logger.info(f"[GEN] Loading checkpoint: {resolved_checkpoint_path}")

        with GenerationSession(resolved_checkpoint_path, num_heads=num_heads, source_docs=source_docs, dataset_path=dataset_path) as session:
            logger.info(f"[OK] Checkpoint loaded")
            logger.info(f"[GEN] Seed: '{prompt}' -> {len(encode_token_ids(session.tokenizer, prompt))} tokens")
            result = session.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stream=False,
            )
            logger.info(f"[OK] Generation complete: {len(result['generated_token_ids'])} generated tokens")
            return result["text"]
        
    except Exception as e:
        logger.error(f"[GEN] Generation failed: {e}")
        return None


def format_generation_probes(probe_results):
    """Format the generation probe results for console output or logs."""
    lines = []
    lines.append(f"Checkpoint: {probe_results.get('checkpoint_path')}")
    lines.append("Tokenizer roundtrip:")
    for item in probe_results.get("tokenizer_roundtrip", []):
        if "error" in item:
            lines.append(f"  - {item['text']!r}: ERROR {item['error']}")
        else:
            lines.append(f"  - {item['text']!r} -> {item['decoded']!r}")

    greedy = probe_results.get("greedy_decode", {})
    lines.append(f"Greedy decode [{greedy.get('prompt')!r}]: {greedy.get('text')!r}")

    sampled = probe_results.get("sampled_decode", {})
    if sampled:
        lines.append(
            f"Sampled decode [temp={sampled.get('temperature')}, top_p={sampled.get('top_p')}, rep_pen={sampled.get('repetition_penalty')}, prompt={sampled.get('prompt')!r}]: {sampled.get('text')!r}"
        )

    memorization = probe_results.get("memorization_prefix", {})
    lines.append(f"Memorization prefix [{memorization.get('prompt')!r}]: {memorization.get('text')!r}")

    lines.append("Logits step 1:")
    for item in probe_results.get("logits_step_1", []):
        lines.append(
            f"  {item['rank']:>2}. id={item['token_id']} piece={item['piece']!r} logit={item['logit']:.6f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from gpu_memory import install_global_memory_pool
    install_global_memory_pool()

    parser = argparse.ArgumentParser(description="Run GPT generation from a trained checkpoint")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to the trained checkpoint .npz file")
    parser.add_argument("--dataset", default=None,
                        help="Path to the dataset used to build the tokenizer vocabulary")
    parser.add_argument("--prompt", default="Once upon a time ", help="Seed prompt text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=40,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Optional top-k filter before sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p nucleus threshold for sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.15,
                        help="Repetition penalty applied to recent token IDs")
    parser.add_argument("--num_heads", type=int, default=None,
                        help="Override attention head count when checkpoint metadata is unavailable")
    args = parser.parse_args()

    execute_inference_engine(
        prompt_seed=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        checkpoint_path=args.checkpoint,
        num_heads=args.num_heads,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        dataset_path=args.dataset,
    )