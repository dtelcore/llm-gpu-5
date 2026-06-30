#!/usr/bin/env python3
"""
Interactive Training Configuration & Execution Script

Prompts user for all training parameters, then runs training with test generation.
"""

import os
import sys
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

# Ensure venv modules are available
import pycuda.driver as cuda
import env_config
import pycuda.autoinit

from logging_config import logger, setup_logging
from corpus_utils import (
    build_shared_tokenizer,
    load_dataset_corpus,
    load_or_build_token_matrix,
    sample_token_batch,
    split_corpus_for_validation,
)
from tokenizer.tokenizer import CharacterGPTTokenizer
from model.gpt import GPTConfig, GPTModel, validate_checkpoint_archive
from core.loss import SoftmaxCrossEntropy
from gpu_memory import install_global_memory_pool, get_memory_pool_stats_mb, free_held_pool_blocks
from training_metrics import TrainingMetrics, validate_model_config, estimate_vram_usage
import numpy as np
from scheduler import CosineWarmupScheduler


PARAMETER_PRESET_ASSUMED_VOCAB = 156
PARAMETER_PRESET_ASSUMED_CTX = 128
from run_config import RunConfig

# --- Per-preset "cost card" estimates (VRAM / throughput / representation risk) ---
# These are deliberately simple, deterministic heuristics (not measured benchmarks)
# calibrated against the vocab size this project actually trains against, so the
# AutoTrain menu can warn about config -> behavior mismatches (e.g. large vocab +
# small embedding_dim => token-boundary fragmentation) before a run is launched,
# not after a long training run produces unreadable text.
ESTIMATE_VOCAB_SIZE = 4096       # realistic deployed vocab size, independent of any one corpus
ESTIMATE_AVAILABLE_VRAM_MB = 3500  # GT730 4GB DDR3 budget used elsewhere in this file

# Throughput baseline: embed=32, 1 layer, seq_len=128 measured at ~2000 tok/s in
# project logs. Relative cost scales roughly with layers * embedding_dim^2 * seq_len
# (dominant term in the QKV/FFN matmul FLOP counts); num_heads does not change total
# FLOPs (just how they're split), so it is intentionally excluded from this estimate.
THROUGHPUT_BASELINE_TOKPS = 2000.0
THROUGHPUT_BASELINE_EMBED = 32
THROUGHPUT_BASELINE_LAYERS = 1
THROUGHPUT_BASELINE_SEQLEN = 128


def estimate_tokens_per_sec(embedding_dim, num_layers, seq_len=PARAMETER_PRESET_ASSUMED_CTX):
    """Rough relative throughput estimate, calibrated to one measured baseline.

    Not a guarantee -- actual tok/s also depends on batch size, grad_accum,
    kernel-launch overhead, and PCIe/host-side bottlenecks not modeled here.
    """
    relative_cost = (
        (num_layers / THROUGHPUT_BASELINE_LAYERS) *
        ((embedding_dim / THROUGHPUT_BASELINE_EMBED) ** 2) *
        (seq_len / THROUGHPUT_BASELINE_SEQLEN)
    )
    relative_cost = max(relative_cost, 1e-6)
    return THROUGHPUT_BASELINE_TOKPS / relative_cost


def embedding_vocab_ratio(embedding_dim, vocab_size=ESTIMATE_VOCAB_SIZE):
    """Representation-adequacy score: embedding dimensions available per bit of
    vocabulary entropy (log2(vocab_size)). Low ratios mean tokens are packed too
    tightly into the embedding space to stay well-separated -- the root cause of
    the token-boundary-fusion artifacts ("andthe", "Heher") observed in this project
    at vocab=4096 with embedding_dim=32."""
    import math
    return embedding_dim / max(1.0, math.log2(max(vocab_size, 2)))


def classify_collapse_risk(embedding_dim, num_layers, vocab_size=ESTIMATE_VOCAB_SIZE):
    """Heuristic collapse-risk label combining representation adequacy (embedding
    vs. vocab entropy) and depth (shallow models are more prone to repetition /
    attractor collapse, independent of embedding size)."""
    ratio = embedding_vocab_ratio(embedding_dim, vocab_size)
    if ratio < 3.0:
        risk = "HIGH (token-boundary fragmentation likely at this vocab size)"
    elif ratio < 5.0:
        risk = "MODERATE (borderline representational capacity)"
    else:
        risk = "LOW"
    if num_layers <= 1:
        risk += "; shallow depth raises repetition/attractor-collapse risk"
    return risk


def build_preset_cost_card(embedding_dim, num_heads, num_layers,
                            seq_len=PARAMETER_PRESET_ASSUMED_CTX,
                            vocab_size=ESTIMATE_VOCAB_SIZE):
    """Compute the full per-preset cost card: VRAM, throughput, and collapse risk,
    all estimated at a fixed reference vocab_size/seq_len so presets are directly
    comparable to each other in the AutoTrain menu."""
    model_vram_mb, training_vram_mb, total_vram_mb = estimate_vram_usage(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        batch_size=1,
        seq_len=seq_len,
    )
    return {
        "total_vram_mb": total_vram_mb,
        "vram_pct_of_budget": 100.0 * total_vram_mb / ESTIMATE_AVAILABLE_VRAM_MB,
        "tokens_per_sec": estimate_tokens_per_sec(embedding_dim, num_layers, seq_len),
        "embedding_ratio": embedding_vocab_ratio(embedding_dim, vocab_size),
        "collapse_risk": classify_collapse_risk(embedding_dim, num_layers, vocab_size),
    }

# Load model architecture presets from config file
MODEL_SIZE_PRESETS = RunConfig.load_presets(os.path.join(os.path.dirname(__file__), "config", "presets_gt730_v2.json"))

# Load presets with category information
def load_presets_with_categories(presets_path):
    """Load presets preserving category grouping."""
    with open(presets_path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "presets_by_category" in data:
        return data["presets_by_category"]
    return None

GOAL_LOSS_THRESHOLD = 2.0
GOAL_PPL_THRESHOLD = 5.0
GOAL_IMPROVEMENT_EPSILON = 1e-4
GOAL_EMA_ALPHA = 0.02
GOAL_MIN_STEP = 200
GOAL_TRAIN_WEIGHT = 0.4
GOAL_VAL_WEIGHT = 0.6
GOAL_VAL_TRUST_MAX_GAP = 1.5  # nats: if ema_val < ema_train - this, val is suspect and ignored
PROBE_CHECKPOINT_PROGRESS_MARKERS = (0.25,0.50,0.75,1.00)
PROBE_PROMPT = "Once upon a "
PROBE_MEMORIZATION_PREFIX_LEN = 64
GEN_DEFAULT_TEMPERATURE = 0.6
GEN_DEFAULT_TOP_P = 0.9
GEN_DEFAULT_REPETITION_PENALTY = 1.15
STAGE2_EARLIEST_AVG_LOSS = 3.0
STAGE2_STRONG_AVG_LOSS = 2.9
STAGE2_READABILITY_THRESHOLD = 0.55
STAGE2_MILESTONE_DIP_TOLERANCE = 0.05
DEFAULT_CORPUS_LIMIT = 5000
VAL_RESAMPLE_INTERVAL_STEPS = 25


def estimate_model_params(vocab_size, max_len, embedding_dim, num_layers):
    """Estimate parameter count using the repo's GPT architecture layout."""
    per_block_params = (
        4 * embedding_dim +
        embedding_dim * (3 * embedding_dim) +
        embedding_dim * embedding_dim +
        embedding_dim * (4 * embedding_dim) +
        (4 * embedding_dim) * embedding_dim +
        (3 * embedding_dim + embedding_dim + 4 * embedding_dim + embedding_dim)
    )

    total_params = 0
    total_params += vocab_size * embedding_dim
    total_params += max_len * embedding_dim
    total_params += num_layers * per_block_params
    total_params += 2 * embedding_dim
    total_params += embedding_dim * vocab_size
    return int(total_params)


def format_param_count(param_count):
    """Format parameter counts in K/M units for the interactive UI."""
    if param_count >= 1_000_000:
        return f"{param_count / 1_000_000:.2f}M"
    if param_count >= 1_000:
        return f"{param_count / 1_000:.1f}K"
    return str(param_count)


def format_lr_token(lr):
    """Create a filename-safe learning-rate token without hiding tiny values."""
    if lr == 0:
        return "0"
    if abs(lr) >= 1e-4:
        return f"{lr:.6f}".rstrip('0').rstrip('.').replace('.', 'p')
    return f"{lr:.1e}".replace('.', 'p').replace('+', '')


def format_model_label(model_name):
    """Extract clean model label (preset name or 'custom') without architecture suffix."""
    name_str = str(model_name or "custom").strip().lower()
    # If the name already contains architecture (e.g., "custom_128d_1l"), extract just the base
    if '_' in name_str and 'd' in name_str and 'l' in name_str:
        parts = name_str.split('_')
        return parts[0]  # Return just the base name (e.g., "custom")
    return name_str


def build_checkpoint_stem(model_name, embedding_dim, num_heads, num_layers, learning_rate, seq_len, total_steps):
    """Build canonical checkpoint stem: gpt_{steps}steps_{lr}lr_ctx{ctx}_{model_label}_{arch_suffix}_{timestamp}
    
    Returns the base stem without file extension or suffix modifiers.
    Derived filenames should be:
      {stem}.npz                      - main checkpoint
      {stem}.step{step}.p{percent}.npz - progress checkpoint
      {stem}.best.npz                 - best checkpoint
    And log files use:
      training_{stem_without_gpt_prefix}.log
    """
    model_label = format_model_label(model_name)
    arch_suffix = f"{int(embedding_dim)}d_{int(num_heads)}h_{int(num_layers)}l"
    lr_formatted = format_lr_token(learning_rate)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"gpt_{total_steps}steps_{lr_formatted}lr_ctx{seq_len}_{model_label}_{arch_suffix}_{timestamp}"


def format_model_token(model_name, embedding_dim, num_layers):
    """[DEPRECATED] Use build_checkpoint_stem instead. Kept for backwards compatibility."""
    name_token = str(model_name or "model").strip().lower().replace(' ', '_')
    return f"{name_token}_{int(embedding_dim)}d_{int(num_layers)}l"


class InteractiveTrainer:
    """Interactive training configuration and execution."""
    
    CONFIG_FILE = "output/last_run_config.json"
    
    def __init__(self):
        self.config = {}
        self.model = None
        self.tokenizer = None

    def refresh_run_artifact_names(self):
        """Regenerate log/checkpoint names from the current training settings using canonical stem builder."""
        checkpoint_stem = build_checkpoint_stem(
            model_name=self.config.get('name'),
            embedding_dim=self.config.get('embedding_dim'),
            num_heads=self.config.get('num_heads'),
            num_layers=self.config.get('num_layers'),
            learning_rate=self.config['learning_rate'],
            seq_len=self.config['seq_len'],
            total_steps=self.config['total_steps'],
        )

        # Derive all artifacts from the canonical checkpoint stem
        log_name = f"training_{checkpoint_stem.replace('gpt_', '', 1)}"
        checkpoint_name = checkpoint_stem

        self.config['log_name'] = log_name
        self.config['checkpoint_name'] = checkpoint_name
        self.config['checkpoint_path'] = f"output/checkpoints/{checkpoint_name}.npz"

    def _reset_goal_tracking(self):
        """Initialize goal-threshold tracking for the current training run."""
        self.config['goal_loss_threshold'] = GOAL_LOSS_THRESHOLD
        self.config['goal_ppl_threshold'] = GOAL_PPL_THRESHOLD
        self.config['best_checkpoint_path'] = None
        self.config['best_checkpoint_step'] = None
        self.config['best_goal_loss'] = None
        self.config['best_goal_ppl'] = None
        self.config['best_goal_score'] = None
        self.config['goal_ema_train_loss'] = None
        self.config['goal_ema_val_loss'] = None

    def _update_goal_stability_score(self, train_loss, val_loss=None):
        """Update EMA-smoothed losses and return a stability score for goal checkpointing."""
        alpha = GOAL_EMA_ALPHA

        ema_train = self.config.get('goal_ema_train_loss')
        train_loss = float(train_loss)
        ema_train = train_loss if ema_train is None else ((1.0 - alpha) * float(ema_train) + alpha * train_loss)
        self.config['goal_ema_train_loss'] = ema_train

        ema_val = self.config.get('goal_ema_val_loss')
        if val_loss is not None:
            val_loss = float(val_loss)
            ema_val = val_loss if ema_val is None else ((1.0 - alpha) * float(ema_val) + alpha * val_loss)
            self.config['goal_ema_val_loss'] = ema_val

        if ema_val is None:
            return float(ema_train)
        # If val EMA is suspiciously far below train EMA (>GOAL_VAL_TRUST_MAX_GAP nats),
        # the val batch is likely too easy / not representative — fall back to train-only score.
        gap = float(ema_train) - float(ema_val)
        if gap > GOAL_VAL_TRUST_MAX_GAP:
            logger.debug(
                f"[GOAL] val EMA suspect (ema_train={ema_train:.4f}, ema_val={ema_val:.4f}, gap={gap:.4f} > {GOAL_VAL_TRUST_MAX_GAP}); using train-only score"
            )
            return float(ema_train)
        return float((GOAL_TRAIN_WEIGHT * float(ema_train)) + (GOAL_VAL_WEIGHT * float(ema_val)))

    def _reset_stage_handoff_tracking(self):
        """Initialize Stage 1 -> Stage 2 handoff tracking for the current run."""
        story_bad_samples = self.config.get('story_bad_samples')
        self.config['stage2_earliest_avg_loss'] = STAGE2_EARLIEST_AVG_LOSS
        self.config['stage2_strong_avg_loss'] = STAGE2_STRONG_AVG_LOSS
        self.config['stage2_readability_threshold'] = STAGE2_READABILITY_THRESHOLD
        self.config['probe_milestone_scores'] = {}
        self.config['probe_milestone_readable'] = {}
        self.config['probe_milestone_steps'] = {}
        self.config['stage1_avg_loss'] = None
        self.config['stage2_handoff_ready'] = False
        self.config['stage2_handoff_report'] = None
        self.config['story_bad_samples'] = story_bad_samples

    def _goal_checkpoint_path(self):
        """Return the checkpoint filename used for goal-qualified best weights."""
        checkpoint_root, checkpoint_ext = os.path.splitext(self.config['checkpoint_path'])
        return f"{checkpoint_root}.best{checkpoint_ext}"

    def _preferred_checkpoint_path(self):
        """Return the checkpoint that should be treated as the best available artifact."""
        return self.config.get('best_checkpoint_path') or self.config['checkpoint_path']

    def _maybe_save_goal_checkpoint(self, step, loss_value, val_loss=None):
        """Save a best checkpoint when smoothed train/val score satisfies goals and improves."""
        if self.model is None:
            raise RuntimeError("Goal checkpoint requested before model initialization")

        current_score = self._update_goal_stability_score(loss_value, val_loss=val_loss)
        current_ppl = float(np.exp(current_score))
        goal_loss = self.config['goal_loss_threshold']
        goal_ppl = self.config['goal_ppl_threshold']

        if int(step) < GOAL_MIN_STEP:
            return False

        if current_score >= goal_loss or current_ppl >= goal_ppl:
            return False

        best_loss = self.config.get('best_goal_score')
        improvement = None
        if best_loss is not None:
            improvement = float(best_loss) - float(current_score)
            if improvement <= GOAL_IMPROVEMENT_EPSILON:
                return False

        best_checkpoint_path = self._goal_checkpoint_path()
        if best_loss is None:
            logger.info(
                f"[GOAL] Targets reached at step {step}: score={current_score:.4f}, ppl={current_ppl:.2f}, "
                f"ema_train={self.config.get('goal_ema_train_loss', current_score):.4f}, "
                f"ema_val={self.config.get('goal_ema_val_loss', current_score):.4f}"
            )
        else:
            logger.info(
                f"[GOAL] Improved best checkpoint at step {step}: score={current_score:.4f}, "
                f"ppl={current_ppl:.2f}, delta_score={improvement:.6f}, "
                f"ema_train={self.config.get('goal_ema_train_loss', current_score):.4f}, "
                f"ema_val={self.config.get('goal_ema_val_loss', current_score):.4f}"
            )
        logger.info(f"[GOAL] Saving best checkpoint to {best_checkpoint_path}...")
        self.model.save_checkpoint(best_checkpoint_path)
        logger.info("[GOAL] Best checkpoint saved")

        self.config['best_checkpoint_path'] = best_checkpoint_path
        self.config['best_checkpoint_step'] = step
        self.config['best_goal_loss'] = current_score
        self.config['best_goal_ppl'] = current_ppl
        self.config['best_goal_score'] = current_score
        self.save_config_to_json()
        return True

    def _log_goal_summary(self):
        """Emit a summary of the run against the requested goal thresholds."""
        logger.info(f"\nGoal Metrics:")
        logger.info(f"  Target loss:      < {self.config['goal_loss_threshold']:.2f}")
        logger.info(f"  Target PPL:       < {self.config['goal_ppl_threshold']:.2f}")
        if self.config.get('best_checkpoint_path'):
            logger.info(f"  Reached:          YES at step {self.config['best_checkpoint_step']}")
            logger.info(f"  Best loss:        {self.config['best_goal_loss']:.6f}")
            logger.info(f"  Best PPL:         {self.config['best_goal_ppl']:.2f}")
            logger.info(f"  Best checkpoint:  {self.config['best_checkpoint_path']}")
        else:
            logger.info(f"  Reached:          NO")

    def _run_generation_probes_for_checkpoint(self, checkpoint_path):
        """Run the shared tokenizer, greedy, memorization, and logits probes for a checkpoint."""
        from generate import format_generation_probes, run_generation_probes

        memorization_prefix = None
        if self.config.get('corpus'):
            memorization_prefix = self.config['corpus'][0][:PROBE_MEMORIZATION_PREFIX_LEN]

        probe_results = run_generation_probes(
            checkpoint_path,
            prompt=PROBE_PROMPT,
            memorization_prefix=memorization_prefix,
            top_k=10,
            max_new_tokens=self.config.get('gen_max_tokens', 40),
            temperature=0.0,
            sampled_temperature=self.config.get('gen_temperature', GEN_DEFAULT_TEMPERATURE),
            sampled_top_p=self.config.get('gen_top_p', GEN_DEFAULT_TOP_P),
            sampled_repetition_penalty=self.config.get('gen_repetition_penalty', GEN_DEFAULT_REPETITION_PENALTY),
            num_heads=self.config.get('num_heads'),
            source_docs=self.config.get('corpus'),
        )
        logger.info("\nGeneration probes:\n" + format_generation_probes(probe_results))
        return probe_results

    def _max_repeated_run(self, text):
        """Return the longest run of a repeated character in text."""
        if not text:
            return 0
        run = 1
        max_run = 1
        for idx in range(1, len(text)):
            if text[idx] == text[idx - 1]:
                run += 1
                if run > max_run:
                    max_run = run
            else:
                run = 1
        return max_run

    def _token_repetition_stats(self, text):
        """Return repetition stats for tokenized text."""
        tokens = [tok for tok in re.findall(r"\w+|[^\w\s]", text.lower()) if tok.strip()]
        token_count = len(tokens)
        if token_count < 2:
            return {
                'dominant_ratio': 0.0,
                'immediate_repeat_ratio': 0.0,
                'token_unique_ratio': 1.0 if token_count == 1 else 0.0,
            }

        counts = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        dominant_ratio = max(counts.values()) / token_count
        immediate_repeats = sum(1 for idx in range(1, token_count) if tokens[idx] == tokens[idx - 1])
        immediate_repeat_ratio = immediate_repeats / max(1, token_count - 1)
        token_unique_ratio = len(counts) / token_count
        return {
            'dominant_ratio': dominant_ratio,
            'immediate_repeat_ratio': immediate_repeat_ratio,
            'token_unique_ratio': token_unique_ratio,
        }

    def _score_text_readability(self, text, prompt):
        """Heuristic readability score for probe text, normalized to [0, 1]."""
        if not text:
            return 0.0

        completion = text
        if prompt and completion.startswith(prompt):
            completion = completion[len(prompt):]
        completion = completion.strip()
        if not completion:
            return 0.0

        length = len(completion)
        printable_ratio = sum(ch.isprintable() for ch in completion) / max(1, length)
        word_like_ratio = sum(ch.isalpha() or ch in " .,;:!?'-\n" for ch in completion) / max(1, length)
        unique_ratio = len(set(completion)) / max(1, length)
        longest_repeat = self._max_repeated_run(completion)
        token_stats = self._token_repetition_stats(completion)

        length_score = min(length / 40.0, 1.0)
        diversity_score = max(0.0, min(1.0, (unique_ratio - 0.08) / 0.30))
        repeat_penalty = min(max(0.0, longest_repeat - 4) / 12.0, 1.0)
        token_repeat_penalty = min(max(0.0, token_stats['dominant_ratio'] - 0.22) / 0.28, 1.0)
        adjacent_repeat_penalty = min(max(0.0, token_stats['immediate_repeat_ratio'] - 0.05) / 0.30, 1.0)
        token_diversity_bonus = max(0.0, min(1.0, (token_stats['token_unique_ratio'] - 0.20) / 0.60))

        score = (
            0.20 * length_score +
            0.25 * printable_ratio +
            0.25 * word_like_ratio +
            0.10 * diversity_score +
            0.10 * token_diversity_bonus -
            0.35 * repeat_penalty -
            0.40 * token_repeat_penalty -
            0.45 * adjacent_repeat_penalty
        )
        score = float(max(0.0, min(1.0, score)))

        # Hard cap obvious repetition loops like "the the the ..." even if other features look normal.
        if token_stats['immediate_repeat_ratio'] >= 0.22 or token_stats['dominant_ratio'] >= 0.45:
            score = min(score, 0.25)
        return score

    def _record_probe_milestone(self, progress_pct, step, probe_results):
        """Capture probe readability/trend metrics for Stage 2 handoff decisions."""
        greedy = probe_results.get('greedy_decode', {})
        sampled = probe_results.get('sampled_decode', {})
        memorization = probe_results.get('memorization_prefix', {})
        prompt = greedy.get('prompt', PROBE_PROMPT)

        scores = {
            'greedy': self._score_text_readability(greedy.get('text', ''), prompt),
            'sampled': self._score_text_readability(sampled.get('text', ''), sampled.get('prompt', prompt)),
            'memorization': self._score_text_readability(memorization.get('text', ''), memorization.get('prompt', prompt)),
        }
        aggregate_score = float(np.mean(list(scores.values())))
        is_readable = aggregate_score >= self.config.get('stage2_readability_threshold', STAGE2_READABILITY_THRESHOLD)

        milestone_key = str(int(progress_pct) / 100)
        self.config['probe_milestone_steps'][milestone_key] = int(step)
        self.config['probe_milestone_scores'][milestone_key] = aggregate_score
        self.config['probe_milestone_readable'][milestone_key] = bool(is_readable)

        logger.info(
            f"[HANDOFF] Probe {milestone_key}% score={aggregate_score:.3f} "
            f"(greedy={scores['greedy']:.3f}, sampled={scores['sampled']:.3f}, memorization={scores['memorization']:.3f}) "
            f"readable={'YES' if is_readable else 'NO'}"
        )

    def _build_stage2_handoff_report(self):
        """Build a structured Stage 2 readiness report from run metrics and probes."""
        expected_markers = [0.25,0.50,0.75,1.00]
        milestone_scores = self.config.get('probe_milestone_scores', {})
        milestone_readable = self.config.get('probe_milestone_readable', {})
        avg_loss = self.config.get('stage1_avg_loss')
        earliest_target = self.config.get('stage2_earliest_avg_loss', STAGE2_EARLIEST_AVG_LOSS)
        strong_target = self.config.get('stage2_strong_avg_loss', STAGE2_STRONG_AVG_LOSS)

        have_all_markers = all(str(marker) in milestone_scores for marker in expected_markers)
        probes_readable = have_all_markers and all(milestone_readable.get(str(marker), False) for marker in expected_markers)

        probe_trend_ok = False
        ordered_scores = []
        if have_all_markers:
            ordered_scores = [float(milestone_scores[str(marker)]) for marker in expected_markers]
            probe_trend_ok = all(
                ordered_scores[idx] + STAGE2_MILESTONE_DIP_TOLERANCE >= ordered_scores[idx - 1]
                for idx in range(1, len(ordered_scores))
            ) and ordered_scores[-1] >= ordered_scores[0]

        loss_ready = avg_loss is not None and float(avg_loss) <= float(earliest_target)
        strong_loss = avg_loss is not None and float(avg_loss) <= float(strong_target)

        story_bad_samples = self.config.get('story_bad_samples')
        story_bad_samples_ready = story_bad_samples == 0

        ready = bool(loss_ready and probes_readable and probe_trend_ok and story_bad_samples_ready)

        return {
            'ready': ready,
            'avg_loss': None if avg_loss is None else float(avg_loss),
            'earliest_target': float(earliest_target),
            'strong_target': float(strong_target),
            'loss_ready': bool(loss_ready),
            'strong_loss': bool(strong_loss),
            'have_all_probe_markers': bool(have_all_markers),
            'probe_scores': ordered_scores,
            'probes_readable': bool(probes_readable),
            'probe_trend_ok': bool(probe_trend_ok),
            'story_bad_samples': story_bad_samples,
            'story_bad_samples_ready': bool(story_bad_samples_ready),
        }

    def _log_stage2_handoff_summary(self, report):
        """Log the Stage 2 readiness summary in one compact block."""
        logger.info("\nStage 1 -> Stage 2 Handoff:")
        if report['avg_loss'] is None:
            logger.info("  Stage 1 avg loss:    n/a")
        else:
            logger.info(
                f"  Stage 1 avg loss:    {report['avg_loss']:.4f} "
                f"(target <= {report['earliest_target']:.2f}, strong <= {report['strong_target']:.2f})"
            )
        logger.info(f"  Probe markers ready: {'YES' if report['have_all_probe_markers'] else 'NO'}")
        logger.info(f"  Probe readable:      {'YES' if report['probes_readable'] else 'NO'}")
        logger.info(f"  Probe trend ok:      {'YES' if report['probe_trend_ok'] else 'NO'}")
        logger.info(f"  story bad_samples=0: {'YES' if report['story_bad_samples_ready'] else 'NO'}")
        logger.info(f"  Stage 2 ready:       {'YES' if report['ready'] else 'NO'}")

    def _save_checkpoint_with_probes(self, checkpoint_path, label):
        """Save a checkpoint and immediately run the generation probes on it."""
        if self.model is None:
            raise RuntimeError("Probe checkpoint requested before model initialization")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        logger.info(f"[{label}] Saving checkpoint to {checkpoint_path}...")
        self.model.save_checkpoint(checkpoint_path)
        logger.info(f"[{label}] Checkpoint saved")
        return self._run_generation_probes_for_checkpoint(checkpoint_path)

    def _build_probe_milestones(self, total_steps):
        """Return unique integer step milestones for 25/50/75/100 percent progress."""
        milestones = set()
        for marker in PROBE_CHECKPOINT_PROGRESS_MARKERS:
            step = int(round(total_steps * marker))
            step = max(1, min(int(total_steps), step))
            milestones.add(step)
        return sorted(milestones)

    def _format_probe_milestone_summary(self, total_steps):
        """Return a single startup log line describing resolved probe milestones."""
        resolved_steps = []
        summary_parts = []
        for marker in PROBE_CHECKPOINT_PROGRESS_MARKERS:
            marker_pct = int(round(marker * 100))
            step = int(round(total_steps * marker))
            step = max(1, min(int(total_steps), step))
            resolved_steps.append(step)
            summary_parts.append(f"{marker_pct}%->step{step}")
        return ", ".join(summary_parts), set(resolved_steps)
        
    def save_config_to_json(self):
        """Save current training config to JSON file for reuse."""
        try:
            os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
            rc = RunConfig(
                vocab_size=self.tokenizer.vocab_size if self.tokenizer else 4096,
                max_len=self.config.get('seq_len', 128),
                embedding_dim=self.config.get('embedding_dim', 64),
                num_heads=self.config.get('num_heads', 2),
                num_layers=self.config.get('num_layers', 1),
                attention_impl=self.config.get('attention_impl', 'strided'),
                batch_size=self.config.get('batch_size', 1),
                grad_accum=self.config.get('grad_accum', 64),
                learning_rate=self.config.get('learning_rate', 0.015),
                total_steps=self.config.get('total_steps', 1000),
                dataset=self.config.get('dataset', 'fineweb'),
                corpus_limit=self.config.get('corpus_size', 3000000),
                name=self.config.get('name', 'gpt_model'),
                preset_name=self.config.get('preset_name', 'custom'),
                init_checkpoint_path=self.config.get('init_checkpoint_path', None)
            )
            rc.save(self.CONFIG_FILE)
            logger.info(f"[OK] Saved run config to {self.CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Failed to save config: {e}")

    def _validate_init_checkpoint(self, checkpoint_path, config):
        """Ensure an initialization checkpoint matches the active model config."""
        checkpoint_path, checkpoint_config = self._resolve_init_checkpoint_config(checkpoint_path)
        config_fields = [
            ('vocab_size', 'vocab'),
            ('max_len', 'ctx'),
            ('embedding_dim', 'embed'),
            ('num_heads', 'heads'),
            ('num_layers', 'layers'),
            ('attention_impl', 'attention'),
        ]
        mismatches = []
        for attr_name, label in config_fields:
            active_value = getattr(config, attr_name)
            checkpoint_value = getattr(checkpoint_config, attr_name)
            if checkpoint_value != active_value:
                mismatches.append(f"{label} checkpoint={checkpoint_value} run={active_value}")

        if mismatches:
            mismatch_text = ", ".join(mismatches)
            raise ValueError(f"Initialization checkpoint config mismatch: {mismatch_text}")

        return checkpoint_path

    def _resolve_init_checkpoint_config(self, checkpoint_path):
        """Load checkpoint metadata needed for prompt-time compatibility checks."""
        checkpoint_path = os.path.normpath(str(checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Initialization checkpoint not found: {checkpoint_path}")

        validate_checkpoint_archive(checkpoint_path)

        from generate import resolve_checkpoint_config

        checkpoint_config = resolve_checkpoint_config(checkpoint_path)
        return checkpoint_path, checkpoint_config

    def _init_checkpoint_prompt_mismatches(self, checkpoint_config):
        """Return prompt-time config differences between the selected run and checkpoint.

        vocab_size is intentionally included here so the user is asked to adopt the
        checkpoint's vocabulary at selection time rather than crashing at training time.
        """
        config_fields = [
            ('seq_len', 'ctx', checkpoint_config.max_len),
            ('embedding_dim', 'embed', checkpoint_config.embedding_dim),
            ('num_heads', 'heads', checkpoint_config.num_heads),
            ('num_layers', 'layers', checkpoint_config.num_layers),
            ('attention_impl', 'attention', checkpoint_config.attention_impl),
            ('vocab_size', 'vocab', checkpoint_config.vocab_size),
        ]

        mismatches = []
        for config_key, label, checkpoint_value in config_fields:
            run_value = self.config.get(config_key)
            if run_value is not None and run_value != checkpoint_value:
                mismatches.append(f"{label} checkpoint={checkpoint_value} run={run_value}")
        return mismatches

    def _infer_model_name_from_checkpoint(self, checkpoint_config):
        """Map checkpoint dimensions back to the nearest preset label when possible.
        
        Returns only the model label (e.g., 'custom'), without architecture suffix.
        Architecture is appended separately in build_checkpoint_stem().
        """
        return "custom"

    def _apply_init_checkpoint_model_config(self, checkpoint_config):
        """Align the requested run with the selected checkpoint architecture.

        vocab_size is adopted here so the tokenizer preflight and model config
        both use the checkpoint's vocabulary size instead of a corpus-derived one.
        """
        self.config['name'] = self._infer_model_name_from_checkpoint(checkpoint_config)
        self.config['embedding_dim'] = checkpoint_config.embedding_dim
        self.config['num_heads'] = checkpoint_config.num_heads
        self.config['num_layers'] = checkpoint_config.num_layers
        self.config['attention_impl'] = checkpoint_config.attention_impl
        self.config['seq_len'] = checkpoint_config.max_len
        self.config['vocab_size'] = checkpoint_config.vocab_size
        self.config['approx_params'] = estimate_model_params(
            PARAMETER_PRESET_ASSUMED_VOCAB,
            checkpoint_config.max_len,
            checkpoint_config.embedding_dim,
            checkpoint_config.num_layers,
        )
    
    def load_config_from_json(self):
        """Load previous training config from JSON file."""
        try:
            if os.path.exists(self.CONFIG_FILE):
                rc = RunConfig.load(self.CONFIG_FILE)
                self.config.update({
                    'dataset': rc.dataset,
                    'corpus_size': rc.corpus_limit,
                    'batch_size': rc.batch_size,
                    'grad_accum': rc.grad_accum,
                    'preset_name': rc.preset_name,
                    'init_checkpoint_path': rc.init_checkpoint_path,
                    'name': rc.name,
                    'embedding_dim': rc.embedding_dim,
                    'num_heads': rc.num_heads,
                    'num_layers': rc.num_layers,
                    'seq_len': rc.max_len,
                    'total_steps': rc.total_steps,
                    'learning_rate': rc.learning_rate,
                    'attention_impl': rc.attention_impl,
                    'log_name': f"training_{rc.total_steps}steps",
                    'checkpoint_name': f"{rc.name}_checkpoint",
                    'checkpoint_path': f"output/checkpoints/{rc.name}_checkpoint.npz",
                    'test_generation': True,
                    'gen_prompt': "the ",
                    'gen_max_tokens': 40
                })
                
                # Reload corpus based on dataset name
                self._reload_corpus_for_dataset(self.config.get('dataset', 'minimal'))
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return False
    
    def _reload_corpus_for_dataset(self, dataset_name):
        """Reload corpus from file based on dataset name."""
        corpus, _ = load_dataset_corpus(dataset_name)
        saved_limit = self.config.get('corpus_limit') or self.config.get('corpus_size')
        if saved_limit is not None:
            saved_limit = int(saved_limit)
            if saved_limit > 0 and len(corpus) > saved_limit:
                corpus = corpus[:saved_limit]
        self.config['corpus'] = corpus
        
    def prompt_dataset(self):
        """Select dataset interactively."""
        print("\n" + "="*73)
        print("STEP 1: SELECT DATASET")
        print("="*73)
        
        datasets = {
            "1": ("minimal", 3),
            "2": ("fineweb", None),
            "3": ("custom", None),
        }
        
        print("\nAvailable datasets:")
        print("  1) Minimal       - 3 test sentences (quick training)")
        print("  2) FineWeb 100MB - Real web text (recommend for quality)")
        print("  3) Custom        - Load from data/*.txt file")
        
        choice = input("\nSelect dataset [1-3] (default: 2): ").strip() or "2"
        
        if choice == "1":
            dataset_name, num_docs = datasets["1"]
            corpus, _ = load_dataset_corpus("minimal")
            logger.info(f"[OK] Loaded minimal dataset: {len(corpus)} documents")
            
        elif choice == "2":
            dataset_name, _ = datasets["2"]
            fineweb_path = "data/fineweb_100mb.txt"
            if os.path.exists(fineweb_path):
                corpus, _ = load_dataset_corpus("fineweb")
                logger.info(f"[OK] Loaded FineWeb dataset: {len(corpus):,} documents")
            else:
                print("[WARN] FineWeb file not found, using minimal instead")
                corpus, _ = load_dataset_corpus("minimal")
                dataset_name = "minimal (fallback)"
                
        elif choice == "3":
            print("\nAvailable custom datasets in data/:")
            custom_files = list(Path("data").glob("*.txt"))
            if not custom_files:
                print("  [NONE] No custom datasets found")
                corpus = datasets["1"][1]
                dataset_name = "minimal (fallback)"
            else:
                for i, f in enumerate(custom_files, 1):
                    size_mb = f.stat().st_size / (1024*1024)
                    print(f"  {i}) {f.name:30} ({size_mb:.1f} MB)")
                file_idx = input(f"\nSelect file [1-{len(custom_files)}]: ").strip()
                try:
                    file_idx = int(file_idx) - 1
                    custom_file = custom_files[file_idx]
                    corpus, _ = load_dataset_corpus(str(custom_file))
                    dataset_name = custom_file.name
                    logger.info(f"[OK] Loaded custom dataset: {len(corpus):,} documents")
                except (ValueError, IndexError):
                    print("[ERR] Invalid selection, using minimal")
                    corpus, _ = load_dataset_corpus("minimal")
                    dataset_name = "minimal (fallback)"
        else:
            print("[ERR] Invalid choice, using minimal")
            corpus, _ = load_dataset_corpus("minimal")
            dataset_name = "minimal (fallback)"
        
        self.config['dataset'] = dataset_name
        self.config['corpus'] = corpus
        
        # Ask about corpus limiting
        if len(corpus) > 5000:
            total_docs = len(corpus)
            print(f"\nDataset has {total_docs:,} documents.")
            limit_str = input(
                f"Limit to how many docs? (default: {DEFAULT_CORPUS_LIMIT}, max/all = full corpus): "
            ).strip()

            if not limit_str:
                limit = DEFAULT_CORPUS_LIMIT
            elif limit_str.lower() in {"max", "all"}:
                limit = total_docs
            else:
                parsed_limit = limit_str.replace(",", "").replace("_", "")
                try:
                    limit = int(parsed_limit)
                    if limit <= 0:
                        raise ValueError("limit must be positive")
                except ValueError:
                    logger.warning(
                        f"[WARN] Invalid doc limit {limit_str!r}; using default {DEFAULT_CORPUS_LIMIT:,}"
                    )
                    limit = DEFAULT_CORPUS_LIMIT

            limit = min(limit, total_docs)
            corpus = corpus[:limit]
            self.config['corpus'] = corpus
            if limit == total_docs:
                logger.info(f"  [OK] Using all {len(corpus):,} documents")
            else:
                logger.info(f"  [OK] Using first {len(corpus):,} documents")
    
    def prompt_model_config(self):
        """Configure model architecture."""
        print("\n" + "="*73)
        print("STEP 2: MODEL ARCHITECTURE")
        print("="*73)

        # Load presets with categories
        presets_path = os.path.join(os.path.dirname(__file__), "config", "presets_gt730_v2.json")
        categories = load_presets_with_categories(presets_path)
        
        if not categories:
            print("[ERROR] Could not load model presets from config file")
            return

        # Step 1: Select category
        print("\n⚠️  GT730 KEPLER NOTE: Models scale from shallow (testing) to head_above_water (experimental).")
        print("[INFO] Start with 'shallow' for quick validation, scale up as needed.")
        print("\nModel complexity tiers:")
        tier_names = list(categories.keys())
        for i, tier in enumerate(tier_names, 1):
            preset_count = len(categories[tier])
            print(f"  {i}) {tier.replace('_', ' ').upper():25} ({preset_count} models, increasing complexity)")

        default_tier = "1"
        tier_choice = input(f"\nSelect tier [1-{len(tier_names)}] (default: {default_tier}): ").strip() or default_tier
        
        try:
            tier_idx = int(tier_choice) - 1
            if tier_idx < 0 or tier_idx >= len(tier_names):
                tier_idx = 0
                print(f"[WARN] Invalid choice, using tier 1")
            selected_tier = tier_names[tier_idx]
        except ValueError:
            selected_tier = tier_names[0]
            print(f"[WARN] Invalid choice, using tier 1")

        tier_presets = categories[selected_tier]
        presets = {}
        for preset in tier_presets:
            config = preset.copy()
            config['approx_params'] = estimate_model_params(
                PARAMETER_PRESET_ASSUMED_VOCAB,
                PARAMETER_PRESET_ASSUMED_CTX,
                config['embedding_dim'],
                config['num_layers'],
            )
            presets[config['key']] = config

        print(f"\n[OK] Selected tier: {selected_tier.replace('_', ' ').upper()}")
        print(f"[INFO] Preset parameter counts are approximate (vocab~{PARAMETER_PRESET_ASSUMED_VOCAB}, ctx={PARAMETER_PRESET_ASSUMED_CTX}).")
        print(f"[INFO] Cost-card estimates below (VRAM/speed/risk) are calibrated at the project's "
              f"real deployed vocab (~{ESTIMATE_VOCAB_SIZE}) and ctx={PARAMETER_PRESET_ASSUMED_CTX}, "
              f"and are rough heuristics, not measured benchmarks.")
        print("\nAvailable models in this tier:")

        for preset in tier_presets:
            config = presets[preset['key']]
            recommended = " [RECOMMENDED]" if preset['key'] == "3" else ""
            cost_card = build_preset_cost_card(
                config['embedding_dim'], config['num_heads'], config['num_layers']
            )
            print(
                f"  {preset['key']}) {config['name'].upper():7} - "
                f"{format_param_count(config['approx_params']):>7} params | "
                f"{config['embedding_dim']:>3}D, {config['num_heads']:>2} heads, {config['num_layers']} layer{'s' if config['num_layers'] > 1 else ''} | "
                f"{config['note']}{recommended}"
            )
            print(
                f"       VRAM: ~{cost_card['total_vram_mb']:.0f}MB ({cost_card['vram_pct_of_budget']:.0f}% of {ESTIMATE_AVAILABLE_VRAM_MB}MB budget) | "
                f"Speed: ~{cost_card['tokens_per_sec']:.0f} tok/s (est.) | "
                f"Embed/vocab ratio: {cost_card['embedding_ratio']:.1f}"
            )
            print(f"       Collapse risk @ vocab~{ESTIMATE_VOCAB_SIZE}: {cost_card['collapse_risk']}")

        default_choice = tier_presets[0]['key']
        choice = input(f"\nSelect model [{tier_presets[0]['key']}-{tier_presets[-1]['key']}] (default: {default_choice}): ").strip() or default_choice

        if choice in presets:
            config = presets[choice].copy()
        else:
            config = presets[default_choice].copy()
            print(f"[WARN] Invalid choice, using {config['name']}")

        print(f"\n[OK] Selected {config['name'].upper()}")
        print(f"     Approx params: {format_param_count(config['approx_params'])} (vocab~{PARAMETER_PRESET_ASSUMED_VOCAB}, ctx={PARAMETER_PRESET_ASSUMED_CTX})")
        print(f"     Embedding dim: {config['embedding_dim']}")
        print(f"     Num heads: {config['num_heads']}")
        print(f"     Num layers: {config['num_layers']}")

        self.config.update({
            'name': config['name'],
            'approx_params': config['approx_params'],
            'embedding_dim': config['embedding_dim'],
            'num_heads': config['num_heads'],
            'num_layers': config['num_layers'],
            'attention_impl': 'strided',
        })
    
    def prompt_training_params(self):
        """Configure training hyperparameters."""
        print("\n" + "="*73)
        print("STEP 3: TRAINING HYPERPARAMETERS")
        print("="*73)

        approx_params = self.config.get('approx_params', 0)
        if approx_params <= 100_000:
            lr_default = "0.01"
            lr_note = "(~10K-100K params: safe range 0.01-0.05)"
        elif approx_params <= 500_000:
            lr_default = "0.005"
            lr_note = "(~100K-500K params: safe range 0.005-0.02)"
        elif approx_params <= 2_000_000:
            lr_default = "0.002"
            lr_note = "(~0.5M-2M params: safe range 0.001-0.01)"
        else:
            lr_default = "0.001"
            lr_note = "(2M-10M params: safe range 0.0005-0.005, experimental on GT730)"
        
        # Learning rate
        print(f"\nLearning rate {lr_note}")
        print("  Tip: If loss plateaus (not changing), increase LR")
        print("  Tip: If NaN occurs, decrease LR and use gradient clipping")
        lr_str = input(f"Learning rate (default: {lr_default}): ").strip() or lr_default
        try:
            self.config['learning_rate'] = float(lr_str)
        except ValueError:
            self.config['learning_rate'] = float(lr_default)
            print(f"[WARN] Invalid LR, using {lr_default}")
        
        # Number of steps
        steps_str = input("Training steps (default: 100): ").strip() or "100"
        try:
            self.config['total_steps'] = int(steps_str)
        except ValueError:
            self.config['total_steps'] = 100
            print(f"[WARN] Invalid steps, using 100")
        
        # Sequence length
        seq_len_str = input("Sequence length T (default: 128): ").strip() or "128"
        try:
            self.config['seq_len'] = int(seq_len_str)
        except ValueError:
            self.config['seq_len'] = 128
            print(f"[WARN] Invalid seq_len, using 128")
        
        print(f"\n[OK] Training config:")
        print(f"     Learning rate: {self.config['learning_rate']}")
        print(f"     Steps: {self.config['total_steps']}")
        print(f"     Sequence length: {self.config['seq_len']}")
    
    def prompt_logging(self):
        """Configure logging and checkpoint names."""
        print("\n" + "="*73)
        print("STEP 4: LOGGING & CHECKPOINT")
        print("="*73)
        self.refresh_run_artifact_names()
        
        print(f"\n[OK] Auto-generated names from training setup:")
        print(f"     Format: training_<steps>steps_<lr>lr_ctx<context>_<timestamp>")
        print(f"     Log file: {self.config['log_name']}.log")
        print(f"     Checkpoint: {self.config['checkpoint_path']}")
    
    def prompt_generation_test(self):
        """Ask if user wants to test generation after training."""
        print("\n" + "="*73)
        print("STEP 5: POST-TRAINING OPTIONS")
        print("="*73)
        
        gen_choice = input("\nTest generation after training? [y/n] (default: y): ").strip().lower() or "y"
        self.config['test_generation'] = gen_choice == "y"
        
        if self.config['test_generation']:
            prompt = input("Generation prompt (default: 'the'): ").strip() or "the"
            self.config['gen_prompt'] = prompt
            max_tokens_str = input("Max tokens to generate (default: 50): ").strip() or "50"
            try:
                self.config['gen_max_tokens'] = int(max_tokens_str)
            except ValueError:
                self.config['gen_max_tokens'] = 50

            temp_str = input(f"Generation temperature (default: {GEN_DEFAULT_TEMPERATURE}): ").strip() or str(GEN_DEFAULT_TEMPERATURE)
            top_p_str = input(f"Generation top_p (default: {GEN_DEFAULT_TOP_P}): ").strip() or str(GEN_DEFAULT_TOP_P)
            rep_pen_str = input(f"Repetition penalty (default: {GEN_DEFAULT_REPETITION_PENALTY}): ").strip() or str(GEN_DEFAULT_REPETITION_PENALTY)
            try:
                self.config['gen_temperature'] = float(temp_str)
            except ValueError:
                self.config['gen_temperature'] = GEN_DEFAULT_TEMPERATURE
            try:
                self.config['gen_top_p'] = float(top_p_str)
            except ValueError:
                self.config['gen_top_p'] = GEN_DEFAULT_TOP_P
            try:
                self.config['gen_repetition_penalty'] = float(rep_pen_str)
            except ValueError:
                self.config['gen_repetition_penalty'] = GEN_DEFAULT_REPETITION_PENALTY
            
            print(f"\n[OK] Generation config:")
            print(f"     Prompt: '{self.config['gen_prompt']}'")
            print(f"     Max tokens: {self.config['gen_max_tokens']}")
            print(f"     Temperature: {self.config['gen_temperature']}")
            print(f"     Top-p: {self.config['gen_top_p']}")
            print(f"     Repetition penalty: {self.config['gen_repetition_penalty']}")

    def prompt_init_checkpoint(self):
        """Optionally initialize the next run from an existing checkpoint."""
        checkpoints_dir = Path("output/checkpoints")
        checkpoint_files = []
        if checkpoints_dir.exists():
            checkpoint_files = sorted(
                checkpoints_dir.glob("*.npz"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )

        print("\n" + "="*73)
        print("OPTIONAL CHECKPOINT INIT")
        print("="*73)
        print("Leave blank to start from random weights.")

        if checkpoint_files:
            print("\nRecent checkpoints:")
            for idx, checkpoint_file in enumerate(checkpoint_files[:5], 1):
                print(f"  {idx}) {checkpoint_file}")

        existing_init = self.config.get('init_checkpoint_path')
        if existing_init:
            print(f"\nCurrent init checkpoint: {existing_init}")

        while True:
            init_choice = input("Checkpoint path or recent index (blank = random init): ").strip()
            if not init_choice:
                self.config['init_checkpoint_path'] = None
                print("[OK] Using random initialization")
                return

            selected_path = None
            if checkpoint_files:
                try:
                    selected_index = int(init_choice) - 1
                    if 0 <= selected_index < min(len(checkpoint_files), 5):
                        selected_path = str(checkpoint_files[selected_index])
                except ValueError:
                    selected_path = None

            if selected_path is None:
                selected_path = init_choice

            try:
                normalized_path, checkpoint_config = self._resolve_init_checkpoint_config(selected_path)
            except Exception as exc:
                print(f"[ERR] {exc}")
                print("[INFO] Select another checkpoint or press Enter for random init.")
                continue

            print("\nCheckpoint config:")
            print(f"  ctx={checkpoint_config.max_len}, embed={checkpoint_config.embedding_dim}, heads={checkpoint_config.num_heads}, layers={checkpoint_config.num_layers}, attention={checkpoint_config.attention_impl}")

            mismatches = self._init_checkpoint_prompt_mismatches(checkpoint_config)
            if mismatches:
                print("\n[WARN] Checkpoint does not match the currently selected run config:")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")

                adopt_choice = input("Adopt checkpoint architecture for this run? [y/n] (default: y): ").strip().lower() or "y"
                if adopt_choice != 'y':
                    print("[INFO] Select another checkpoint or press Enter for random init.")
                    continue

                self._apply_init_checkpoint_model_config(checkpoint_config)
                self.refresh_run_artifact_names()
                print("\n[OK] Adopted checkpoint architecture:")
                print(f"     Model: {self.config['name'].upper()}")
                print(f"     Embedding: {self.config['embedding_dim']}D")
                print(f"     Heads: {self.config['num_heads']}")
                print(f"     Layers: {self.config['num_layers']}")
                print(f"     Seq length: {self.config['seq_len']}")
                print(f"     Attention: {self.config['attention_impl']}")
                print(f"     Checkpoint name: {self.config['checkpoint_path']}")

            self.config['init_checkpoint_path'] = normalized_path
            print(f"[OK] Init checkpoint: {self.config['init_checkpoint_path']}")
            return
    
    def print_summary(self):
        """Print summary of configuration without doing expensive preflight work."""
        print("\n" + "="*73)
        print("TRAINING CONFIGURATION SUMMARY")
        print("="*73)
        
        print(f"\nDataset:        {self.config['dataset']}")
        print(f"  Docs:         {len(self.config['corpus']):,}")
        if self.config.get('init_checkpoint_path'):
            print(f"  Init from:    {self.config['init_checkpoint_path']}")
        
        print(f"\nModel:          {self.config['name'].upper()}")
        if self.config.get('approx_params'):
            print(f"  Params:       ~{format_param_count(self.config['approx_params'])} (selector estimate)")
        print(f"  Embedding:    {self.config['embedding_dim']}D")
        print(f"  Heads:        {self.config['num_heads']}")
        print(f"  Layers:       {self.config['num_layers']}")
        print(f"  Attention:    {self.config['attention_impl']}")

        cost_card = build_preset_cost_card(
            self.config['embedding_dim'], self.config['num_heads'], self.config['num_layers']
        )
        print(f"  Est. VRAM:    ~{cost_card['total_vram_mb']:.0f}MB ({cost_card['vram_pct_of_budget']:.0f}% of {ESTIMATE_AVAILABLE_VRAM_MB}MB budget, @vocab~{ESTIMATE_VOCAB_SIZE})")
        print(f"  Est. speed:   ~{cost_card['tokens_per_sec']:.0f} tok/s (rough heuristic)")
        print(f"  Collapse risk: {cost_card['collapse_risk']}")
        
        print(f"\nTraining:")
        print(f"  LR:           {self.config['learning_rate']}")
        print(f"  Steps:        {self.config['total_steps']}")
        print(f"  Seq length:   {self.config['seq_len']}")
        
        print(f"\nCheckpoint:     {self.config['checkpoint_path']}")
        
        print(f"\nLogging:        {self.config['log_name']}.log")
        
        if self.config['test_generation']:
            print(f"\nGeneration:")
            print(f"  Prompt:       '{self.config['gen_prompt']}'")
            print(f"  Max tokens:   {self.config['gen_max_tokens']}")
            print(f"  Temperature:  {self.config.get('gen_temperature', GEN_DEFAULT_TEMPERATURE)}")
            print(f"  Top-p:        {self.config.get('gen_top_p', GEN_DEFAULT_TOP_P)}")
            print(f"  Rep penalty:  {self.config.get('gen_repetition_penalty', GEN_DEFAULT_REPETITION_PENALTY)}")

    def confirm_and_start(self):
        """Ask for final approval, then run expensive preflight and training."""
        start_now = input("\nStart training now? [y/n] (default: y): ").strip().lower() or "y"
        if start_now not in ("y", "yes"):
            logger.info("[OK] Training cancelled before launch")
            return False

        print("\nPreparing tokenizer and VRAM estimate...")

        # When an init checkpoint is present, resolve its vocab_size now if it was not
        # already stored by _apply_init_checkpoint_model_config (which is only called when
        # there are architecture mismatches).  Without this, forced_vocab_size stays None
        # and we would pass max_vocab_size=None to the tokenizer constructor, crashing it.
        init_checkpoint_path = self.config.get('init_checkpoint_path')
        forced_vocab_size = self.config.get('vocab_size')
        if init_checkpoint_path and forced_vocab_size is None:
            try:
                _, _cp_cfg = self._resolve_init_checkpoint_config(init_checkpoint_path)
                forced_vocab_size = _cp_cfg.vocab_size
                self.config['vocab_size'] = forced_vocab_size
            except Exception as _exc:
                print(f"[WARN] Could not read checkpoint vocab size ({_exc}); will derive from corpus")

        loaded_from_vocab_file = False
        if init_checkpoint_path and forced_vocab_size:
            # Load the static vocab JSON that was saved alongside the checkpoint so the
            # preflight tokenizer is byte-for-byte identical to what training used.
            import re as _re
            from pathlib import Path as _Path
            cp = _Path(init_checkpoint_path)
            stem = cp.name
            stem = _re.sub(r'\.step\d+\.p\d+\.npz$', '.npz', stem)
            stem = stem.replace('.best.npz', '.npz')
            vocab_json_name = stem.replace('.npz', '.json')
            if vocab_json_name.startswith('gpt_'):
                vocab_json_name = vocab_json_name.replace('gpt_', 'vocab_', 1)
            elif vocab_json_name.startswith('training_'):
                vocab_json_name = vocab_json_name.replace('training_', 'vocab_', 1)
            else:
                vocab_json_name = 'vocab_' + vocab_json_name
            vocab_json_path = cp.parent / vocab_json_name
            if vocab_json_path.exists():
                try:
                    static_tokenizer = CharacterGPTTokenizer.load_vocab(str(vocab_json_path))
                    if static_tokenizer.vocab_size == forced_vocab_size:
                        print(f"[OK] Loaded static vocab from checkpoint: {vocab_json_path.name}")
                        print(f"     Vocab size: {static_tokenizer.vocab_size}")
                        self.tokenizer = static_tokenizer
                        self.config['estimated_vocab_size'] = static_tokenizer.vocab_size
                        self.config['preflight_tokenizer_ready'] = True
                        self.config['preflight_tokenizer_docs'] = len(self.config['corpus'])
                        loaded_from_vocab_file = True
                    else:
                        print(f"[WARN] Static vocab size {static_tokenizer.vocab_size} != expected {forced_vocab_size}, falling back to corpus preflight")
                except Exception as exc:
                    print(f"[WARN] Could not load static vocab ({exc}), falling back to corpus preflight")
            else:
                print(f"[INFO] No static vocab JSON found at {vocab_json_path}, running corpus preflight")

        if not loaded_from_vocab_file:
            # Build tokenizer on the same train split used by train() so we can reuse it.
            preflight_train_corpus, _ = split_corpus_for_validation(self.config['corpus'])
            print(f"Tokenizer preflight docs (train split): {len(preflight_train_corpus):,}")

            tokenizer_kwargs = dict(
                source_docs=preflight_train_corpus,
                fallback_docs=preflight_train_corpus,
            )
            if forced_vocab_size is not None:
                tokenizer_kwargs['max_vocab_size'] = forced_vocab_size

            estimated_tokenizer, _, _ = build_shared_tokenizer(
                self.config.get('dataset', 'fineweb'),
                **tokenizer_kwargs,
            )
            self.tokenizer = estimated_tokenizer
            self.config['preflight_tokenizer_ready'] = True
            self.config['preflight_tokenizer_docs'] = len(preflight_train_corpus)
            self.config['estimated_vocab_size'] = self.tokenizer.vocab_size

        estimated_vocab_size = self.config['estimated_vocab_size']
        print(f"Estimated tokenizer vocab: {estimated_vocab_size}")

        print("\nStage 1 -> Stage 2 handoff rule:")
        print("  - Use avg loss <= 3.0 as the earliest handoff gate; high-2s are stronger")
        print("  - Require readable 25/50/75/100 milestone probes, not just a falling loss")
        print("  - Require story bad_samples = 0 before advancing to Stage 2")

        # VRAM validation
        print(f"\n" + "="*73)
        print("VRAM ESTIMATION")
        print("="*73)

        B = 1
        model_vram, training_vram, total_vram = estimate_vram_usage(
            vocab_size=estimated_vocab_size,
            embedding_dim=self.config['embedding_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            batch_size=B,
            seq_len=self.config['seq_len']
        )

        print(f"\nEstimated GPU Memory:")
        print(f"  Model weights:    ~{model_vram:.0f}MB")
        print(f"  Training overhead: ~{training_vram:.0f}MB")
        print(f"  Total needed:     ~{total_vram:.0f}MB")
        print(f"  GT730 v2 available: ~3500MB (4GB DDR3 total)")

        is_valid, _, warning_msg = validate_model_config(
            vocab_size=estimated_vocab_size,
            embedding_dim=self.config['embedding_dim'],
            num_heads=self.config['num_heads'],
            num_layers=self.config['num_layers'],
            batch_size=B,
            seq_len=self.config['seq_len'],
            available_vram_mb=3500
        )

        if warning_msg:
            print(f"\n[WARNING] {warning_msg}")
        else:
            print(f"\n[OK] Model should fit in VRAM")

        if not is_valid:
            retry = input("\nVRAM estimate is high. Continue anyway? [y/n] (default: n): ").strip().lower()
            if retry not in ("y", "yes"):
                logger.info("[OK] Training cancelled after VRAM estimate")
                return False

        print("\nStarting training...")
        self.train()
        self.test_generation()
        self.print_completion()
        return True
    
    def train(self):
        """Run training with configured parameters."""
        print("\n" + "="*73)
        print("STARTING TRAINING")
        print("="*73 + "\n")
        
        # Initialize logging with custom log name
        setup_logging(log_filename=self.config['log_name'])
        install_global_memory_pool()
        
        corpus = self.config['corpus']
        lr = self.config['learning_rate']
        total_steps = self.config['total_steps']
        T = self.config['seq_len']
        embedding_dim = self.config['embedding_dim']
        num_heads = self.config['num_heads']
        num_layers = self.config['num_layers']
        init_checkpoint_path = self.config.get('init_checkpoint_path')
        self._reset_goal_tracking()
        self._reset_stage_handoff_tracking()
        
        try:
            train_corpus, val_corpus = split_corpus_for_validation(corpus)
            self.config['corpus'] = train_corpus
            self.config['val_corpus'] = val_corpus

            # Build tokenizer once in preflight and reuse here when possible.
            if (
                self.tokenizer is not None
                and self.config.get('preflight_tokenizer_ready')
                and self.config.get('preflight_tokenizer_docs') == len(train_corpus)
            ):
                logger.info(
                    f"[OK] Reusing preflight tokenizer built from {self.config.get('preflight_tokenizer_docs'):,} training documents"
                )
            else:
                logger.info(f"Building tokenizer from {len(train_corpus):,} training documents...")
                self.tokenizer, tokenizer_docs, tokenizer_source = build_shared_tokenizer(
                    self.config.get('dataset', 'fineweb'),
                    source_docs=train_corpus,
                    fallback_docs=train_corpus,
                )
                logger.info(
                    f"[OK] Shared tokenizer vocab built from {len(tokenizer_docs):,} documents ({tokenizer_source})"
                )
            logger.info(f"[OK] Vocab size: {self.tokenizer.vocab_size}")
            
            from pathlib import Path
            cp_path = Path(self.config['checkpoint_path'])
            vocab_name = cp_path.name.replace('.npz', '.json')
            if vocab_name.startswith('training_'):
                vocab_name = vocab_name.replace('training_', 'vocab_', 1)
            elif vocab_name.startswith('gpt_'):
                vocab_name = vocab_name.replace('gpt_', 'vocab_', 1)
            else:
                vocab_name = "vocab_" + vocab_name
            self.tokenizer.save_vocab(str(cp_path.parent / vocab_name))

            actual_params = estimate_model_params(
                self.tokenizer.vocab_size,
                T,
                embedding_dim,
                num_layers,
            )
            logger.info(f"[OK] Estimated actual parameter count: {format_param_count(actual_params)}")
            
            # Create model config
            config = GPTConfig(
                vocab_size=self.tokenizer.vocab_size,
                max_len=T,
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                attention_impl=self.config.get('attention_impl', 'strided'),
                dropout_prob=0.0
            )
            logger.info(f"[OK] Model config created")
            
            # Instantiate model
            self.model = GPTModel(config)
            criterion = SoftmaxCrossEntropy()
            logger.info(f"[OK] Model instantiated")

            if init_checkpoint_path:
                validated_init_checkpoint = self._validate_init_checkpoint(init_checkpoint_path, config)
                logger.info(f"[INIT] Loading initialization checkpoint: {validated_init_checkpoint}")
                if not self.model.load_checkpoint(validated_init_checkpoint):
                    raise RuntimeError(f"Failed to load initialization checkpoint: {validated_init_checkpoint}")
                logger.info("[OK] Initialization checkpoint loaded")
            
            # Encode corpus
            logger.info(f"Encoding corpus...")
            raw_aligned_matrix = load_or_build_token_matrix(
                self.tokenizer,
                train_corpus,
                max_sequence_length=T + 1,
                cache_namespace="auto_train",
                dataset_name=self.config.get('dataset'),
            )
            logger.info(f"[OK] Encoded shape: {raw_aligned_matrix.shape}")

            val_aligned_matrix = load_or_build_token_matrix(
                self.tokenizer,
                val_corpus,
                max_sequence_length=T + 1,
                cache_namespace="auto_train_val",
                dataset_name=f"{self.config.get('dataset')}_val",
            )
            logger.info(f"[OK] Validation encoded shape: {val_aligned_matrix.shape}")
            
            B = 1
            batch_rng = np.random.default_rng()
            input_tokens, target_tokens, _ = sample_token_batch(raw_aligned_matrix, B, T, rng=batch_rng)
            val_batch_rng = np.random.default_rng(2975)
            val_batch_size = min(B, max(1, len(val_corpus)))
            val_input_tokens, val_target_tokens, _ = sample_token_batch(val_aligned_matrix, val_batch_size, T, rng=val_batch_rng)
            
            # Transfer to GPU
            logger.info(f"Transferring to GPU...")
            gpu_input = cuda.mem_alloc(input_tokens.nbytes)
            gpu_target = cuda.mem_alloc(target_tokens.nbytes)
            gpu_val_input = cuda.mem_alloc(val_input_tokens.nbytes)
            gpu_val_target = cuda.mem_alloc(val_target_tokens.nbytes)
            cuda.memcpy_htod(gpu_input, input_tokens.astype(np.int32))
            cuda.memcpy_htod(gpu_target, target_tokens.astype(np.int32))
            cuda.memcpy_htod(gpu_val_input, val_input_tokens.astype(np.int32))
            cuda.memcpy_htod(gpu_val_target, val_target_tokens.astype(np.int32))
            logger.info(f"[OK] GPU memory allocated")
            
            N = B * T
            V = config.vocab_size
            
            # Initialize metrics tracking (report every 1 step)
            metrics = TrainingMetrics(total_steps, log_interval=1, backend="cuda")
            metrics.start()
            
            scheduler = CosineWarmupScheduler(max_lr=lr, total_steps=total_steps, warmup_steps=min(200, total_steps // 10))
            
            grad_accum = self.config.get('grad_accum', 1)
            batch_tokens = B * T * grad_accum
            
            # Training loop
            logger.info(f"Starting {total_steps} training steps with real-time metrics...")
            logger.info("="*73)
            probe_milestones_summary, probe_milestones = self._format_probe_milestone_summary(total_steps)
            logger.info(f"Probe milestones resolved at startup: {probe_milestones_summary}")
            logger.info(f"Validation batch resample interval: every {VAL_RESAMPLE_INTERVAL_STEPS} step(s)")
            
            # Track loss changes to detect learning plateau
            recent_losses = []
            plateau_threshold = 0.001  # If loss changes < 0.1%, warn about low LR
            plateau_count = 0  # How many times we've detected plateau
            
            for step in range(1, total_steps + 1):
                metrics.step_start()

                self.model.zero_grad()
                step_loss_value = 0.0

                for micro_step in range(grad_accum):
                    input_tokens, target_tokens, _ = sample_token_batch(raw_aligned_matrix, B, T, rng=batch_rng)
                    cuda.memcpy_htod(gpu_input, input_tokens)
                    cuda.memcpy_htod(gpu_target, target_tokens)
                    
                    # Forward
                    gpu_logits = self.model.forward(gpu_input, B, T)
                    
                    # Loss & backward
                    micro_loss, gpu_dLogits = criterion(gpu_logits, gpu_target, N, V)
                    gpu_logits.free()
                    
                    # CRITICAL: Detect NaN/Inf loss and stop training
                    if not np.isfinite(micro_loss):
                        logger.error(f"[CRITICAL] NaN/Inf loss at step {step}, micro_step {micro_step}: {micro_loss}")
                        logger.error(f"[SOLUTION] Reduce learning rate (current: {lr}) or use a smaller model")
                        logger.error(f"[DEBUG] Per-param gradient clipping was disabled")
                        raise RuntimeError(f"Training diverged: NaN loss at step {step}")
                    
                    step_loss_value += micro_loss / grad_accum
                    
                    self.model.backward(gpu_dLogits, B, T, scale=1.0/grad_accum, accumulate=True)
                    gpu_dLogits.free()

                    self.model.free_forward_caches()
                
                loss_val = step_loss_value
                
                # Track loss for plateau detection every 10 steps
                recent_losses.append(loss_val)
                if len(recent_losses) > 10:
                    recent_losses.pop(0)
                    if step > 50:  # After warmup
                        loss_change = abs(recent_losses[-1] - recent_losses[0]) / (recent_losses[0] + 1e-10)
                        if loss_change < plateau_threshold:
                            plateau_count += 1
                        else:
                            plateau_count = 0  # Reset if loss changes

                grad_norm = None
                if metrics.should_log_step(step):
                    grad_norm = self.model.compute_grad_norm()
                
                # Optimize (gradient clipping disabled - testing gradient flow)
                current_lr = scheduler.get_lr(step)
                self.model.update_weights(lr=current_lr, step=step)
                pool_used_mb, pool_total_mb = get_memory_pool_stats_mb()

                val_loss = None
                if metrics.should_log_step(step):
                    if (step - 1) % VAL_RESAMPLE_INTERVAL_STEPS == 0:
                        val_input_tokens, val_target_tokens, _ = sample_token_batch(
                            val_aligned_matrix,
                            val_batch_size,
                            T,
                            rng=val_batch_rng,
                        )
                        cuda.memcpy_htod(gpu_val_input, val_input_tokens.astype(np.int32))
                        cuda.memcpy_htod(gpu_val_target, val_target_tokens.astype(np.int32))

                    val_logits = self.model.forward(gpu_val_input, val_batch_size, T)
                    val_loss, gpu_val_dLogits = criterion(val_logits, gpu_val_target, val_batch_size * T, V)
                    val_logits.free()
                    gpu_val_dLogits.free()
                    self.model.free_forward_caches()

                self._maybe_save_goal_checkpoint(step, loss_val, val_loss=val_loss)
                
                # Record step metrics
                metrics.step_end(
                    loss_val,
                    lr=current_lr,
                    grad_norm=grad_norm,
                    batch_tokens=batch_tokens,
                    pool_used_mb=pool_used_mb,
                    pool_total_mb=pool_total_mb,
                    val_loss=val_loss,
                )
                
                if step % 1000 == 0:
                    try:
                        subprocess.Popen([sys.executable, "training_log_plotter.py", "--save", "output/training_metrics_latest.png", "--no-forecast"])
                        subprocess.Popen([sys.executable, "loss_landscape_plotter.py"])
                    except Exception:
                        pass

                current_step = step
                if current_step in probe_milestones:
                    checkpoint_root, checkpoint_ext = os.path.splitext(self.config['checkpoint_path'])
                    progress_pct = int(round((current_step / total_steps) * 100))
                    probe_checkpoint_path = (
                        f"{checkpoint_root}.step{current_step}.p{progress_pct}{checkpoint_ext}"
                    )
                    probe_results = self._save_checkpoint_with_probes(
                        probe_checkpoint_path,
                        f"PROBE@{progress_pct}%_step{current_step}",
                    )
                    self._record_probe_milestone(progress_pct, current_step, probe_results)
            
            logger.info("="*73)
            logger.info(f"[OK] Training complete!")
            
            # Finalize metrics
            metrics.finalize()
            try:
                subprocess.Popen([sys.executable, "training_log_plotter.py", "--save", "output/training_metrics_latest.png", "--no-forecast"])
                subprocess.Popen([sys.executable, "loss_landscape_plotter.py"])
            except Exception:
                pass
            if metrics.losses:
                self.config['stage1_avg_loss'] = float(np.mean(metrics.losses))
            handoff_report = self._build_stage2_handoff_report()
            self.config['stage2_handoff_report'] = handoff_report
            self.config['stage2_handoff_ready'] = handoff_report['ready']
            self._log_goal_summary()
            self._log_stage2_handoff_summary(handoff_report)
            
            # Save checkpoint
            logger.info(f"Saving checkpoint to {self.config['checkpoint_path']}...")
            os.makedirs(os.path.dirname(self.config['checkpoint_path']), exist_ok=True)
            self.model.save_checkpoint(self.config['checkpoint_path'])
            logger.info(f"[OK] Checkpoint saved")

            # Persist checkpoint metadata before any post-train generation probe.
            self.save_config_to_json()
            
            # Cleanup
            logger.info(f"Cleaning up GPU memory...")
            gpu_input.free()
            gpu_target.free()
            gpu_val_input.free()
            gpu_val_target.free()
            self.model.embedding.wte.free()
            self.model.embedding.wpe.free()
            for block in self.model.blocks:
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
            self.model.ln_f_gamma.free()
            self.model.ln_f_beta.free()
            self.model.lm_head_w.free()
            free_held_pool_blocks()
            logger.info(f"[OK] GPU memory freed")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def test_generation(self):
        """Test generation with trained model."""
        if not self.config['test_generation']:
            return
        
        print("\n" + "="*73)
        print("TESTING GENERATION")
        print("="*73 + "\n")
        
        try:
            from generate import generate_from_seed, run_generation_probes, format_generation_probes
            
            prompt = self.config['gen_prompt']
            max_tokens = self.config['gen_max_tokens']
            checkpoint = self._preferred_checkpoint_path()
            
            logger.info(f"Generating from prompt: '{prompt}'")
            logger.info(f"Max tokens: {max_tokens}")
            logger.info(f"Checkpoint: {checkpoint}")
            
            if not os.path.exists(checkpoint):
                logger.warning(f"[WARN] Checkpoint not found: {checkpoint}")
                logger.info(f"[INFO] To generate text, train first and then run:")
                logger.info(f"       python generate.py")
                return
            
            # Call generation function
            generated_text = generate_from_seed(
                checkpoint,
                prompt,
                max_tokens,
                temperature=self.config.get('gen_temperature', GEN_DEFAULT_TEMPERATURE),
                top_p=self.config.get('gen_top_p', GEN_DEFAULT_TOP_P),
                repetition_penalty=self.config.get('gen_repetition_penalty', GEN_DEFAULT_REPETITION_PENALTY),
                num_heads=self.config.get('num_heads'),
                source_docs=self.config.get('corpus'),
            )
            
            if generated_text:
                logger.info(f"\n[OK] Generated text:\n{generated_text}")
                print(f"\n✨ Generated: {generated_text[:100]}..." if len(generated_text) > 100 else f"\n✨ Generated: {generated_text}")
            else:
                logger.warning("[WARN] Generation returned None")

            memorization_prefix = None
            if self.config.get('corpus'):
                memorization_prefix = self.config['corpus'][0][:32]

            probe_results = run_generation_probes(
                checkpoint,
                prompt=prompt,
                memorization_prefix=memorization_prefix,
                top_k=8,
                max_new_tokens=max_tokens,
                temperature=0.8,
                sampled_temperature=self.config.get('gen_temperature', GEN_DEFAULT_TEMPERATURE),
                sampled_top_p=self.config.get('gen_top_p', GEN_DEFAULT_TOP_P),
                sampled_repetition_penalty=self.config.get('gen_repetition_penalty', GEN_DEFAULT_REPETITION_PENALTY),
                num_heads=self.config.get('num_heads'),
                source_docs=self.config.get('corpus'),
            )
            logger.info("\nGeneration probes:\n" + format_generation_probes(probe_results))
            
        except ImportError:
            logger.warning("[WARN] generate module not found - skipping generation test")
        except Exception as e:
            logger.warning(f"Generation test failed: {e}")
    
    def run(self):
        """Run full interactive training pipeline."""
        print("\n" + "="*73)
        print("INTERACTIVE GPT TRAINING LAUNCHER")
        print("="*73)
        
        # Check if previous run config exists
        if os.path.exists(self.CONFIG_FILE):
            print("\n[INFO] Found previous run configuration")
            reuse = input("Reuse last run's settings? [y/n] (default: n): ").strip().lower()
            if reuse == "y":
                if self.load_config_from_json():
                    print(f"[OK] Loaded previous config:")
                    print(f"     Model: {self.config.get('name', 'N/A').upper()}")
                    print(f"     Dataset: {self.config.get('dataset', 'N/A')}")
                    print(f"     Docs: {self.config.get('corpus_size', 'N/A')}")
                    print(f"     Steps: {self.config.get('total_steps', 'N/A')}")
                    print(f"     LR: {self.config.get('learning_rate', 'N/A')}")
                    
                    mod_choice = "5"
                    modify = input("\nModify any settings? [y/n] (default: n): ").strip().lower()
                    if modify == "y":
                        # Allow selective modification
                        print("\nWhich settings to modify?")
                        print("  1) Steps")
                        print("  2) Learning rate")
                        print("  3) Prompt (for generation)")
                        print("  4) Log/Checkpoint names")
                        print("  5) Start training with loaded config")
                        
                        mod_choice = input("Select [1-5]: ").strip()
                        
                        if mod_choice == "1":
                            steps_str = input(f"Training steps (current: {self.config.get('total_steps')}): ").strip()
                            if steps_str:
                                try:
                                    self.config['total_steps'] = int(steps_str)
                                except ValueError:
                                    pass
                        elif mod_choice == "2":
                            lr_str = input(f"Learning rate (current: {self.config.get('learning_rate')}): ").strip()
                            if lr_str:
                                try:
                                    self.config['learning_rate'] = float(lr_str)
                                except ValueError:
                                    pass
                        elif mod_choice == "3":
                            if self.config.get('test_generation'):
                                prompt = input(f"Generation prompt (current: '{self.config.get('gen_prompt')}'): ").strip()
                                if prompt:
                                    self.config['gen_prompt'] = prompt
                        elif mod_choice == "4":
                            log_name = input(f"Log file name (current: {self.config.get('log_name')}): ").strip()
                            if log_name:
                                self.config['log_name'] = log_name
                            checkpoint_name = input(f"Checkpoint name (current: {self.config.get('checkpoint_name')}): ").strip()
                            if checkpoint_name:
                                self.config['checkpoint_name'] = checkpoint_name
                                self.config['checkpoint_path'] = f"output/checkpoints/{checkpoint_name}.npz"

                    if mod_choice != "4":
                        self.refresh_run_artifact_names()
                        print("\n[INFO] Regenerated log/checkpoint names for this run:")
                        print(f"       Log file: {self.config['log_name']}.log")
                        print(f"       Checkpoint: {self.config['checkpoint_path']}")

                    self.prompt_init_checkpoint()
                    
                    # Skip all prompts and go to training
                    self.print_summary()
                    if not self.confirm_and_start():
                        print("\n[INFO] Training cancelled by user")
                        sys.exit(0)
                    return
        
        # Normal interactive flow
        self.prompt_dataset()
        self.prompt_model_config()
        self.prompt_training_params()
        self.prompt_logging()
        self.prompt_init_checkpoint()
        self.prompt_generation_test()
        
        self.print_summary()
        if not self.confirm_and_start():
            print("\n[INFO] Training cancelled by user")
            sys.exit(0)
    
    def print_completion(self):
        """Print training completion message and save config."""
        if self.config.get('story_bad_samples') is None:
            bad_samples_str = input("\nStory bad_samples from latest audit (blank = unknown): ").strip()
            if bad_samples_str:
                try:
                    parsed_bad_samples = int(bad_samples_str)
                    if parsed_bad_samples < 0:
                        raise ValueError("bad_samples must be >= 0")
                    self.config['story_bad_samples'] = parsed_bad_samples
                except ValueError:
                    logger.warning(f"[WARN] Invalid bad_samples value: {bad_samples_str!r}; keeping unknown")

        handoff_report = self._build_stage2_handoff_report()
        self.config['stage2_handoff_report'] = handoff_report
        self.config['stage2_handoff_ready'] = handoff_report['ready']

        print("\n" + "="*73)
        print("TRAINING COMPLETE!")
        print("="*73)
        if self.config.get('init_checkpoint_path'):
            print(f"\nInitialized from:   {self.config['init_checkpoint_path']}")
        print(f"\nCheckpoint saved to: {self.config['checkpoint_path']}")
        if self.config.get('best_checkpoint_path'):
            print(f"Best checkpoint:    {self.config['best_checkpoint_path']}")
            print(f"Recommended:        {self._preferred_checkpoint_path()}")

        print(f"\nStage 1 -> Stage 2 handoff:")
        avg_loss = handoff_report.get('avg_loss')
        avg_loss_text = f"{avg_loss:.4f}" if avg_loss is not None else "n/a"
        print(f"  Stage 1 avg loss:           {avg_loss_text} (target <= {handoff_report['earliest_target']:.2f})")
        print(f"  Probe markers present:      {'yes' if handoff_report['have_all_probe_markers'] else 'no'}")
        print(f"  Probe outputs readable:     {'yes' if handoff_report['probes_readable'] else 'no'}")
        print(f"  Probe trend improving:      {'yes' if handoff_report['probe_trend_ok'] else 'no'}")
        story_bad_samples = handoff_report.get('story_bad_samples')
        story_bad_samples_text = "unknown" if story_bad_samples is None else str(story_bad_samples)
        print(f"  story bad_samples:          {story_bad_samples_text} (must be 0)")
        print(f"  Ready for Stage 2:          {'YES' if handoff_report['ready'] else 'NO'}")

        print(f"\nNext steps:")
        print(f"  - Test generation: python generate.py --checkpoint {self._preferred_checkpoint_path()}")
        if self.config.get('best_checkpoint_path'):
            print(f"  - Test final checkpoint: python generate.py --checkpoint {self.config['checkpoint_path']}")
        print(f"  - View logs: tail -f output/logs/training_*.log")
        print(f"  - Reuse these settings: Run auto_train.py again and select 'y' to reuse config")
        
        # Save config for next run
        self.save_config_to_json()


if __name__ == "__main__":
    trainer = InteractiveTrainer()
    trainer.run()
