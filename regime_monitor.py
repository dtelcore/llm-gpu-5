"""
BIS / TTR / RCI / Phi telemetry and the bounded Regime Controller.

This module is deliberately pure Python + numpy (no CUDA, no pycuda import) so it
can be unit tested standalone and reasoned about independently of the GPU training
stack. It is consumed by auto_train.py's InteractiveTrainer.train() loop.

Definitions
-----------
BIS (Boundary Integrity Score): does the model understand "wordness"? Computed by
    detecting glued-word artifacts -- short, common function words directly fused
    onto a preceding letter with no whitespace boundary (e.g. "andthe", "tothe",
    "Heher"-style adjacency fusion) -- the exact failure mode diagnosed earlier in
    this project at vocab=4096 with too-small an embedding_dim.

TTR (Type-Token Ratio): vocabulary richness/variance, taken directly from
    token_repetition_stats()['token_unique_ratio'].

RCI (Repetition Collapse Index): tendency to fall into "attractor basins" (loops
    like "The The The"), derived from token_repetition_stats()'s dominant_ratio and
    immediate_repeat_ratio.

Phi (Regime Transition Function): Phi = (BIS * TTR) / RCI, classified into one of
    four regimes (attractor_collapse / syntactic_drift / semantic_emergence /
    stable_generator) by fixed thresholds.

Architecture constraint (important): embedding_dim/num_heads/num_layers are frozen
    once GPTModel(config) is constructed -- there is no way to resize them mid-run
    without rebuilding the model and losing optimizer state. The RegimeController
    therefore NEVER mutates architecture live; "EMBEDDING_EXPANSION_RECOMMENDED" is
    always a logged recommendation for the *next* run, never a live action.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# NOTE: generate.py transitively imports pycuda.autoinit (requires a CUDA device),
# so it is intentionally NOT imported at module scope here -- only lazily inside
# lightweight_greedy_probe(). Everything else in this module (BIS/TTR/RCI/Phi math,
# RegimeTracker, RegimeController) stays pure Python/numpy and CUDA-free, so it can
# be unit tested without a GPU (see test_regime_monitor.py).


# ============================================================================
# Tokenization / repetition statistics (shared with auto_train.py)
# ============================================================================

_WORD_OR_PUNCT_RE = re.compile(r"\w+|[^\w\s]")

# Common short English function words / pronouns that, when found concatenated
# back-to-back with no whitespace between them (e.g. "and"+"the" -> "andthe",
# "to"+"the" -> "tothe", "he"+"her" -> "Heher"), are a strong signal of
# word-boundary collapse. A *single* occurrence of any of these is a perfectly
# normal word (e.g. "the", "was"), so the detector below only flags spans made
# of TWO OR MORE of them tiled together with no leftover characters -- it does
# NOT match substrings inside unrelated real words (no lookbehind/partial
# matching), avoiding false positives like "cat" (contains "at") or "was"
# (contains "as").
_GLUED_FUNCTION_WORDS = (
    "the", "and", "to", "of", "in", "on", "is", "it", "for", "with",
    "that", "this", "was", "but", "not", "as", "at", "be", "by",
    "he", "she", "they", "her", "him", "his", "we", "you",
)
_GLUED_FUNCTION_WORDS_SORTED = sorted(_GLUED_FUNCTION_WORDS, key=len, reverse=True)
_GLUED_WORD_RE = re.compile(
    r"\b(?:" + "|".join(_GLUED_FUNCTION_WORDS_SORTED) + r"){2,}\b",
    re.IGNORECASE,
)

RCI_EPSILON = 0.05  # floor for RCI's denominator role in Phi to avoid blow-up


def tokenize_words(text):
    """Lowercased word/punctuation tokens, matching the regex used elsewhere in
    this project's readability heuristics (auto_train.py's probe scoring)."""
    return [tok for tok in _WORD_OR_PUNCT_RE.findall(text.lower()) if tok.strip()]


def token_repetition_stats(text):
    """Return repetition stats for tokenized text.

    Returns a dict with:
        dominant_ratio: share of tokens taken by the single most common token
        immediate_repeat_ratio: fraction of adjacent token pairs that repeat
        token_unique_ratio: unique tokens / total tokens (this module's TTR)
    """
    tokens = tokenize_words(text)
    token_count = len(tokens)
    if token_count < 2:
        return {
            "dominant_ratio": 0.0,
            "immediate_repeat_ratio": 0.0,
            "token_unique_ratio": 1.0 if token_count == 1 else 0.0,
        }

    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    dominant_ratio = max(counts.values()) / token_count
    immediate_repeats = sum(1 for idx in range(1, token_count) if tokens[idx] == tokens[idx - 1])
    immediate_repeat_ratio = immediate_repeats / max(1, token_count - 1)
    token_unique_ratio = len(counts) / token_count
    return {
        "dominant_ratio": dominant_ratio,
        "immediate_repeat_ratio": immediate_repeat_ratio,
        "token_unique_ratio": token_unique_ratio,
    }


# ============================================================================
# BIS / RCI / Phi
# ============================================================================

def compute_boundary_integrity_score(text):
    """BIS in [0, 1]: 1.0 means no detected word-boundary fusion artifacts.

    glued_count counts regex matches of common short function words immediately
    preceded by a letter with no whitespace boundary (e.g. the "the" in "andthe").
    """
    words = tokenize_words(text)
    word_count = sum(1 for tok in words if tok.isalpha())
    if word_count == 0:
        return 1.0
    glued_count = len(_GLUED_WORD_RE.findall(text))
    return float(max(0.0, min(1.0, 1.0 - glued_count / max(1, word_count))))


def compute_repetition_collapse_index(token_stats):
    """RCI in [0, 1]: higher means stronger tendency toward attractor-basin
    repetition loops (e.g. "The The The")."""
    rci = 0.5 * token_stats["dominant_ratio"] + 0.5 * token_stats["immediate_repeat_ratio"]
    return float(max(0.0, min(1.0, rci)))


def compute_phase_transition_score(bis, ttr, rci):
    """Phi = (BIS * TTR) / max(RCI, RCI_EPSILON)."""
    return float((bis * ttr) / max(rci, RCI_EPSILON))


def classify_regime(phi):
    """Classify a Phi score into one of the four regimes."""
    if phi < 0.2:
        return "attractor_collapse"
    if phi < 0.6:
        return "syntactic_drift"
    if phi < 1.2:
        return "semantic_emergence"
    return "stable_generator"


_ADVANCED_REGIMES = {"semantic_emergence", "stable_generator"}
_COLLAPSED_REGIMES = {"attractor_collapse", "syntactic_drift"}


@dataclass
class RegimeSample:
    step: int
    bis: float
    ttr: float
    rci: float
    phi: float
    regime: str


# ============================================================================
# Lightweight in-process generation probe (no checkpoint disk round-trip)
# ============================================================================

def lightweight_greedy_probe(model, tokenizer, prompt, max_new_tokens=40):
    """Greedy-decode a short continuation directly from the live in-memory
    training model/tokenizer. Reuses generate.py's plain encode/decode/sampling
    helpers but skips GenerationSession's checkpoint load, so it's cheap enough
    to run every ~100 training steps.
    """
    import pycuda.driver as cuda
    from generate import encode_token_ids, decode_token_ids, sample_next_token

    current_tokens = encode_token_ids(tokenizer, prompt)
    max_context = getattr(model.config, "max_len", None) or 128
    vocab_size = getattr(model.config, "vocab_size")

    for _ in range(max_new_tokens):
        context_window = current_tokens[-max_context:]
        current_t_length = len(context_window)
        host_input = np.array([context_window], dtype=np.int32)
        gpu_input = cuda.mem_alloc(host_input.nbytes)
        try:
            cuda.memcpy_htod(gpu_input, host_input)
            gpu_logits = model.forward(gpu_input, B=1, T=current_t_length)
            host_last_logits = np.empty(vocab_size, dtype=np.float32)
            last_row_offset = (current_t_length - 1) * vocab_size * 4
            cuda.memcpy_dtoh(host_last_logits, int(gpu_logits) + last_row_offset)
            gpu_logits.free()
            model.free_forward_caches()
        finally:
            gpu_input.free()

        next_token_id = sample_next_token(
            host_last_logits,
            temperature=0.0,
            recent_token_ids=current_tokens[-32:],
            bos_token_id=getattr(tokenizer, "BOS_ID", None),
        )
        current_tokens.append(next_token_id)

    return decode_token_ids(tokenizer, current_tokens)


# ============================================================================
# RegimeTracker: EMA(Phi) + trend
# ============================================================================

class RegimeTracker:
    """Tracks Phi history and derives an EMA + rising/falling/flat trend."""

    def __init__(self, alpha=0.3, flat_tolerance=0.02):
        self.alpha = alpha
        self.flat_tolerance = flat_tolerance
        self.history = []  # list[RegimeSample]
        self.ema_phi = None
        self.previous_ema_phi = None

    def update(self, sample: RegimeSample):
        self.history.append(sample)
        self.previous_ema_phi = self.ema_phi
        if self.ema_phi is None:
            self.ema_phi = sample.phi
        else:
            self.ema_phi = (1.0 - self.alpha) * self.ema_phi + self.alpha * sample.phi

        trend = "flat"
        if self.previous_ema_phi is not None:
            delta = self.ema_phi - self.previous_ema_phi
            if delta > self.flat_tolerance:
                trend = "rising"
            elif delta < -self.flat_tolerance:
                trend = "falling"
        return self.ema_phi, trend

    @property
    def previous_regime(self):
        if len(self.history) < 2:
            return None
        return self.history[-2].regime


# ============================================================================
# RegimeController: bounded, cooldown-gated decision logic
# ============================================================================

@dataclass
class RegimeAction:
    action_type: str
    reason: str
    payload: dict = field(default_factory=dict)


LABEL_SMOOTHING_STEP = 0.02
LABEL_SMOOTHING_MAX = 0.3
LR_MULTIPLIER_DECAY = 0.9
LR_MULTIPLIER_MIN = 0.5
DEFAULT_COOLDOWN_STEPS = 300


class RegimeController:
    """Stateful decision logic. Call decide() once per probe; apply the returned
    actions' bounded math externally (this class only proposes/clamps values, the
    caller -- auto_train.py -- is responsible for actually mutating its config).

    Safety properties:
      - EMBEDDING_EXPANSION_RECOMMENDED never mutates anything live; it only ever
        proposes a value for the caller to persist as a *next-run* hint.
      - INCREASE_LABEL_SMOOTHING is clamped to [0.0, LABEL_SMOOTHING_MAX].
      - DECAY_LEARNING_RATE only ever decreases lr_multiplier (floored at
        LR_MULTIPLIER_MIN); nothing in this controller increases it back.
      - Every action type has an independent cooldown so repeated triggers can't
        oscillate or spam checkpoints.
    """

    def __init__(self, min_step=50, cooldown_steps=DEFAULT_COOLDOWN_STEPS):
        self.min_step = min_step
        self.cooldown_steps = cooldown_steps
        self._last_triggered_step = {}

    def _cooldown_ok(self, action_type, step):
        last_step = self._last_triggered_step.get(action_type)
        return last_step is None or (step - last_step) >= self.cooldown_steps

    def _mark_triggered(self, action_type, step):
        self._last_triggered_step[action_type] = step

    def decide(self, sample: RegimeSample, ema_phi, trend, current_label_smoothing,
               current_lr_multiplier, previous_regime):
        """Return a list of RegimeAction to apply (possibly empty)."""
        if sample.step < self.min_step:
            return []

        actions = []

        # 1. Fragmented regime: low boundary integrity -> next-run recommendation only.
        if sample.bis < 0.5 and self._cooldown_ok("EMBEDDING_EXPANSION_RECOMMENDED", sample.step):
            actions.append(RegimeAction(
                action_type="EMBEDDING_EXPANSION_RECOMMENDED",
                reason=f"BIS={sample.bis:.2f} < 0.5 at step {sample.step}: boundary collapse detected",
                payload={},
            ))
            self._mark_triggered("EMBEDDING_EXPANSION_RECOMMENDED", sample.step)

        # 2. Syntactic attractor: high repetition collapse + low vocabulary richness.
        elif (sample.rci > 0.6 and sample.ttr < 0.4
              and current_label_smoothing < LABEL_SMOOTHING_MAX
              and self._cooldown_ok("INCREASE_LABEL_SMOOTHING", sample.step)):
            new_value = min(LABEL_SMOOTHING_MAX, current_label_smoothing + LABEL_SMOOTHING_STEP)
            actions.append(RegimeAction(
                action_type="INCREASE_LABEL_SMOOTHING",
                reason=f"RCI={sample.rci:.2f} > 0.6 and TTR={sample.ttr:.2f} < 0.4: syntactic attractor risk",
                payload={"new_label_smoothing": new_value},
            ))
            self._mark_triggered("INCREASE_LABEL_SMOOTHING", sample.step)

        # 3. Semantic transition: lock in gains by decaying LR a bit faster.
        elif (sample.bis > 0.75 and trend == "rising"
              and current_lr_multiplier > LR_MULTIPLIER_MIN
              and self._cooldown_ok("DECAY_LEARNING_RATE", sample.step)):
            new_value = max(LR_MULTIPLIER_MIN, current_lr_multiplier * LR_MULTIPLIER_DECAY)
            actions.append(RegimeAction(
                action_type="DECAY_LEARNING_RATE",
                reason=f"BIS={sample.bis:.2f} > 0.75 and Phi trend rising: locking in gains",
                payload={"new_lr_multiplier": new_value},
            ))
            self._mark_triggered("DECAY_LEARNING_RATE", sample.step)

        # 4. Pivot point: regime crossed from collapsed into advanced.
        if (previous_regime in _COLLAPSED_REGIMES and sample.regime in _ADVANCED_REGIMES
                and self._cooldown_ok("SAVE_AWAKENING_CHECKPOINT", sample.step)):
            actions.append(RegimeAction(
                action_type="SAVE_AWAKENING_CHECKPOINT",
                reason=f"Regime crossed {previous_regime} -> {sample.regime} at step {sample.step}",
                payload={},
            ))
            self._mark_triggered("SAVE_AWAKENING_CHECKPOINT", sample.step)

        return actions
