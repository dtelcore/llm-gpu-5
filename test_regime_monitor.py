"""
Pure-Python (no CUDA) tests for regime_monitor.py: BIS/TTR/RCI/Phi math, regime
classification, and the RegimeController's cooldown / no-live-architecture-mutation
safety guarantees.

Run with: python test_regime_monitor.py
(Does not require a GPU -- regime_monitor.py only imports pycuda lazily, inside
lightweight_greedy_probe(), which is not exercised here.)
"""

import regime_monitor as rm


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True
    print(f"[FAIL] {message}")
    return False


def test_bis_clean_text():
    text = ("Once upon a time there was a cat. The cat sat on the mat quietly "
            "and watched the birds outside the window.")
    bis = rm.compute_boundary_integrity_score(text)
    return check(bis == 1.0, f"BIS on clean text == 1.0 (got {bis:.3f})")


def test_bis_glued_text():
    text = "andthe tothe andthe heher tothe andthe weshe theyhe"
    bis = rm.compute_boundary_integrity_score(text)
    return check(bis < 0.5, f"BIS on heavily glued text < 0.5 (got {bis:.3f})")


def test_bis_no_false_positives_on_normal_words():
    # "was" contains "as", "upon" contains "on", "cat" contains "at" -- none of
    # these should be flagged as glued artifacts.
    text = "upon a time was a cat"
    matches = rm._GLUED_WORD_RE.findall(text)
    return check(matches == [], f"No false-positive glue matches on normal words (got {matches})")


def test_phi_threshold_classification():
    results = [
        (0.1, "attractor_collapse"),
        (0.19, "attractor_collapse"),
        (0.2, "syntactic_drift"),
        (0.59, "syntactic_drift"),
        (0.6, "semantic_emergence"),
        (1.19, "semantic_emergence"),
        (1.2, "stable_generator"),
        (5.0, "stable_generator"),
    ]
    all_passed = True
    for phi, expected in results:
        actual = rm.classify_regime(phi)
        all_passed = check(actual == expected, f"classify_regime({phi}) == {expected!r} (got {actual!r})") and all_passed
    return all_passed


def test_rci_repetition_collapse():
    stats = rm.token_repetition_stats("the the the the the the")
    rci = rm.compute_repetition_collapse_index(stats)
    return check(rci > 0.9, f"RCI on pure repetition loop is near 1.0 (got {rci:.3f})")


def test_phi_formula():
    phi = rm.compute_phase_transition_score(bis=0.8, ttr=0.6, rci=0.4)
    expected = (0.8 * 0.6) / 0.4
    return check(abs(phi - expected) < 1e-9, f"Phi formula matches (BIS*TTR)/RCI (got {phi}, expected {expected})")


def test_controller_cooldown_blocks_repeat_trigger():
    """Same trigger condition held constant across two probes within the cooldown
    window must only fire once."""
    controller = rm.RegimeController(min_step=10, cooldown_steps=200)
    tracker = rm.RegimeTracker()

    # Construct a sample that ONLY satisfies the label-smoothing trigger
    # (rci > 0.6 and ttr < 0.4), keeping bis high enough to avoid the
    # higher-priority embedding-expansion branch.
    def make_sample(step):
        bis, ttr, rci = 0.9, 0.2, 0.7
        phi = rm.compute_phase_transition_score(bis, ttr, rci)
        regime = rm.classify_regime(phi)
        return rm.RegimeSample(step=step, bis=bis, ttr=ttr, rci=rci, phi=phi, regime=regime)

    sample1 = make_sample(20)
    ema1, trend1 = tracker.update(sample1)
    actions1 = controller.decide(sample1, ema1, trend1, current_label_smoothing=0.1,
                                  current_lr_multiplier=1.0, previous_regime=tracker.previous_regime)

    sample2 = make_sample(40)  # 20 steps later, well within the 200-step cooldown
    ema2, trend2 = tracker.update(sample2)
    actions2 = controller.decide(sample2, ema2, trend2, current_label_smoothing=0.1,
                                  current_lr_multiplier=1.0, previous_regime=tracker.previous_regime)

    first_fired = any(a.action_type == "INCREASE_LABEL_SMOOTHING" for a in actions1)
    second_blocked = not any(a.action_type == "INCREASE_LABEL_SMOOTHING" for a in actions2)
    return check(first_fired and second_blocked,
                 f"INCREASE_LABEL_SMOOTHING fires once then is cooldown-blocked "
                 f"(actions1={[a.action_type for a in actions1]}, actions2={[a.action_type for a in actions2]})")


def test_controller_never_mutates_architecture_live():
    """EMBEDDING_EXPANSION_RECOMMENDED must be the only architecture-related
    action type, and its payload must never carry a value the caller could
    apply live (auto_train.py only ever uses it to set a *next-run* hint)."""
    controller = rm.RegimeController(min_step=10, cooldown_steps=200)
    tracker = rm.RegimeTracker()

    bis, ttr, rci = 0.2, 0.5, 0.3  # low BIS -> should trigger the recommendation path
    phi = rm.compute_phase_transition_score(bis, ttr, rci)
    sample = rm.RegimeSample(step=20, bis=bis, ttr=ttr, rci=rci, phi=phi, regime=rm.classify_regime(phi))
    ema, trend = tracker.update(sample)
    actions = controller.decide(sample, ema, trend, current_label_smoothing=0.1,
                                 current_lr_multiplier=1.0, previous_regime=tracker.previous_regime)

    arch_actions = [a for a in actions if a.action_type == "EMBEDDING_EXPANSION_RECOMMENDED"]
    no_live_payload = all(a.payload == {} for a in arch_actions)
    return check(len(arch_actions) == 1 and no_live_payload,
                 f"Low-BIS sample triggers exactly one EMBEDDING_EXPANSION_RECOMMENDED "
                 f"action with an empty (non-mutating) payload (got {arch_actions})")


def test_lr_multiplier_only_decreases():
    controller = rm.RegimeController(min_step=10, cooldown_steps=10)
    tracker = rm.RegimeTracker()

    bis, ttr, rci = 0.9, 0.9, 0.1  # high BIS, low RCI -> should trigger LR decay path
    phi = rm.compute_phase_transition_score(bis, ttr, rci)
    sample1 = rm.RegimeSample(step=20, bis=bis, ttr=ttr, rci=rci, phi=phi, regime=rm.classify_regime(phi))
    ema1, trend1 = tracker.update(sample1)
    # Force a "rising" trend by updating again with a slightly higher phi.
    sample2 = rm.RegimeSample(step=21, bis=bis, ttr=ttr, rci=rci * 0.9, phi=phi * 1.5,
                               regime=rm.classify_regime(phi * 1.5))
    ema2, trend2 = tracker.update(sample2)

    actions = controller.decide(sample2, ema2, trend2, current_label_smoothing=0.1,
                                 current_lr_multiplier=1.0, previous_regime=tracker.previous_regime)
    decay_actions = [a for a in actions if a.action_type == "DECAY_LEARNING_RATE"]
    if not decay_actions:
        return check(False, "DECAY_LEARNING_RATE should fire when BIS>0.75 and trend is rising")
    new_multiplier = decay_actions[0].payload["new_lr_multiplier"]
    return check(new_multiplier < 1.0, f"DECAY_LEARNING_RATE only ever decreases the multiplier (got {new_multiplier})")


def main():
    tests = [
        test_bis_clean_text,
        test_bis_glued_text,
        test_bis_no_false_positives_on_normal_words,
        test_phi_threshold_classification,
        test_rci_repetition_collapse,
        test_phi_formula,
        test_controller_cooldown_blocks_repeat_trigger,
        test_controller_never_mutates_architecture_live,
        test_lr_multiplier_only_decreases,
    ]
    all_passed = True
    for test in tests:
        print(f"\n--- {test.__name__} ---")
        all_passed = test() and all_passed

    print()
    if all_passed:
        print("ALL REGIME MONITOR CHECKS PASSED.")
    else:
        print("REGIME MONITOR CHECKS FAILED: see [FAIL] lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
