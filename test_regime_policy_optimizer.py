"""
Pure-Python tests for regime_policy_optimizer.py trajectory scoring.
"""

import json
import tempfile
from pathlib import Path

from regime_policy_optimizer import load_regime_records, score_trajectory, format_report


def _rec(step, bis, phi, regime):
    return {"step": step, "bis": bis, "phi": phi, "regime": regime}


def test_empty_records():
    score = score_trajectory([])
    assert score.num_probes == 0
    assert score.score_j == 0.0
    print("[PASS] empty records -> J=0")


def test_fast_emergence_scores_higher():
    fast = [
        _rec(100, 0.8, 0.7, "semantic_emergence"),
        _rec(200, 0.85, 1.0, "stable_generator"),
    ]
    slow = [
        _rec(500, 0.3, 0.2, "syntactic_drift"),
        _rec(600, 0.8, 0.7, "semantic_emergence"),
    ]
    j_fast = score_trajectory(fast).score_j
    j_slow = score_trajectory(slow).score_j
    assert j_fast > j_slow, f"fast={j_fast} should beat slow={j_slow}"
    print(f"[PASS] fast emergence J={j_fast:.4f} > slow J={j_slow:.4f}")


def test_fragmentation_penalized():
    clean = [_rec(100, 0.9, 0.8, "semantic_emergence")] * 5
    fragmented = [_rec(100, 0.2, 0.1, "attractor_collapse")] * 5
    j_clean = score_trajectory(clean).score_j
    j_frag = score_trajectory(fragmented).score_j
    assert j_clean > j_frag
    assert score_trajectory(fragmented).fragmentation_rate == 1.0
    print(f"[PASS] fragmentation penalized: clean J={j_clean:.4f} > frag J={j_frag:.4f}")


def test_load_jsonl_roundtrip():
    records = [_rec(100, 0.7, 0.65, "semantic_emergence")]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f)
            f.write("\n")
        path = f.name
    loaded = load_regime_records(path)
    Path(path).unlink(missing_ok=True)
    assert len(loaded) == 1
    assert loaded[0]["step"] == 100
    print("[PASS] JSONL load roundtrip")


def test_format_report():
    score = score_trajectory([_rec(200, 0.8, 0.9, "semantic_emergence")])
    text = format_report(score, "test.jsonl")
    assert "Trajectory score J" in text
    print("[PASS] format_report")


def main():
    test_empty_records()
    test_fast_emergence_scores_higher()
    test_fragmentation_penalized()
    test_load_jsonl_roundtrip()
    test_format_report()
    print("\nALL REGIME POLICY OPTIMIZER CHECKS PASSED.")


if __name__ == "__main__":
    main()
