"""
Offline trajectory scoring for Regime Controller runs.

Reads probe history from output/regime_metrics_latest.jsonl (or any compatible
JSONL file) and computes a scalar trajectory score J that rewards:

  - faster semantic emergence (fewer steps to first Phi >= phi_emerge_threshold)
  - longer dwell in advanced regimes (semantic_emergence + stable_generator)
  - lower fragmentation probability (fraction of probes with BIS < bis_frag_threshold)

This is policy *evaluation*, not live control — use it to compare runs or tune
controller threshold grids offline without GPU time.

Usage:
    python regime_policy_optimizer.py [path/to/regime_metrics.jsonl]
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ADVANCED_REGIMES = frozenset({"semantic_emergence", "stable_generator"})
DEFAULT_PHI_EMERGE = 0.6
DEFAULT_BIS_FRAG = 0.5


@dataclass
class TrajectoryScore:
    """Scalar summary of one training run's regime trajectory."""
    num_probes: int
    steps_to_emergence: Optional[int]
    dwell_steps: int
    fragmentation_rate: float
    mean_phi: float
    final_regime: Optional[str]
    score_j: float

    def as_dict(self):
        return {
            "num_probes": self.num_probes,
            "steps_to_emergence": self.steps_to_emergence,
            "dwell_steps": self.dwell_steps,
            "fragmentation_rate": round(self.fragmentation_rate, 4),
            "mean_phi": round(self.mean_phi, 4),
            "final_regime": self.final_regime,
            "score_j": round(self.score_j, 4),
        }


def load_regime_records(path):
    """Load regime probe records from JSONL. Skips blank lines."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def score_trajectory(records, *,
                     alpha=1.0, beta=0.001, gamma=2.0,
                     phi_emerge_threshold=DEFAULT_PHI_EMERGE,
                     bis_frag_threshold=DEFAULT_BIS_FRAG,
                     probe_interval_hint=100):
    """Compute trajectory score J from a list of regime probe records.

    Args:
        records: list of dicts with at least step, bis, phi, regime
        alpha: weight for faster emergence (higher = reward lower T_emerge)
        beta: weight per probe-step spent in advanced regimes
        gamma: penalty weight for fragmentation rate
        phi_emerge_threshold: Phi level defining semantic emergence
        bis_frag_threshold: BIS level defining fragmentation
        probe_interval_hint: used to convert probe count to approximate step dwell
            when records don't cover every step between probes

    Returns:
        TrajectoryScore
    """
    if not records:
        return TrajectoryScore(
            num_probes=0, steps_to_emergence=None, dwell_steps=0,
            fragmentation_rate=0.0, mean_phi=0.0, final_regime=None, score_j=0.0,
        )

    steps_to_emergence = None
    for rec in records:
        phi = float(rec.get("phi", 0.0))
        step = int(rec.get("step", 0))
        if phi >= phi_emerge_threshold:
            steps_to_emergence = step
            break

    dwell_probes = sum(1 for rec in records if rec.get("regime") in ADVANCED_REGIMES)
    dwell_steps = dwell_probes * probe_interval_hint

    frag_probes = sum(1 for rec in records if float(rec.get("bis", 1.0)) < bis_frag_threshold)
    fragmentation_rate = frag_probes / len(records)

    mean_phi = sum(float(rec.get("phi", 0.0)) for rec in records) / len(records)
    final_regime = records[-1].get("regime")

    # J: reward fast emergence + long dwell, penalize fragmentation
    emerge_term = 0.0
    if steps_to_emergence is not None and steps_to_emergence > 0:
        emerge_term = alpha / steps_to_emergence
    elif steps_to_emergence == 0:
        emerge_term = alpha

    score_j = emerge_term + beta * dwell_steps - gamma * fragmentation_rate

    return TrajectoryScore(
        num_probes=len(records),
        steps_to_emergence=steps_to_emergence,
        dwell_steps=dwell_steps,
        fragmentation_rate=fragmentation_rate,
        mean_phi=mean_phi,
        final_regime=final_regime,
        score_j=score_j,
    )


def format_report(score: TrajectoryScore, path=None):
    lines = ["Regime Trajectory Report"]
    if path:
        lines.append(f"  Source: {path}")
    lines.append(f"  Probes:              {score.num_probes}")
    if score.steps_to_emergence is None:
        lines.append("  Steps to emergence:  never reached (Phi >= 0.6)")
    else:
        lines.append(f"  Steps to emergence:  {score.steps_to_emergence}")
    lines.append(f"  Advanced-regime dwell (approx steps): {score.dwell_steps}")
    lines.append(f"  Fragmentation rate:  {score.fragmentation_rate:.1%} (BIS < 0.5)")
    lines.append(f"  Mean Phi:            {score.mean_phi:.3f}")
    lines.append(f"  Final regime:        {score.final_regime or 'n/a'}")
    lines.append(f"  Trajectory score J:  {score.score_j:.4f}")
    lines.append("")
    lines.append("  Higher J = faster emergence + longer stable dwell + less fragmentation.")
    return "\n".join(lines)


def main():
    default_path = Path("output/regime_metrics_latest.jsonl")
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not path.exists():
        print(f"No regime telemetry at {path}")
        print("Run auto_train.py first; probes write JSONL every 100 steps after step 50.")
        raise SystemExit(1)

    records = load_regime_records(path)
    score = score_trajectory(records)
    print(format_report(score, path))
    print(json.dumps(score.as_dict(), indent=2))


if __name__ == "__main__":
    main()
