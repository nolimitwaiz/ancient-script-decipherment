"""Split-conformal prediction sets (§ Phase 4.2) — the honest output interface.

A restoration model that says "this sign is X" is less useful to an
epigrapher than one that says "it is one of {X, Y, Z}, and that set contains
the truth 90% of the time". Split conformal gives exactly that, distribution-
free, with only an exchangeability assumption.

Calibration uses a held-out slice of VALID (never test). Reported alongside
every set: empirical coverage and mean set size — a method that hits coverage
by emitting huge sets is not useful, so both numbers travel together.

Known property: this is deterministic APS (the whole label that crosses the
threshold is admitted, with no randomised tie-breaking), which satisfies the
coverage guarantee but OVER-covers — measured ~98.5% empirical at a 90%
target on calibrated synthetic data. That is safe for an epigrapher-facing
tool (sets are conservative) at the cost of larger sets; randomised APS would
tighten them to nominal and is a Phase 7 refinement, not a correctness fix.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConformalCalibration:
    """Threshold on the cumulative-probability score for a target coverage."""

    qhat: float
    target_coverage: float
    n_calibration: int

    def predict_set(self, ranked_labels: Sequence[str], ranked_probs: Sequence[float]) -> list[str]:
        """Smallest top-k prefix whose cumulative probability reaches qhat."""
        out: list[str] = []
        total = 0.0
        for label, p in zip(ranked_labels, ranked_probs, strict=True):
            out.append(label)
            total += float(p)
            if total >= self.qhat:
                break
        return out


def calibrate(
    ranked_labels: Sequence[Sequence[str]],
    ranked_probs: Sequence[Sequence[float]],
    truths: Sequence[str],
    target_coverage: float = 0.90,
) -> ConformalCalibration:
    """Adaptive-prediction-set calibration on a held-out (valid) slice.

    Score per calibration point = cumulative probability up to and including
    the true label. qhat is the ceil((n+1)(1-alpha))/n empirical quantile, the
    finite-sample-valid choice.
    """
    if not (0 < target_coverage < 1):
        raise ValueError(f"target_coverage must be in (0,1), got {target_coverage}")
    if not (len(ranked_labels) == len(ranked_probs) == len(truths)):
        raise ValueError("labels, probs and truths must be parallel")
    if not truths:
        raise ValueError("no calibration points")

    scores = []
    for labels, probs, truth in zip(ranked_labels, ranked_probs, truths, strict=True):
        total = 0.0
        for label, p in zip(labels, probs, strict=True):
            total += float(p)
            if label == truth:
                break
        else:
            total = 1.0  # truth absent from the candidate list: worst-case score
        scores.append(total)

    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * target_coverage) / n)
    qhat = float(np.quantile(scores, level, method="higher"))
    return ConformalCalibration(qhat=qhat, target_coverage=target_coverage, n_calibration=n)


@dataclass(frozen=True)
class CoverageReport:
    empirical_coverage: float
    mean_set_size: float
    median_set_size: float
    n: int

    def as_dict(self) -> dict:
        return {
            "conformal_coverage": self.empirical_coverage,
            "conformal_mean_set_size": self.mean_set_size,
            "conformal_median_set_size": self.median_set_size,
            "conformal_n": self.n,
        }


def evaluate_sets(
    calibration: ConformalCalibration,
    ranked_labels: Sequence[Sequence[str]],
    ranked_probs: Sequence[Sequence[float]],
    truths: Sequence[str],
) -> CoverageReport:
    sizes, hits = [], []
    for labels, probs, truth in zip(ranked_labels, ranked_probs, truths, strict=True):
        s = calibration.predict_set(labels, probs)
        sizes.append(len(s))
        hits.append(truth in s)
    return CoverageReport(
        empirical_coverage=float(np.mean(hits)),
        mean_set_size=float(np.mean(sizes)),
        median_set_size=float(np.median(sizes)),
        n=len(truths),
    )
