"""Ithaca-style restoration (§ Phase 4.2) — the task the pixel hypothesis
should actually win.

Rationale for the whole project's bet: when a sign is damaged, a *tokenizer*
must make a hard discrete decision (or emit UNK) before the model sees
anything, while a *pixel* encoder sees the surviving strokes and degrades
gracefully. Translation from clean transliteration gives text tokens every
advantage; restoration from degraded input is where images should pay off.

Masking curriculum (spec §6.2): single sign, Poisson spans, and
damage-shaped masks sampled from the Phase-2 degradation blobs. The model
outputs a ranked distribution over the sign vocabulary per masked slot;
metrics are top-k recall, per-frequency-bin breakdown (rare signs are the
point), and expected calibration error.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np

from glyphos.utils.seed import derive_seed

SINGLE, SPAN, DAMAGE = "single", "span", "damage"
CURRICULUM = (SINGLE, SPAN, DAMAGE)


@dataclass(frozen=True)
class MaskPlan:
    """Which sign positions are hidden, and by which curriculum mode."""

    positions: tuple[int, ...]
    mode: str

    def __len__(self) -> int:
        return len(self.positions)


def make_mask(
    n_signs: int,
    mode: str,
    seed: int,
    span_lambda: float = 3.0,
    damage_coverage: float = 0.25,
) -> MaskPlan:
    """Deterministic mask plan for a sequence of `n_signs` signs."""
    if n_signs <= 0:
        raise ValueError("cannot mask an empty sign sequence")
    if mode not in CURRICULUM:
        raise ValueError(f"mode must be one of {CURRICULUM}, got {mode!r}")
    rng = np.random.default_rng(derive_seed(seed, "restoration-mask", mode, str(n_signs)))

    if mode == SINGLE:
        return MaskPlan((int(rng.integers(0, n_signs)),), mode)

    if mode == SPAN:
        length = max(1, min(n_signs, int(rng.poisson(span_lambda)) or 1))
        start = int(rng.integers(0, n_signs - length + 1))
        return MaskPlan(tuple(range(start, start + length)), mode)

    # DAMAGE: contiguous-ish blobs, mirroring how physical damage lands —
    # matches the Phase-2 occlusion geometry rather than uniform dropout.
    target = max(1, round(damage_coverage * n_signs))
    hidden: set[int] = set()
    while len(hidden) < target:
        centre = int(rng.integers(0, n_signs))
        width = max(1, int(rng.poisson(2.0)))
        for p in range(max(0, centre - width // 2), min(n_signs, centre + width // 2 + 1)):
            hidden.add(p)
            if len(hidden) >= target:
                break
    return MaskPlan(tuple(sorted(hidden)), mode)


@dataclass
class RestorationScores:
    """top-k recall overall and split by sign frequency bin."""

    n_slots: int
    recall_at: dict[int, float]
    recall_by_bin: dict[str, dict[int, float]] = field(default_factory=dict)
    ece: float = float("nan")
    bin_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {"n_slots": self.n_slots, "ece": self.ece}
        out.update({f"recall@{k}": v for k, v in sorted(self.recall_at.items())})
        for name, rec in sorted(self.recall_by_bin.items()):
            out.update({f"recall@{k}_{name}": v for k, v in sorted(rec.items())})
        out.update({f"n_{name}": c for name, c in sorted(self.bin_counts.items())})
        return out


def frequency_bins(counts: dict[str, int], n_bins: int = 3) -> dict[str, str]:
    """Map each sign to a frequency bin label; the rare bin is what matters."""
    if not counts:
        raise ValueError("empty frequency table")
    ranked = sorted(counts, key=lambda s: (-counts[s], s))
    edges = [len(ranked) * (i + 1) // n_bins for i in range(n_bins)]
    names = ["frequent", "medium", "rare"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    out, start = {}, 0
    for name, end in zip(names, edges, strict=True):
        for sign in ranked[start:end]:
            out[sign] = name
        start = end
    return out


def score_restoration(
    ranked_predictions: Sequence[Sequence[str]],
    truths: Sequence[str],
    ks: Sequence[int] = (1, 5, 20),
    sign_bins: dict[str, str] | None = None,
    confidences: Sequence[float] | None = None,
    n_calibration_bins: int = 10,
) -> RestorationScores:
    """`ranked_predictions[i]` is the ranked candidate list for masked slot i."""
    if len(ranked_predictions) != len(truths):
        raise ValueError("predictions and truths must be parallel")
    if not truths:
        raise ValueError("no masked slots to score")

    recall = {k: 0 for k in ks}
    per_bin: dict[str, dict[int, int]] = {}
    bin_counts: dict[str, int] = {}
    for ranked, truth in zip(ranked_predictions, truths, strict=True):
        label = (sign_bins or {}).get(truth, "all")
        bin_counts[label] = bin_counts.get(label, 0) + 1
        per_bin.setdefault(label, dict.fromkeys(ks, 0))
        for k in ks:
            if truth in list(ranked)[:k]:
                recall[k] += 1
                per_bin[label][k] += 1

    n = len(truths)
    scores = RestorationScores(
        n_slots=n,
        recall_at={k: recall[k] / n for k in ks},
        recall_by_bin={
            name: {k: hits[k] / bin_counts[name] for k in ks} for name, hits in per_bin.items()
        },
        bin_counts=bin_counts,
    )
    if confidences is not None:
        scores.ece = expected_calibration_error(
            confidences,
            [
                bool(list(r)) and next(iter(r)) == t
                for r, t in zip(ranked_predictions, truths, strict=True)
            ],
            n_bins=n_calibration_bins,
        )
    return scores


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> float:
    """Standard ECE: |accuracy - confidence| averaged over equal-width bins."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correctness must be parallel")
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.size == 0:
        raise ValueError("no predictions to calibrate")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in pairwise(edges):
        sel = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not sel.any():
            continue
        ece += sel.mean() * abs(hit[sel].mean() - conf[sel].mean())
    return float(ece)
