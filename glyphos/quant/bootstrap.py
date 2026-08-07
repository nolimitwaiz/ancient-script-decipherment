"""Document-level block bootstrap (§ Phase 6.2), pulled forward because no
headline number may ship without a CI.

Resampling is at the DOCUMENT level, never the sentence level: sentences
within a text are correlated (formulaic funerary phrases, shared scribe,
shared vocabulary), so sentence-level resampling understates variance and
manufactures significance. Paired bootstrap compares two systems on the same
resampled documents, which is the correct test for "is A better than B".
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from glyphos.utils.seed import derive_seed


@dataclass(frozen=True)
class CI:
    point: float
    low: float
    high: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.point:.2f} [{self.low:.2f}, {self.high:.2f}]"


def _group_indices(doc_ids: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for i, d in enumerate(doc_ids):
        groups.setdefault(d, []).append(i)
    return groups


def block_bootstrap(
    doc_ids: Sequence[str],
    score_fn: Callable[[Sequence[int]], float],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 1234,
) -> CI:
    """Resample documents with replacement; `score_fn` scores an index subset."""
    groups = _group_indices(doc_ids)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(f"need >= 2 documents to bootstrap, got {len(keys)}")
    rng = np.random.default_rng(derive_seed(seed, "block-bootstrap"))
    point = score_fn(list(range(len(doc_ids))))
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in picked for i in groups[keys[k]]]
        draws.append(score_fn(idx))
    alpha = (1 - level) / 2
    lo, hi = np.quantile(draws, [alpha, 1 - alpha])
    return CI(point=point, low=float(lo), high=float(hi), level=level)


@dataclass(frozen=True)
class PairedResult:
    delta: float
    low: float
    high: float
    p_value: float
    n_boot: int

    @property
    def significant(self) -> bool:
        """CI excludes zero at the requested level."""
        return (self.low > 0) or (self.high < 0)


def paired_block_bootstrap(
    doc_ids: Sequence[str],
    score_a: Callable[[Sequence[int]], float],
    score_b: Callable[[Sequence[int]], float],
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 1234,
) -> PairedResult:
    """Bootstrap the A-B difference on identical document resamples.

    p is the two-sided fraction of resamples where the difference reverses
    sign relative to the observed delta (a standard bootstrap significance
    proxy, reported alongside the CI rather than instead of it).
    """
    groups = _group_indices(doc_ids)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(f"need >= 2 documents to bootstrap, got {len(keys)}")
    rng = np.random.default_rng(derive_seed(seed, "paired-block-bootstrap"))
    all_idx = list(range(len(doc_ids)))
    observed = score_a(all_idx) - score_b(all_idx)
    deltas = []
    for _ in range(n_boot):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        idx = [i for k in picked for i in groups[keys[k]]]
        deltas.append(score_a(idx) - score_b(idx))
    deltas_arr = np.asarray(deltas)
    alpha = (1 - level) / 2
    lo, hi = np.quantile(deltas_arr, [alpha, 1 - alpha])
    reversals = float((deltas_arr * np.sign(observed) <= 0).mean())
    return PairedResult(
        delta=float(observed),
        low=float(lo),
        high=float(hi),
        p_value=min(1.0, 2 * reversals),
        n_boot=n_boot,
    )
