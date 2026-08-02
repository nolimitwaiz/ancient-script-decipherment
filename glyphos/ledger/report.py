"""Multiple-testing-aware ledger reports (§ Phase 0.2).

Per experiment family the report shows the number of variants attempted, the
full distribution of the selection metric (never just the max), and rank
stability of configs across seeds. A winner is only reportable alongside
everything that was tried — White's reality-check culture, enforced by tooling.
"""

import statistics
from dataclasses import dataclass

from glyphos.ledger.ledger import Ledger, RunRecord


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Kendall tau-a over paired scores. O(n^2); n here is the number of configs."""
    if len(a) != len(b):
        raise ValueError("kendall_tau: sequences must have equal length")
    n = len(a)
    if n < 2:
        return float("nan")
    concordant_minus_discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = (a[i] > a[j]) - (a[i] < a[j])
            db = (b[i] > b[j]) - (b[i] < b[j])
            concordant_minus_discordant += da * db
    return concordant_minus_discordant / (n * (n - 1) / 2)


def _selection_value(rec: RunRecord) -> float | None:
    value = rec.all_metrics.get(rec.selection_metric)
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


@dataclass
class FamilyReport:
    family: str
    selection_metric: str
    n_registered: int
    n_completed: int
    n_failed: int
    n_abandoned: int
    n_still_open: int
    values: list[float]  # completed runs' selection-metric values, best first
    n_configs: int
    n_seeds: int
    rank_stability: float | None  # mean pairwise Kendall tau of config rankings across seeds

    @property
    def best(self) -> float | None:
        return self.values[0] if self.values else None

    @property
    def median(self) -> float | None:
        return statistics.median(self.values) if self.values else None

    @property
    def worst(self) -> float | None:
        return self.values[-1] if self.values else None


def _rank_stability(completed: list[RunRecord]) -> tuple[int, int, float | None]:
    """Mean pairwise Kendall tau between per-seed rankings of configs.

    For each seed, configs are scored by their best completed value under that
    seed; for every pair of seeds, tau is computed over the configs both seeds
    share (>= 2 required). None when fewer than 2 seeds/configs overlap.
    """
    by_seed: dict[int, dict[str, float]] = {}
    for rec in completed:
        value = _selection_value(rec)
        if value is None:
            continue
        seed_scores = by_seed.setdefault(rec.seed, {})
        prior = seed_scores.get(rec.config_hash)
        seed_scores[rec.config_hash] = value if prior is None else max(prior, value)

    configs = {c for scores in by_seed.values() for c in scores}
    seeds = sorted(by_seed)
    taus = []
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i + 1 :]:
            shared = sorted(set(by_seed[s1]) & set(by_seed[s2]))
            if len(shared) >= 2:
                taus.append(
                    kendall_tau(
                        [by_seed[s1][c] for c in shared],
                        [by_seed[s2][c] for c in shared],
                    )
                )
    return len(configs), len(seeds), (sum(taus) / len(taus) if taus else None)


def family_report(family: str, records: list[RunRecord]) -> FamilyReport:
    completed = [r for r in records if r.status == "completed"]
    values = sorted((v for r in completed if (v := _selection_value(r)) is not None), reverse=True)
    metrics = {r.selection_metric for r in records}
    n_configs, n_seeds, stability = _rank_stability(completed)
    return FamilyReport(
        family=family,
        selection_metric=" / ".join(sorted(metrics)),
        n_registered=len(records),
        n_completed=len(completed),
        n_failed=sum(r.status == "failed" for r in records),
        n_abandoned=sum(r.status == "abandoned" for r in records),
        n_still_open=sum(r.status == "registered" for r in records),
        values=values,
        n_configs=n_configs,
        n_seeds=n_seeds,
        rank_stability=stability,
    )


def build_reports(ledger: Ledger) -> list[FamilyReport]:
    return [family_report(fam, recs) for fam, recs in sorted(ledger.families().items())]


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.4f}"


def format_report(reports: list[FamilyReport]) -> str:
    if not reports:
        return "ledger is empty — no runs registered yet\n"
    lines = [
        "| family | metric | tried | done | fail | aband | open | best | median | worst "
        "| configs | seeds | rank-stability |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r.family} | {r.selection_metric} | {r.n_registered} | {r.n_completed} "
            f"| {r.n_failed} | {r.n_abandoned} | {r.n_still_open} | {_fmt(r.best)} "
            f"| {_fmt(r.median)} | {_fmt(r.worst)} | {r.n_configs} | {r.n_seeds} "
            f"| {_fmt(r.rank_stability)} |"
        )
    lines.append("")
    for r in reports:
        if r.values:
            shown = ", ".join(f"{v:.4f}" for v in r.values[:20])
            suffix = f" … (+{len(r.values) - 20} more)" if len(r.values) > 20 else ""
            lines.append(f"{r.family}: all completed values (best→worst): {shown}{suffix}")
    lines.append("")
    lines.append(
        "Reporting rule: any exported winner must cite its family's `tried` count "
        "and this distribution, not the best value alone."
    )
    return "\n".join(lines) + "\n"
