import math

from glyphos.ledger import Ledger
from glyphos.ledger.report import build_reports, format_report, kendall_tau


def test_kendall_tau_perfect_and_reversed():
    assert kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
    assert math.isnan(kendall_tau([1], [1]))


def _run(ledger, family, config_hash, seed, value):
    with ledger.run(
        hypothesis="grid point",
        phase="phase0",
        family=family,
        config_hash=config_hash,
        data_version="d",
        split_version="s",
        seed=seed,
        selection_metric="acc",
        git_hash="abc123",
    ) as run:
        run.log_metric("acc", value)


def test_family_report_distribution_and_rank_stability():
    ledger = Ledger()
    # 2 configs x 2 seeds; config c2 wins under both seeds -> rank stability 1.0
    _run(ledger, "fam", "c1", 1, 0.50)
    _run(ledger, "fam", "c2", 1, 0.70)
    _run(ledger, "fam", "c1", 2, 0.55)
    _run(ledger, "fam", "c2", 2, 0.75)
    [report] = build_reports(ledger)
    assert report.n_registered == 4
    assert report.n_completed == 4
    assert report.values == [0.75, 0.70, 0.55, 0.50]
    assert report.best == 0.75
    assert report.worst == 0.50
    assert report.n_configs == 2
    assert report.n_seeds == 2
    assert report.rank_stability == 1.0


def test_rank_stability_detects_disagreement():
    ledger = Ledger()
    _run(ledger, "fam", "c1", 1, 0.9)
    _run(ledger, "fam", "c2", 1, 0.1)
    _run(ledger, "fam", "c1", 2, 0.1)
    _run(ledger, "fam", "c2", 2, 0.9)
    [report] = build_reports(ledger)
    assert report.rank_stability == -1.0


def test_format_report_shows_distribution_not_just_max():
    ledger = Ledger()
    _run(ledger, "fam", "c1", 1, 0.50)
    _run(ledger, "fam", "c2", 1, 0.70)
    text = format_report(build_reports(ledger))
    assert "0.7000" in text and "0.5000" in text
    assert "tried" in text


def test_empty_ledger_report():
    assert "empty" in format_report(build_reports(Ledger()))
