"""The smoke pipeline is itself under test: pytest runs the same end-to-end
path as `make smoke`, against fully isolated tmp state (see conftest)."""

from glyphos.data import guard
from glyphos.ledger import Ledger
from glyphos.tasks.smoke import SmokeConfig, run_smoke


def test_run_smoke_end_to_end():
    metrics = run_smoke(SmokeConfig())

    assert metrics["key_accuracy"] >= 0.5
    assert 0.0 <= metrics["valid_token_accuracy"] <= 1.0
    assert 0.0 <= metrics["test_token_accuracy"] <= 1.0
    assert metrics["n_train"] > metrics["n_test"] > 0

    # exactly one run, preregistered and completed, with metrics attached
    [rec] = Ledger().load()
    assert rec.status == "completed"
    assert rec.phase == "phase0"
    assert rec.hypothesis
    assert rec.all_metrics["key_accuracy"] == metrics["key_accuracy"]
    assert rec.n_variants_tried_so_far_in_this_family == 1

    # the locked-test read was audited and attributed to that run
    entries = [e for e in guard.read_access_log() if e["run_id"] == rec.run_id]
    assert any("/test/" in e["path"] for e in entries)


def test_run_smoke_is_deterministic():
    m1 = run_smoke(SmokeConfig())
    m2 = run_smoke(SmokeConfig())
    assert m1 == m2
    # both runs are in the ledger; the family counter advanced
    counts = sorted(r.n_variants_tried_so_far_in_this_family for r in Ledger().load())
    assert counts == [1, 2]
