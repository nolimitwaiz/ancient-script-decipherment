import json

import pytest

from glyphos.ledger import Ledger, LedgerError


def _register(ledger, **overrides):
    kwargs = {
        "hypothesis": "toy hypothesis: metric should exceed 0.5",
        "phase": "phase0",
        "family": "fam-a",
        "config_hash": "cfg000000001",
        "data_version": "dat000000001",
        "split_version": "spl000000001",
        "seed": 1,
        "selection_metric": "acc",
        "git_hash": "abc123",
    }
    kwargs.update(overrides)
    return ledger.register(**kwargs)


def test_register_requires_hypothesis():
    ledger = Ledger()
    with pytest.raises(LedgerError, match="hypothesis"):
        _register(ledger, hypothesis="   ")


def test_register_requires_int_seed():
    ledger = Ledger()
    with pytest.raises(LedgerError, match="seed"):
        _register(ledger, seed="1")


def test_register_requires_nonempty_metadata():
    ledger = Ledger()
    with pytest.raises(LedgerError, match="data_version"):
        _register(ledger, data_version="")


def test_variant_counter_increments_per_family():
    ledger = Ledger()
    _register(ledger, family="fam-a")
    _register(ledger, family="fam-a")
    _register(ledger, family="fam-b")
    records = {(r.family, r.n_variants_tried_so_far_in_this_family) for r in ledger.load()}
    assert ("fam-a", 1) in records
    assert ("fam-a", 2) in records
    assert ("fam-b", 1) in records


def test_complete_merges_and_is_append_only():
    ledger = Ledger()
    run_id = _register(ledger)
    ledger.complete(run_id, "completed", {"acc": 0.9}, notes="fine")
    [rec] = ledger.load()
    assert rec.status == "completed"
    assert rec.all_metrics == {"acc": 0.9}
    # two physical lines: register + complete, nothing rewritten
    ledger.compact()  # events land in the run's shard until compacted
    lines = [json.loads(line) for line in ledger.path.read_text().splitlines() if line.strip()]
    assert [e["event"] for e in lines] == ["register", "complete"]


def test_complete_unknown_run_rejected():
    ledger = Ledger()
    with pytest.raises(LedgerError, match="unknown run_id"):
        ledger.complete("nope", "completed")


def test_double_complete_rejected():
    ledger = Ledger()
    run_id = _register(ledger)
    ledger.complete(run_id, "completed")
    with pytest.raises(LedgerError, match="append-only"):
        ledger.complete(run_id, "failed")


def test_invalid_status_rejected():
    ledger = Ledger()
    run_id = _register(ledger)
    with pytest.raises(LedgerError, match="status"):
        ledger.complete(run_id, "great")


def test_run_context_manager_records_failure_and_reraises():
    ledger = Ledger()
    with (
        pytest.raises(ValueError, match="boom"),
        ledger.run(
            hypothesis="will crash",
            phase="phase0",
            family="fam-crash",
            config_hash="c",
            data_version="d",
            split_version="s",
            seed=1,
            selection_metric="acc",
            git_hash="abc123",
        ) as run,
    ):
        run.log_metric("partial", 0.1)
        raise ValueError("boom")
    [rec] = ledger.load()
    assert rec.status == "failed"
    assert rec.all_metrics == {"partial": 0.1}
    assert "boom" in rec.notes


def test_corrupt_line_is_a_hard_error(tmp_path):
    ledger = Ledger(tmp_path / "ledger.jsonl")
    _register(ledger)
    with open(ledger.path, "a") as f:
        f.write("{not json\n")
    with pytest.raises(LedgerError, match="corrupt"):
        ledger.load()


def test_concurrent_runs_do_not_lose_events(tmp_path):
    """Regression: SSL kill-gate 1709903 lost its registration when six SLURM
    jobs appended to one NFS file. Each run now owns a shard."""
    import multiprocessing as mp

    ledger_path = tmp_path / "ledger.jsonl"

    def worker(i):
        led = Ledger(ledger_path)
        with led.run(
            hypothesis=f"concurrent run {i}",
            phase="phase3",
            family="concurrency",
            config_hash=f"cfg{i}",
            data_version="d",
            split_version="s",
            seed=i,
            selection_metric="acc",
            git_hash="abc123",
        ) as run:
            run.log_metric("acc", 0.1 * i)

    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=worker, args=(i,)) for i in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)

    records = Ledger(ledger_path).load()
    assert len(records) == 8, f"lost events: only {len(records)} of 8 runs survived"
    assert all(r.status == "completed" for r in records)
    assert {r.seed for r in records} == set(range(8))


def test_compact_folds_shards_into_canonical_file(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    led = Ledger(ledger_path)
    run_id = _register(led)
    led.complete(run_id, "completed", {"acc": 0.5})
    assert led.shard_dir.exists() and list(led.shard_dir.glob("*.jsonl"))

    merged = led.compact()
    assert merged == 2  # register + complete
    assert not list(led.shard_dir.glob("*.jsonl"))
    [rec] = Ledger(ledger_path).load()
    assert rec.status == "completed" and rec.all_metrics == {"acc": 0.5}
    assert led.compact() == 0  # idempotent


def test_load_tolerates_event_present_in_both_file_and_shard(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    led = Ledger(ledger_path)
    run_id = _register(led)
    led.complete(run_id, "completed", {"acc": 1.0})
    shard = next(led.shard_dir.glob("*.jsonl"))
    lines = shard.read_text()
    with open(ledger_path, "a") as f:  # duplicate every event into the main file
        f.write(lines)
    [rec] = Ledger(ledger_path).load()  # must not raise "duplicate register"
    assert rec.status == "completed"
