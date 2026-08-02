import json

from glyphos.ledger import Ledger
from glyphos.ledger.cli import main


def test_cli_register_complete_report_roundtrip(tmp_path, capsys):
    ledger_path = tmp_path / "ledger.jsonl"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("lr: 0.001\nlayers: 12\n")

    rc = main(
        [
            "--ledger",
            str(ledger_path),
            "register",
            "--hypothesis",
            "pixel > bpe on toy",
            "--phase",
            "phase0",
            "--family",
            "cli-family",
            "--config",
            str(cfg),
            "--data-version",
            "d1",
            "--split-version",
            "s1",
            "--seed",
            "3",
            "--selection-metric",
            "chrf",
        ]
    )
    assert rc == 0
    run_id = capsys.readouterr().out.strip()

    metrics_file = tmp_path / "m.json"
    metrics_file.write_text(json.dumps({"chrf": 41.2}))
    rc = main(
        [
            "--ledger",
            str(ledger_path),
            "complete",
            run_id,
            "--status",
            "completed",
            "--metrics-file",
            str(metrics_file),
        ]
    )
    assert rc == 0
    capsys.readouterr()

    assert main(["--ledger", str(ledger_path), "report"]) == 0
    out = capsys.readouterr().out
    assert "cli-family" in out
    assert "41.2" in out

    [rec] = Ledger(ledger_path).load()
    assert rec.all_metrics == {"chrf": 41.2}


def test_cli_show_missing_run(tmp_path, capsys):
    assert main(["--ledger", str(tmp_path / "l.jsonl"), "show", "ghost"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_register_rejects_empty_hypothesis(tmp_path, capsys):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1\n")
    rc = main(
        [
            "--ledger",
            str(tmp_path / "l.jsonl"),
            "register",
            "--hypothesis",
            "  ",
            "--phase",
            "p",
            "--family",
            "f",
            "--config",
            str(cfg),
            "--data-version",
            "d",
            "--split-version",
            "s",
            "--seed",
            "1",
            "--selection-metric",
            "m",
        ]
    )
    assert rc == 1
    assert "hypothesis" in capsys.readouterr().err
