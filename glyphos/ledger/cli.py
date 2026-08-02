"""Ledger CLI: `glyphos-ledger {report,list,show,register,complete}`.

`register`/`complete` exist so shell-driven runs (e.g. SLURM sbatch wrappers)
carry the exact same discipline as Python-driven ones.
"""

import argparse
import json
import sys

from glyphos.ledger.ledger import Ledger, LedgerError
from glyphos.ledger.report import build_reports, format_report
from glyphos.utils.config import load_yaml
from glyphos.utils.hashing import config_hash


def _cmd_report(ledger: Ledger, _args: argparse.Namespace) -> int:
    sys.stdout.write(format_report(build_reports(ledger)))
    return 0


def _cmd_list(ledger: Ledger, args: argparse.Namespace) -> int:
    for rec in ledger.load():
        if args.family and rec.family != args.family:
            continue
        if args.phase and rec.phase != args.phase:
            continue
        if args.status and rec.status != args.status:
            continue
        print(
            f"{rec.run_id}  {rec.status:<10}  {rec.phase:<8}  {rec.family:<28}  "
            f"seed={rec.seed}  cfg={rec.config_hash}  {rec.selection_metric}="
            f"{rec.all_metrics.get(rec.selection_metric, '-')}"
        )
    return 0


def _cmd_show(ledger: Ledger, args: argparse.Namespace) -> int:
    for rec in ledger.load():
        if rec.run_id == args.run_id:
            print(json.dumps(rec.__dict__, indent=2, sort_keys=True, ensure_ascii=False))
            return 0
    print(f"run {args.run_id!r} not found", file=sys.stderr)
    return 1


def _cmd_register(ledger: Ledger, args: argparse.Namespace) -> int:
    run_id = ledger.register(
        hypothesis=args.hypothesis,
        phase=args.phase,
        family=args.family,
        config_hash=config_hash(load_yaml(args.config)),
        data_version=args.data_version,
        split_version=args.split_version,
        seed=args.seed,
        selection_metric=args.selection_metric,
        notes=args.notes,
    )
    print(run_id)
    return 0


def _cmd_complete(ledger: Ledger, args: argparse.Namespace) -> int:
    if args.metrics_file:
        with open(args.metrics_file, encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = json.loads(args.metrics_json) if args.metrics_json else {}
    ledger.complete(args.run_id, args.status, metrics, notes=args.notes)
    print(f"{args.run_id} -> {args.status}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphos-ledger", description=__doc__)
    parser.add_argument("--ledger", default=None, help="ledger path (default: runs/ledger.jsonl)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("report", help="per-family multiple-testing report")

    p_list = sub.add_parser("list", help="list runs")
    p_list.add_argument("--family")
    p_list.add_argument("--phase")
    p_list.add_argument("--status")

    p_show = sub.add_parser("show", help="dump one run as JSON")
    p_show.add_argument("run_id")

    p_reg = sub.add_parser("register", help="preregister a run (prints run_id)")
    p_reg.add_argument("--hypothesis", required=True)
    p_reg.add_argument("--phase", required=True)
    p_reg.add_argument("--family", required=True)
    p_reg.add_argument("--config", required=True, help="YAML config file (hashed canonically)")
    p_reg.add_argument("--data-version", required=True)
    p_reg.add_argument("--split-version", required=True)
    p_reg.add_argument("--seed", required=True, type=int)
    p_reg.add_argument("--selection-metric", required=True)
    p_reg.add_argument("--notes", default="")

    p_done = sub.add_parser("complete", help="record a terminal status for a run")
    p_done.add_argument("run_id")
    p_done.add_argument("--status", required=True, choices=["completed", "failed", "abandoned"])
    p_done.add_argument("--metrics-json", default=None, help="e.g. '{\"chrf\": 41.2}'")
    p_done.add_argument("--metrics-file", default=None, help="JSON file of metrics")
    p_done.add_argument("--notes", default="")

    return parser


_COMMANDS = {
    "report": _cmd_report,
    "list": _cmd_list,
    "show": _cmd_show,
    "register": _cmd_register,
    "complete": _cmd_complete,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ledger = Ledger(args.ledger)
    try:
        return _COMMANDS[args.command](ledger, args)
    except LedgerError as exc:
        print(f"ledger error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
