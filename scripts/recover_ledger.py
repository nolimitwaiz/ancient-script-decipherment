#!/usr/bin/env python
"""Reconstruct ledger events for runs whose entries were lost.

Evidence, all written BEFORE results existed or by the job itself:
  runs/checkpoints/<run_id>/config.json  — the run config (hypothesis included),
                                            written at run start
  runs/slurm-<jobid>.out                 — the job's own stdout (metrics)

Reconstruction is NOT preregistration. Every recovered run is marked
`reconstructed=true` in its metrics and its notes cite the evidence, so the
record is honest about its provenance. Runs that already exist are skipped.

    python scripts/recover_ledger.py --checkpoints runs/checkpoints \
        --logs runs --apply
"""

import argparse
import json
import re
from pathlib import Path

from glyphos.ledger import Ledger
from glyphos.utils.hashing import config_hash

DONE_RE = re.compile(
    r"\[train\] done: (\d+) steps, loss ([\d.]+) -> ([\d.]+), best_eval=([\d.a-z]+)"
)
PARAMS_RE = re.compile(r"\[train\] ([\d,]+) params -> (\S+)")


def scan_logs(log_dir: Path) -> dict[str, dict]:
    """run_id -> {job, steps, first_loss, last_loss, best_eval, params}."""
    out: dict[str, dict] = {}
    for log in sorted(log_dir.glob("slurm-*.out")):
        text = log.read_text(errors="replace")
        pm = PARAMS_RE.search(text)
        dm = DONE_RE.search(text)
        if not (pm and dm):
            continue
        run_id = Path(pm.group(2)).name
        best = dm.group(4)
        out[run_id] = {
            "job": log.stem.replace("slurm-", ""),
            "params": int(pm.group(1).replace(",", "")),
            "steps": int(dm.group(1)),
            "first_loss": float(dm.group(2)),
            "last_loss": float(dm.group(3)),
            "best_eval": float("inf") if best == "inf" else float(best),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", default="runs/checkpoints")
    ap.add_argument("--logs", default="runs")
    ap.add_argument("--apply", action="store_true", help="write events (default: dry run)")
    args = ap.parse_args(argv)

    ledger = Ledger()
    known = {r.run_id for r in ledger.load()}
    metrics_by_run = scan_logs(Path(args.logs))

    recovered = 0
    for cfg_path in sorted(Path(args.checkpoints).glob("*/config.json")):
        run_id = cfg_path.parent.name
        if run_id in known:
            continue
        metrics = metrics_by_run.get(run_id)
        if metrics is None:
            print(f"[skip] {run_id}: no completed-run evidence in logs")
            continue
        cfg = json.loads(cfg_path.read_text())
        note = (
            f"RECONSTRUCTED after sync.sh overwrote the cluster ledger (2026-08-06); "
            f"evidence: slurm-{metrics['job']}.out + checkpoints/{run_id}/config.json"
        )
        print(
            f"[recover] {run_id} {cfg['family']} seed={cfg['seed']} "
            f"best_eval={metrics['best_eval']:.4f} ({note.split(';')[0]})"
        )
        if not args.apply:
            recovered += 1
            continue
        # register with the ORIGINAL run_id preserved, then complete
        event_common = {
            "schema_version": 1,
            "run_id": run_id,
            "timestamp": _ts_from_run_id(run_id),
        }
        ledger._append(
            {
                **event_common,
                "event": "register",
                "git_hash": "reconstructed",
                "hypothesis": cfg["hypothesis"].strip(),
                "phase": "phase3",
                "family": cfg["family"],
                "config_hash": config_hash(cfg),
                "data_version": cfg.get("data_version", "see config.json"),
                "split_version": cfg.get("split_version", "see config.json"),
                "seed": cfg["seed"],
                "selection_metric": "best_eval",
                "n_variants_tried_so_far_in_this_family": 0,
                "notes": note,
            }
        )
        ledger._append(
            {
                **event_common,
                "event": "complete",
                "status": "completed",
                "all_metrics": {**metrics, "reconstructed": True},
                "notes": note,
            }
        )
        recovered += 1

    print(f"\n{'recovered' if args.apply else 'would recover'} {recovered} run(s)")
    if args.apply and recovered:
        print(f"compacted {ledger.compact()} event(s)")
    return 0


def _ts_from_run_id(run_id: str) -> str:
    d, t = run_id.split("-")[:2]
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}+00:00"


if __name__ == "__main__":
    raise SystemExit(main())
