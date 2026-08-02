#!/usr/bin/env python
"""`make smoke` entry point: full toy pipeline on CPU, self-checking.

Smoke data is isolated under runs/smoke/data (never the real data/ tree) unless
GLYPHOS_DATA_ROOT is set explicitly. Smoke runs ARE logged to the real ledger:
every run is logged, no exceptions — including trivial ones.
"""

import argparse
import os
import sys
import time

# Isolate smoke data before glyphos.utils.paths is consulted.
if "GLYPHOS_DATA_ROOT" not in os.environ:
    _default_runs = os.environ.get(
        "GLYPHOS_RUNS_DIR",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"),
    )
    os.environ["GLYPHOS_DATA_ROOT"] = os.path.join(_default_runs, "smoke", "data")

from glyphos.ledger import Ledger
from glyphos.ledger.report import build_reports, format_report
from glyphos.tasks.smoke import SmokeConfig, SmokeFailure, run_smoke
from glyphos.utils.config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/smoke.yaml")
    args = parser.parse_args(argv)

    started = time.monotonic()
    cfg = load_config(SmokeConfig, args.config)
    try:
        run_smoke(cfg)
    except SmokeFailure as exc:
        print(f"[smoke] FAIL: {exc}", file=sys.stderr)
        return 1

    print("\n[smoke] ledger report:")
    print(format_report(build_reports(Ledger())))
    print(f"[smoke] PASS in {time.monotonic() - started:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
