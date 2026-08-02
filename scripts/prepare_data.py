#!/usr/bin/env python
"""Data acquisition, census, and split freezing (Phase 1).

Subcommands (arriving in Phase 1): ingest, census, split, freeze.
This stub exists so the entry-point layout is stable from day one; it fails
loudly rather than pretending (no silent fallbacks, working-style rule 4).
"""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["ingest", "census", "split", "freeze"])
    args = parser.parse_args(argv)
    print(
        f"prepare_data {args.command}: not implemented until Phase 1 "
        "(see docs/roadmap.md; current status in MEMORY.md)",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
