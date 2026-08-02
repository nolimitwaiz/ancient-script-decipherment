#!/usr/bin/env python
"""Stale-report gate: every report .tex must have a PDF that is (a) present,
(b) not older than its source, and (c) contains the source's \\ReportStamp
string (extracted via pypdf). Convention: bump the stamp on every substantive
edit (phase0-v1 -> phase0-v2 ...), then run `make reports`.
"""

import re
from pathlib import Path

from pypdf import PdfReader

REPORTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "reports"
STAMP_RE = re.compile(r"\\newcommand\{\\ReportStamp\}\{([^}]*)\}")


def main() -> int:
    mains = [
        tex
        for tex in sorted(REPORTS_DIR.rglob("*.tex"))
        if "\\documentclass" in tex.read_text(encoding="utf-8")
    ]
    if not mains:
        print("check_reports_fresh: no reports yet")
        return 0

    failures = []
    for tex in mains:
        pdf = tex.with_suffix(".pdf")
        if not pdf.exists():
            failures.append(f"{tex}: PDF missing — run `make reports`")
            continue
        if tex.stat().st_mtime > pdf.stat().st_mtime:
            failures.append(f"{tex}: .tex newer than PDF — run `make reports`")
            continue
        match = STAMP_RE.search(tex.read_text(encoding="utf-8"))
        if not match:
            failures.append(f"{tex}: missing \\ReportStamp{{...}} (required in every report)")
            continue
        stamp = match.group(1)
        text = "".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
        if stamp not in text:
            failures.append(
                f"{tex}: stamp {stamp!r} not found in {pdf.name} — stale PDF, run `make reports`"
            )

    if failures:
        print("STALE REPORTS:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"check_reports_fresh: all {len(mains)} report(s) fresh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
