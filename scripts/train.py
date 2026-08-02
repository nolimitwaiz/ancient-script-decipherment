#!/usr/bin/env python
"""Model training entry point (Phase 3): explicit PyTorch loop, single-GPU and
DDP, bf16 on GPU, ledger-registered before the first step.

Stub until Phase 3; fails loudly rather than pretending.
"""

import sys


def main() -> int:
    print(
        "train.py: not implemented until Phase 3 (see docs/roadmap.md). "
        "Reminder: real training runs on the CLSP cluster, not this machine.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
