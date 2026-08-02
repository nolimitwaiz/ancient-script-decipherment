"""Repo-relative path resolution with environment overrides.

All locations are resolved per call (never cached at import) so tests and the
smoke pipeline can redirect state via environment variables.
"""

import os
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Root of all corpora. Contract: data/<corpus>/splits/<scheme>/<version>/<partition>/."""
    return Path(os.environ.get("GLYPHOS_DATA_ROOT", str(repo_root() / "data")))


def runs_dir() -> Path:
    return Path(os.environ.get("GLYPHOS_RUNS_DIR", str(repo_root() / "runs")))


def ledger_path() -> Path:
    return Path(os.environ.get("GLYPHOS_LEDGER_PATH", str(runs_dir() / "ledger.jsonl")))


def test_access_log_path() -> Path:
    return Path(os.environ.get("GLYPHOS_TEST_ACCESS_LOG", str(runs_dir() / "test_access.jsonl")))
