"""Content hashing for configs, files, and data directories.

These hashes are identity: `config_hash` and `data_version` recorded in the
ledger must be reproducible from content alone (key order, formatting, and
filesystem ordering must not matter).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_LEN = 12


def sha256_hex(data: bytes, n: int = DEFAULT_LEN) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, non-JSON types stringified."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(cfg: Any, n: int = DEFAULT_LEN) -> str:
    return sha256_hex(canonical_json(cfg).encode("utf-8"), n)


def hash_file(path: str | Path, n: int = DEFAULT_LEN) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def hash_dir(root: str | Path, pattern: str = "**/*", n: int = DEFAULT_LEN) -> str:
    """Hash of (relative path, content) pairs over all files, sorted by path."""
    root = Path(root)
    h = hashlib.sha256()
    files = sorted(p for p in root.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"hash_dir: no files under {root} matching {pattern!r}")
    for p in files:
        h.update(str(p.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(hash_file(p, n=64).encode("ascii"))
        h.update(b"\0")
    return h.hexdigest()[:n]
