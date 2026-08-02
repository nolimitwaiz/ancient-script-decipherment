"""Frozen test partitions (§ Phase 1.3): hash, commit, enforce.

`write_split` materializes a split under the data-dir contract;
`freeze_split` records the test partition's content hash in the repo-committed
manifest (configs/frozen_splits.json — data/ itself is never committed, so
split identity must live in git). The guard consults this manifest on every
audited access: a read of a frozen file whose hash no longer matches, or any
write to a frozen file, is a hard error — not a log line.
"""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from glyphos.data.schema import Record, write_records
from glyphos.data.splits import PARTS, split_version
from glyphos.utils import paths
from glyphos.utils.hashing import hash_file


class FrozenSplitViolation(RuntimeError):
    pass


def manifest_path() -> Path:
    return Path(
        os.environ.get(
            "GLYPHOS_FREEZE_MANIFEST", str(paths.repo_root() / "configs" / "frozen_splits.json")
        )
    )


_cache: tuple[str, float, dict] | None = None


def load_manifest() -> dict:
    global _cache
    mp = manifest_path()
    if not mp.exists():
        return {}
    key = (str(mp), mp.stat().st_mtime)
    if _cache is not None and _cache[:2] == key:
        return _cache[2]
    with open(mp, encoding="utf-8") as f:
        manifest = json.load(f)
    _cache = (*key, manifest)
    return manifest


def _save_manifest(manifest: dict) -> None:
    global _cache
    mp = manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    _cache = None


def _rel_key(path: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(paths.data_root().resolve()))
    except ValueError:
        return None


def write_split(
    splits: dict[str, list[Record]], corpus: str, scheme: str, tag: str = "v1"
) -> tuple[str, Path]:
    """Write partitions under the data-dir contract; returns (split_version, dir)."""
    root = paths.data_root() / corpus / "splits" / scheme / tag
    for part in PARTS:
        write_records(splits[part], root / part / "records.jsonl")
    return split_version(splits), root


def freeze_split(corpus: str, scheme: str, tag: str = "v1") -> str:
    """Freeze the TEST partition of an existing split; returns its hash."""
    test_file = paths.data_root() / corpus / "splits" / scheme / tag / "test" / "records.jsonl"
    if not test_file.exists():
        raise FrozenSplitViolation(f"cannot freeze missing test partition: {test_file}")
    key = _rel_key(test_file)
    digest = hash_file(test_file, n=64)
    manifest = dict(load_manifest())
    existing = manifest.get(key)
    if existing is not None and existing["sha256"] != digest:
        raise FrozenSplitViolation(
            f"{key} is already frozen with a different hash; frozen test sets are immutable — "
            "a changed test partition needs a NEW tag, never a re-freeze"
        )
    manifest[key] = {
        "sha256": digest,
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_bytes": test_file.stat().st_size,
    }
    _save_manifest(manifest)
    return digest


def frozen_entry(path: str | os.PathLike) -> dict | None:
    key = _rel_key(Path(os.fsdecode(path)))
    if key is None:
        return None
    return load_manifest().get(key)


def check_read(path: str | os.PathLike) -> None:
    """Raise if a frozen file's content no longer matches its manifest hash."""
    entry = frozen_entry(path)
    if entry is None:
        return
    actual = hash_file(path, n=64)
    if actual != entry["sha256"]:
        raise FrozenSplitViolation(
            f"frozen test partition {os.fsdecode(path)} has been modified "
            f"(hash {actual[:12]}… != frozen {entry['sha256'][:12]}…); "
            "restore it or mint a new split tag — never edit a frozen test set"
        )


def check_write(path: str | os.PathLike) -> None:
    if frozen_entry(path) is not None:
        raise FrozenSplitViolation(
            f"refusing to open frozen test partition {os.fsdecode(path)} for writing; "
            "frozen test sets are immutable — mint a new split tag instead"
        )


def verify_all() -> list[str]:
    """Re-hash every frozen file; returns failure messages (empty = all good)."""
    failures = []
    root = paths.data_root()
    for key, entry in sorted(load_manifest().items()):
        target = root / key
        if not target.exists():
            failures.append(f"{key}: frozen file missing under {root}")
        elif hash_file(target, n=64) != entry["sha256"]:
            failures.append(f"{key}: content differs from frozen hash")
    return failures
