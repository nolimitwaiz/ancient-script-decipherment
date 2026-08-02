"""Locked-test-set access guard (§ Phase 0.4).

Every read of a path under data/**/test/ is appended to an audit log stamped
with the active ledger run_id. Phase 0 logs (it does not block); split freezing
in Phase 1 adds enforcement on top of this trail.

`install_guard()` patches both `builtins.open` and `io.open` (pathlib's
`Path.open` resolves through `io.open`, so both routes are covered). The audit
writer uses `os.open`/`os.write` directly so it can never recurse through the
patched functions. Entry points call `install_guard()` first thing; library
code that touches corpora directly should use `guarded_open` regardless, so it
stays audited even in unpatched interpreters (e.g. notebooks).
"""

import builtins
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from glyphos.utils import paths, runctx

_ORIG_BUILTIN_OPEN = builtins.open
_ORIG_IO_OPEN = io.open
_installed = False

_GUARD_FILE = str(Path(__file__).resolve())


def is_test_path(file: str | os.PathLike, data_root: Path | None = None) -> bool:
    """True for any file under the data root (or any generic .../data/... path)
    that sits below a `test` directory or is itself named `test.*`."""
    try:
        name = os.fsdecode(file)
    except TypeError:  # file-like or int fd — nothing to audit
        return False
    p = Path(name).expanduser()
    try:
        p = p.resolve(strict=False)
    except OSError:
        p = p.absolute()

    root = (data_root or paths.data_root()).resolve()
    try:
        rel_parts = p.relative_to(root).parts
    except ValueError:
        parts = p.parts
        if "data" not in parts:
            return False
        rel_parts = parts[parts.index("data") + 1 :]
    if not rel_parts:
        return False
    *dirs, basename = rel_parts
    return "test" in dirs or basename.startswith("test.")


def _caller() -> str:
    frame = sys._getframe()
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename != _GUARD_FILE and not filename.startswith("<"):
            return f"{filename}:{frame.f_lineno}"
        frame = frame.f_back
    return "unknown"


def _log_access(resolved: str, mode: str) -> None:
    record = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": runctx.get_current_run(),
        "path": resolved,
        "mode": mode,
        "pid": os.getpid(),
        "caller": _caller(),
    }
    log_path = paths.test_access_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _is_read_mode(mode: str) -> bool:
    return "r" in mode or "+" in mode


def audit(file, mode: str = "r") -> None:
    if isinstance(file, int):  # already-open fd — no path to classify
        return
    if _is_read_mode(mode) and is_test_path(file):
        _log_access(str(Path(os.fsdecode(file)).expanduser().resolve(strict=False)), mode)


def guarded_open(file, mode: str = "r", *args, **kwargs):
    audit(file, mode)
    return _ORIG_BUILTIN_OPEN(file, mode, *args, **kwargs)


def install_guard() -> None:
    global _installed
    if _installed:
        return
    builtins.open = guarded_open
    io.open = guarded_open
    _installed = True


def uninstall_guard() -> None:
    global _installed
    builtins.open = _ORIG_BUILTIN_OPEN
    io.open = _ORIG_IO_OPEN
    _installed = False


def guard_installed() -> bool:
    return _installed


def read_access_log() -> list[dict]:
    log_path = paths.test_access_log_path()
    if not log_path.exists():
        return []
    with _ORIG_BUILTIN_OPEN(log_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
