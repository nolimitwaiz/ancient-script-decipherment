"""Append-only experiment ledger (§ Phase 0.2).

Discipline enforced here, not by convention:
- a run is REGISTERED — with a non-empty hypothesis — before any result exists
  (kill-gate preregistration, § working style 2);
- a second event marks it completed/failed/abandoned; nothing is ever rewritten;
- `n_variants_tried_so_far_in_this_family` is computed at registration time so
  the multiple-testing count cannot be reconstructed selectively after the fact.

File format: one JSON object per line. `event: register` carries the full
run metadata; `event: complete` carries terminal status and all_metrics.
A corrupt line is a hard error — the ledger is the scientific record.
"""

import json
import os
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from glyphos.utils import paths, runctx

SCHEMA_VERSION = 1
TERMINAL_STATUSES = ("completed", "failed", "abandoned")


class LedgerError(RuntimeError):
    pass


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def detect_git_hash(cwd: Path | None = None) -> str:
    """Short commit hash of the repo, '-dirty' suffixed if the tree has changes."""
    cwd = cwd or paths.repo_root()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()[:12]
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        ).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class RunRecord:
    """Merged view of a run: its registration plus (if any) its terminal event."""

    run_id: str
    registered_at: str
    git_hash: str
    hypothesis: str
    phase: str
    family: str
    config_hash: str
    data_version: str
    split_version: str
    seed: int
    selection_metric: str
    n_variants_tried_so_far_in_this_family: int
    status: str = "registered"
    all_metrics: dict = field(default_factory=dict)
    completed_at: str | None = None
    notes: str = ""


@dataclass
class RunHandle:
    """Mutable collector handed to code inside `Ledger.run(...)`."""

    run_id: str
    metrics: dict = field(default_factory=dict)
    status: str = "completed"
    notes: str = ""

    def log_metric(self, name: str, value) -> None:
        self.metrics[name] = value

    def log_metrics(self, metrics: dict) -> None:
        self.metrics.update(metrics)


class Ledger:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else paths.ledger_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # -- storage ------------------------------------------------------------

    def _append(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True, ensure_ascii=False)
        with open(self.path, "a", encoding="utf-8") as f:
            try:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except ImportError:  # non-POSIX; single-writer assumption holds
                pass
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _events(self) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LedgerError(
                        f"{self.path}:{lineno}: corrupt ledger line ({exc}); "
                        "the ledger is append-only — investigate, do not repair silently"
                    ) from exc
        return events

    # -- write API ----------------------------------------------------------

    def register(
        self,
        *,
        hypothesis: str,
        phase: str,
        family: str,
        config_hash: str,
        data_version: str,
        split_version: str,
        seed: int,
        selection_metric: str,
        notes: str = "",
        git_hash: str | None = None,
    ) -> str:
        if not str(hypothesis).strip():
            raise LedgerError(
                "hypothesis is required: preregister what you expect BEFORE running (§ Phase 0.2)"
            )
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise LedgerError(f"seed must be an int, got {seed!r}")
        required = {
            "phase": phase,
            "family": family,
            "config_hash": config_hash,
            "data_version": data_version,
            "split_version": split_version,
            "selection_metric": selection_metric,
        }
        for name, value in required.items():
            if not str(value).strip():
                raise LedgerError(f"{name} is required and must be non-empty")

        n_prior = sum(
            1 for e in self._events() if e.get("event") == "register" and e.get("family") == family
        )
        run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        self._append(
            {
                "event": "register",
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "timestamp": _utcnow(),
                "git_hash": git_hash if git_hash is not None else detect_git_hash(),
                "hypothesis": str(hypothesis).strip(),
                "phase": phase,
                "family": family,
                "config_hash": config_hash,
                "data_version": data_version,
                "split_version": split_version,
                "seed": seed,
                "selection_metric": selection_metric,
                "n_variants_tried_so_far_in_this_family": n_prior + 1,
                "notes": notes,
            }
        )
        runctx.set_current_run(run_id)
        return run_id

    def complete(
        self, run_id: str, status: str, all_metrics: dict | None = None, notes: str = ""
    ) -> None:
        if status not in TERMINAL_STATUSES:
            raise LedgerError(f"status must be one of {TERMINAL_STATUSES}, got {status!r}")
        records = {r.run_id: r for r in self.load()}
        if run_id not in records:
            raise LedgerError(f"unknown run_id {run_id!r}: runs must be registered first")
        if records[run_id].status != "registered":
            raise LedgerError(
                f"run {run_id} already terminal ({records[run_id].status}); ledger is append-only"
            )
        self._append(
            {
                "event": "complete",
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "timestamp": _utcnow(),
                "status": status,
                "all_metrics": all_metrics or {},
                "notes": notes,
            }
        )
        if runctx.get_current_run() == run_id:
            runctx.clear_current_run()

    @contextmanager
    def run(self, **register_kwargs) -> Iterator[RunHandle]:
        """Register, yield a metric collector, and guarantee a terminal event.

        An escaping exception records the run as `failed` (with the exception in
        notes) and re-raises — abandoned/crashed runs land in the ledger too.
        """
        run_id = self.register(**register_kwargs)
        handle = RunHandle(run_id=run_id)
        try:
            yield handle
        except BaseException as exc:
            self.complete(run_id, "failed", handle.metrics, notes=f"{type(exc).__name__}: {exc}")
            raise
        self.complete(run_id, handle.status, handle.metrics, notes=handle.notes)

    # -- read API -----------------------------------------------------------

    def load(self) -> list[RunRecord]:
        records: dict[str, RunRecord] = {}
        for e in self._events():
            kind = e.get("event")
            if kind == "register":
                if e["run_id"] in records:
                    raise LedgerError(f"duplicate register event for run {e['run_id']}")
                records[e["run_id"]] = RunRecord(
                    run_id=e["run_id"],
                    registered_at=e["timestamp"],
                    git_hash=e["git_hash"],
                    hypothesis=e["hypothesis"],
                    phase=e["phase"],
                    family=e["family"],
                    config_hash=e["config_hash"],
                    data_version=e["data_version"],
                    split_version=e["split_version"],
                    seed=e["seed"],
                    selection_metric=e["selection_metric"],
                    n_variants_tried_so_far_in_this_family=e[
                        "n_variants_tried_so_far_in_this_family"
                    ],
                    notes=e.get("notes", ""),
                )
            elif kind == "complete":
                rec = records.get(e["run_id"])
                if rec is None:
                    raise LedgerError(f"complete event for unknown run {e['run_id']}")
                rec.status = e["status"]
                rec.all_metrics = e.get("all_metrics", {})
                rec.completed_at = e["timestamp"]
                if e.get("notes"):
                    rec.notes = f"{rec.notes} | {e['notes']}" if rec.notes else e["notes"]
            else:
                raise LedgerError(f"unknown ledger event type {kind!r}")
        return list(records.values())

    def families(self) -> dict[str, list[RunRecord]]:
        out: dict[str, list[RunRecord]] = {}
        for rec in self.load():
            out.setdefault(rec.family, []).append(rec)
        return out
