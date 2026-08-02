"""Process-global current-run registry.

One training/eval process corresponds to at most one ledger run at a time; the
test-set access guard stamps every audited read with this run id.
"""

_current_run_id: str | None = None


def set_current_run(run_id: str) -> None:
    global _current_run_id
    _current_run_id = run_id


def get_current_run() -> str | None:
    return _current_run_id


def clear_current_run() -> None:
    global _current_run_id
    _current_run_id = None
