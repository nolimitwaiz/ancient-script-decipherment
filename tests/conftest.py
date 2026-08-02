"""Every test runs against isolated state: tmp data root, tmp runs dir (and
therefore tmp ledger + tmp guard log), a cleared run context, and a guaranteed
guard uninstall afterwards so no test can leak a patched open() into another.
"""

import pytest

from glyphos.data import guard
from glyphos.utils import runctx


@pytest.fixture(autouse=True)
def _isolated_state(monkeypatch, tmp_path):
    monkeypatch.setenv("GLYPHOS_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("GLYPHOS_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("GLYPHOS_LEDGER_PATH", raising=False)
    monkeypatch.delenv("GLYPHOS_TEST_ACCESS_LOG", raising=False)
    runctx.clear_current_run()
    yield
    guard.uninstall_guard()
    runctx.clear_current_run()
