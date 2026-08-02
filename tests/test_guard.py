import builtins
import io
from pathlib import Path

from glyphos.data import guard
from glyphos.utils import paths, runctx


def _make(path: Path, content: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_is_test_path_under_data_root():
    root = paths.data_root()
    assert guard.is_test_path(root / "toy/splits/document_heldout/v1/test/sentences.jsonl")
    assert guard.is_test_path(root / "toy/splits/random/v1/test.jsonl")
    assert not guard.is_test_path(root / "toy/splits/document_heldout/v1/train/sentences.jsonl")
    assert not guard.is_test_path(root / "toy/private/key.json")


def test_is_test_path_generic_data_fallback(tmp_path):
    outside = tmp_path / "elsewhere" / "data" / "corpus" / "test" / "f.jsonl"
    assert guard.is_test_path(outside)
    assert not guard.is_test_path(tmp_path / "elsewhere" / "nodata" / "test" / "f.jsonl")


def test_guard_logs_test_reads_with_run_id():
    test_file = _make(paths.data_root() / "toy/splits/s/v1/test/sentences.jsonl")
    train_file = _make(paths.data_root() / "toy/splits/s/v1/train/sentences.jsonl")
    runctx.set_current_run("run-xyz")
    guard.install_guard()

    open(test_file).close()
    open(train_file).close()

    log = guard.read_access_log()
    assert len(log) == 1
    entry = log[0]
    assert entry["run_id"] == "run-xyz"
    assert entry["path"].endswith("test/sentences.jsonl")
    assert "test_guard.py" in entry["caller"]


def test_guard_covers_pathlib_open():
    test_file = _make(paths.data_root() / "c/splits/s/v1/test/f.jsonl")
    guard.install_guard()
    with Path(test_file).open() as f:
        f.read()
    assert len(guard.read_access_log()) == 1


def test_guard_ignores_writes():
    guard.install_guard()
    target = paths.data_root() / "c/splits/s/v1/test/new.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        f.write("x")
    assert guard.read_access_log() == []


def test_guard_audits_null_run_id_when_no_run_active():
    test_file = _make(paths.data_root() / "c/splits/s/v1/test/f.jsonl")
    guard.install_guard()
    open(test_file).close()
    assert guard.read_access_log()[0]["run_id"] is None


def test_install_uninstall_restores_originals():
    original_open = builtins.open
    guard.install_guard()
    assert builtins.open is guard.guarded_open
    assert io.open is guard.guarded_open
    guard.install_guard()  # idempotent
    guard.uninstall_guard()
    assert builtins.open is original_open
    assert io.open is original_open
    assert not guard.guard_installed()


def test_guarded_open_works_without_install():
    test_file = _make(paths.data_root() / "c/splits/s/v1/test/f.jsonl", "hello\n")
    with guard.guarded_open(test_file) as f:
        assert f.read() == "hello\n"
    assert len(guard.read_access_log()) == 1
