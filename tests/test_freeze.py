import pytest

from glyphos.data import freeze, guard
from glyphos.data.schema import Record
from glyphos.utils import paths


def _splits():
    def rec(i, doc):
        return Record("toycorp", doc, f"s{i:03d}", {"text": f"sentence {i}"}, {})

    return {
        "train": [rec(i, f"d{i % 4}") for i in range(8)],
        "valid": [rec(10 + i, "d8") for i in range(2)],
        "test": [rec(20 + i, "d9") for i in range(2)],
    }


def test_write_split_layout_and_version():
    version, out_dir = freeze.write_split(_splits(), "toycorp", "document_heldout")
    assert (out_dir / "test" / "records.jsonl").exists()
    assert out_dir == paths.data_root() / "toycorp" / "splits" / "document_heldout" / "v1"
    version2, _ = freeze.write_split(_splits(), "toycorp", "document_heldout")
    assert version == version2


def test_freeze_and_verify_roundtrip():
    freeze.write_split(_splits(), "toycorp", "document_heldout")
    digest = freeze.freeze_split("toycorp", "document_heldout")
    assert len(digest) == 64
    assert freeze.verify_all() == []
    # re-freezing identical content is idempotent
    assert freeze.freeze_split("toycorp", "document_heldout") == digest


def test_refreeze_with_changed_content_rejected():
    freeze.write_split(_splits(), "toycorp", "document_heldout")
    freeze.freeze_split("toycorp", "document_heldout")
    test_file = paths.data_root() / "toycorp/splits/document_heldout/v1/test/records.jsonl"
    test_file.write_text("tampered\n")
    with pytest.raises(freeze.FrozenSplitViolation, match="immutable"):
        freeze.freeze_split("toycorp", "document_heldout")


def test_verify_all_detects_tamper_and_missing():
    freeze.write_split(_splits(), "toycorp", "document_heldout")
    freeze.freeze_split("toycorp", "document_heldout")
    test_file = paths.data_root() / "toycorp/splits/document_heldout/v1/test/records.jsonl"
    test_file.write_text("tampered\n")
    failures = freeze.verify_all()
    assert len(failures) == 1 and "differs" in failures[0]
    test_file.unlink()
    failures = freeze.verify_all()
    assert len(failures) == 1 and "missing" in failures[0]


def test_guard_blocks_reads_of_tampered_frozen_file():
    freeze.write_split(_splits(), "toycorp", "document_heldout")
    freeze.freeze_split("toycorp", "document_heldout")
    test_file = paths.data_root() / "toycorp/splits/document_heldout/v1/test/records.jsonl"
    guard.install_guard()
    with open(test_file) as f:  # intact: read allowed (and audited)
        f.read()
    guard.uninstall_guard()
    test_file.write_text("tampered\n")
    guard.install_guard()
    with pytest.raises(freeze.FrozenSplitViolation, match="modified"):
        open(test_file)  # noqa: SIM115 — asserting the open itself raises


def test_guard_blocks_writes_to_frozen_file():
    freeze.write_split(_splits(), "toycorp", "document_heldout")
    freeze.freeze_split("toycorp", "document_heldout")
    test_file = paths.data_root() / "toycorp/splits/document_heldout/v1/test/records.jsonl"
    guard.install_guard()
    with pytest.raises(freeze.FrozenSplitViolation, match="refusing"):
        open(test_file, "w")  # noqa: SIM115 — asserting the open itself raises
    # unfrozen sibling (train) stays writable
    train_file = test_file.parent.parent / "train" / "records.jsonl"
    with open(train_file, "a") as f:
        f.write("")


def test_unfrozen_test_files_still_only_logged():
    freeze.write_split(_splits(), "toycorp", "document_heldout")  # no freeze call
    test_file = paths.data_root() / "toycorp/splits/document_heldout/v1/test/records.jsonl"
    guard.install_guard()
    with open(test_file) as f:
        f.read()
    assert len(guard.read_access_log()) == 1
