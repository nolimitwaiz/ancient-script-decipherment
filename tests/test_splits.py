import pytest

from glyphos.data.schema import Record
from glyphos.data.splits import (
    DEFAULT_RATIOS,
    SplitError,
    dedup_against_train,
    extract_signs,
    levenshtein_within,
    normalize_target,
    period_bucket,
    split_document,
    split_period,
    split_random,
    split_sign_heldout,
    split_version,
)


def _rec(i: int, doc: str, text: str = "", **meta) -> Record:
    return Record(
        corpus="c",
        doc_id=doc,
        sent_id=f"s{i:04d}",
        fields={"text": text or f"sentence number {i} with words"},
        meta=meta,
    )


def _corpus(n_docs: int = 30, per_doc: int = 10) -> list[Record]:
    return [_rec(d * per_doc + s, f"d{d:03d}") for d in range(n_docs) for s in range(per_doc)]


def test_random_split_sizes_and_determinism():
    records = _corpus()
    a = split_random(records, DEFAULT_RATIOS, seed=7)
    b = split_random(records, DEFAULT_RATIOS, seed=7)
    assert {p: len(v) for p, v in a.items()} == {"train": 240, "valid": 30, "test": 30}
    assert [r.sent_id for r in a["test"]] == [r.sent_id for r in b["test"]]
    c = split_random(records, DEFAULT_RATIOS, seed=8)
    assert [r.sent_id for r in a["test"]] != [r.sent_id for r in c["test"]]


def test_ratio_validation():
    with pytest.raises(SplitError, match="sum to 1"):
        split_random(_corpus(), {"train": 0.9, "valid": 0.2, "test": 0.1}, seed=1)
    with pytest.raises(SplitError, match="exactly"):
        split_random(_corpus(), {"train": 1.0}, seed=1)


def test_document_split_never_splits_a_doc():
    splits = split_document(_corpus(), DEFAULT_RATIOS, seed=7)
    docs = {p: {r.doc_id for r in splits[p]} for p in splits}
    assert not (docs["train"] & docs["valid"])
    assert not (docs["train"] & docs["test"])
    assert not (docs["valid"] & docs["test"])
    assert sum(len(v) for v in splits.values()) == 300


def test_document_split_too_few_docs_is_loud():
    with pytest.raises(SplitError, match="empty"):
        split_document(_corpus(n_docs=2), DEFAULT_RATIOS, seed=7)


def test_period_bucket_and_split():
    assert period_bucket(_rec(0, "d", dateNotBefore="-1580")) == "c-16"
    assert period_bucket(_rec(0, "d", dateNotBefore="250")) == "c+2"
    assert period_bucket(_rec(0, "d")) == "unknown"
    records = [_rec(i, f"d{i}", dateNotBefore=str(-2000 + 100 * (i % 8))) for i in range(400)]
    splits = split_period(records, DEFAULT_RATIOS, seed=3)
    buckets = {p: {period_bucket(r) for r in splits[p]} for p in splits}
    assert not (buckets["train"] & buckets["test"])


def test_period_split_needs_enough_buckets():
    records = [_rec(i, f"d{i}", dateNotBefore="-1500") for i in range(50)]
    with pytest.raises(SplitError, match="century buckets"):
        split_period(records, DEFAULT_RATIOS, seed=1)


# -- dedup ------------------------------------------------------------------


def test_normalize_target():
    assert normalize_target("  Es werde‚ zerrieben!  ") == "es werde zerrieben"


def test_levenshtein_within():
    assert levenshtein_within("kitten", "sitten", 1)
    assert not levenshtein_within("kitten", "sitting", 2)
    assert levenshtein_within("kitten", "sitting", 3)
    assert not levenshtein_within("abc", "abcdefgh", 3)


def test_dedup_removes_exact_and_near_duplicates():
    train = [_rec(i, "dtrain", f"unique training sentence {i}") for i in range(20)]
    train.append(_rec(99, "dtrain", "the king made an offering to osiris"))
    splits = {
        "train": train,
        "valid": [
            _rec(200, "dv", "the king made an offering to osiris"),  # exact
            _rec(201, "dv", "a completely different valid sentence"),
        ],
        "test": [
            _rec(300, "dt", "the king made an offering to osiriX"),  # near (1 edit)
            _rec(301, "dt", "entirely unrelated held out content"),
        ],
    }
    deduped, report = dedup_against_train(splits, target_field="text")
    assert report.removed_exact == 1
    assert report.removed_near == 1
    assert [r.sent_id for r in deduped["valid"]] == ["s0201"]
    assert [r.sent_id for r in deduped["test"]] == ["s0301"]
    assert len(deduped["train"]) == len(train)


def test_dedup_keeps_distant_text():
    splits = {
        "train": [_rec(i, "d0", f"train sentence {i}") for i in range(10)],
        "valid": [_rec(50, "d1", "nothing like the training data at all")],
        "test": [_rec(51, "d2", "also totally novel material here")],
    }
    deduped, report = dedup_against_train(splits, target_field="text")
    assert report.removed == 0
    assert len(deduped["valid"]) == 1 and len(deduped["test"]) == 1


# -- signs ------------------------------------------------------------------


def test_extract_signs_unicode_and_gtags():
    text = "\U00013000\U00013001 x <g>Ff101</g> \U00013000"
    assert extract_signs(text) == {"\U00013000", "\U00013001", "Ff101"}


def test_sign_heldout_semantics():
    # 40 docs x 8 sents; each sentence uses 3 signs from a 30-sign inventory
    signs = [chr(0x13000 + i) for i in range(30)]
    records = []
    for d in range(40):
        for s in range(8):
            i = d * 8 + s
            used = "".join(signs[(i + k) % 30] for k in range(3))
            records.append(Record("c", f"d{d:03d}", f"s{i:04d}", {"hiero": used, "text": "x"}, {}))
    result = split_sign_heldout(
        records, "hiero", DEFAULT_RATIOS, seed=5, n_signs=3, freq_band=(0.0, 1.0)
    )
    held = set(result.held_out_signs)
    assert len(held) == 3
    for part in ("train", "valid"):
        for rec in result.splits[part]:
            assert not (extract_signs(rec.fields["hiero"]) & held)
    for rec in result.splits["test"]:
        assert extract_signs(rec.fields["hiero"]) & held


def test_sign_heldout_needs_signs():
    records = [_rec(i, "d0", "no hieroglyphs here") for i in range(10)]
    with pytest.raises(SplitError, match="no signs"):
        split_sign_heldout(records, "text", DEFAULT_RATIOS, seed=1)


def test_split_version_depends_on_membership():
    records = _corpus()
    a = split_random(records, DEFAULT_RATIOS, seed=7)
    b = split_random(records, DEFAULT_RATIOS, seed=8)
    assert split_version(a) == split_version(a)
    assert split_version(a) != split_version(b)
