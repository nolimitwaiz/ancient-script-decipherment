"""TLA (Thesaurus Linguae Aegyptiae) ingestion — the primary corpus.

Three official premium exports on the HuggingFace org
`thesaurus-linguae-aegyptiae` (enumerated 2026-08-01; datasets, not models —
allowed under the hard constraint): Earlier Egyptian (12,773 sents), Late
Egyptian (3,606), Demotic (13,383, no hieroglyphs column). Fields per row:
hieroglyphs? / transliteration / lemmatization / UPOS / glossing / German
translation / dateNotBefore / dateNotAfter (/ authors for Demotic).

PSEUDO-DOCUMENTS: the exports carry NO text IDs, and empirically they are
NOT contiguously text-ordered (contiguous date runs give ~1 sentence per
run). Instead, doc_id is the DATING COHORT: all sentences sharing the exact
(dateNotBefore, dateNotAfter, authors) triple, non-contiguously. A real
text's sentences always share its dating and editors, so no text can
straddle partitions under cohort-held-out splitting — cohorts merge many
texts (95/34/130 cohorts for the three corpora), which is the conservative,
stricter-than-document direction. Real text IDs remain an open gap
(docs/data_gaps.md).
"""

import os
from collections.abc import Iterator

from glyphos.data.ingest import CorpusMeta, NotIngestable
from glyphos.data.schema import Record
from glyphos.utils import paths

HF_ORG = "thesaurus-linguae-aegyptiae"

REPOS = {
    "tla_earlier_egyptian": "tla-Earlier_Egyptian_original-v18-premium",
    "tla_late_egyptian": "tla-late_egyptian-v19-premium",
    "tla_demotic": "tla-demotic-v18-premium",
}

_BASE_NOTES = (
    "pseudo-documents = non-contiguous dating cohorts (dateNotBefore, dateNotAfter, authors); "
    "signs outside Unicode appear as <g>GardinerCode</g> markup"
)


def _load_rows(corpus: str) -> list[dict]:
    os.environ.setdefault("HF_HOME", str(paths.data_root() / "_hf_cache"))
    import datasets  # deferred: heavy import, only needed at ingest time

    repo = f"{HF_ORG}/{REPOS[corpus]}"
    try:
        ds = datasets.load_dataset(repo, split="train")
    except Exception as exc:
        raise NotIngestable(
            f"could not load {repo} (network/HF outage?): {exc}\n"
            f"Retry with: uv run python scripts/prepare_data.py ingest --corpus {corpus}"
        ) from exc
    return list(ds)


def meta_for(corpus: str) -> CorpusMeta:
    has_hiero = corpus != "tla_demotic"
    return CorpusMeta(
        name=corpus,
        kind="parallel",
        source=f"https://huggingface.co/datasets/{HF_ORG}/{REPOS[corpus]}",
        license="CC-BY-SA-4.0",
        encoding="unicode hieroglyphs + transliteration" if has_hiero else "transliteration only",
        primary_field="hieroglyphs" if has_hiero else "transliteration",
        translation_field="translation_de",
        translation_lang="de",
        notes=_BASE_NOTES,
    )


def parse_for(corpus: str) -> Iterator[Record]:
    rows = _load_rows(corpus)
    cohorts: dict[tuple, int] = {}
    for i, row in enumerate(rows):
        key = (row.get("dateNotBefore", ""), row.get("dateNotAfter", ""), row.get("authors", ""))
        doc_num = cohorts.setdefault(key, len(cohorts) + 1)
        fields = {
            "transliteration": row["transliteration"],
            "lemmatization": row["lemmatization"],
            "upos": row["UPOS"],
            "glossing": row["glossing"],
            "translation_de": row["translation"],
        }
        if "hieroglyphs" in row:
            fields["hieroglyphs"] = row["hieroglyphs"]
        meta = {"dateNotBefore": row.get("dateNotBefore"), "dateNotAfter": row.get("dateNotAfter")}
        if "authors" in row:
            meta["authors"] = row["authors"]
        yield Record(
            corpus=corpus,
            doc_id=f"{corpus}-d{doc_num:05d}",
            sent_id=f"{corpus}-s{i:06d}",
            fields=fields,
            meta=meta,
        )


def make_module(corpus: str):
    """Per-corpus (CORPUS, meta, parse) triple for the registry."""
    return corpus, (lambda: meta_for(corpus)), (lambda: parse_for(corpus))
