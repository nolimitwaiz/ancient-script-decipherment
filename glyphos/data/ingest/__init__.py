"""Corpus ingestion (Phase 1): raw sources -> normalized records + manifest.

Each corpus module exposes:
    CORPUS: str                     canonical corpus name
    meta() -> CorpusMeta            provenance/license/encoding facts
    parse() -> Iterator[Record]     normalized records ("inventory" corpora
                                    return no records and fill manifest_extra)

Raw material must already exist under <data_root>/<corpus>/raw/ (downloads are
explicit, logged steps — see scripts/prepare_data.py); a missing raw tree is a
loud error carrying the exact command that fetches it.
"""

from dataclasses import dataclass, field
from pathlib import Path

from glyphos.utils import paths


class NotIngestable(RuntimeError):
    """Raised by stub corpora and missing raw trees; message says how to unblock."""


@dataclass(frozen=True)
class CorpusMeta:
    name: str
    kind: str  # parallel | monolingual | cognate_pairs | inventory | stub
    source: str
    license: str
    encoding: str
    primary_field: str | None
    translation_field: str | None = None
    translation_lang: str | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)


def corpus_dir(corpus: str) -> Path:
    return paths.data_root() / corpus


def raw_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / "raw"


def processed_dir(corpus: str) -> Path:
    return corpus_dir(corpus) / "processed"


def require_raw(corpus: str, subpath: str, fetch_hint: str) -> Path:
    target = raw_dir(corpus) / subpath
    if not target.exists():
        raise NotIngestable(
            f"raw data missing for {corpus!r}: {target}\nFetch it with:\n  {fetch_hint}"
        )
    return target
