"""Corpus registry: every corpus the pipeline knows, ready or not.

Applicable split schemes are declared here so `prepare_data.py split --all`
can never silently skip a scheme a corpus should have (and never fabricate
one it can't).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from glyphos.data.ingest import (
    CorpusMeta,
    cdli,
    cognates,
    coptic,
    greek,
    hebrew,
    logogram,
    logogram_egy,
    meroitic,
    stubs,
    tla,
)
from glyphos.data.schema import Record


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    meta: Callable[[], CorpusMeta]
    parse: Callable[[], Iterator[Record]] | None  # None: inventory/stub corpora
    inventory: Callable[[], dict] | None = None
    schemes: tuple[str, ...] = field(default=())


def _tla_spec(corpus: str, with_signs: bool, with_period: bool = True) -> CorpusSpec:
    _, meta_fn, parse_fn = tla.make_module(corpus)
    schemes = ["random", "document_heldout", "dedup"]
    if with_period:  # late_egyptian: only 8 century buckets — period holdout not meaningful
        schemes.append("period_heldout")
    if with_signs:
        schemes.append("sign_heldout")
    return CorpusSpec(name=corpus, meta=meta_fn, parse=parse_fn, schemes=tuple(schemes))


def _cognate_spec(corpus: str) -> CorpusSpec:
    _, meta_fn, parse_fn = cognates.make_module(corpus)
    # cognate splitting is part of the Phase 5 artificial-decipherment protocol
    return CorpusSpec(name=corpus, meta=meta_fn, parse=parse_fn, schemes=())


def _stub_spec(corpus: str) -> CorpusSpec:
    return CorpusSpec(
        name=corpus,
        meta=lambda: stubs.meta_for(corpus),
        parse=lambda: stubs.parse_for(corpus),
        schemes=(),
    )


CORPORA: dict[str, CorpusSpec] = {
    spec.name: spec
    for spec in [
        _tla_spec("tla_earlier_egyptian", with_signs=True),
        _tla_spec("tla_late_egyptian", with_signs=True, with_period=False),
        _tla_spec("tla_demotic", with_signs=False),
        CorpusSpec(
            name=coptic.CORPUS,
            meta=coptic.meta,
            parse=coptic.parse,
            schemes=("random", "document_heldout", "dedup"),
        ),
        CorpusSpec(
            name=hebrew.CORPUS,
            meta=hebrew.meta,
            parse=hebrew.parse,
            schemes=("random", "document_heldout", "dedup"),
        ),
        _cognate_spec("ugaritic_hebrew_cognates"),
        _cognate_spec("linearb_greek_cognates"),
        CorpusSpec(
            name=logogram.CORPUS,
            meta=logogram.meta,
            parse=None,
            inventory=logogram.inventory,
            schemes=(),
        ),
        CorpusSpec(
            name=logogram_egy.CORPUS,
            meta=logogram_egy.meta,
            parse=logogram_egy.parse,
            schemes=("random", "document_heldout", "dedup"),
        ),
        CorpusSpec(
            name=greek.CORPUS,
            meta=greek.meta,
            parse=greek.parse,
            schemes=("random", "document_heldout", "dedup"),
        ),
        CorpusSpec(
            name=cdli.CORPUS,
            meta=cdli.meta,
            parse=cdli.parse,
            schemes=("random", "document_heldout", "dedup", "period_heldout"),
        ),
        CorpusSpec(
            name=meroitic.CORPUS,
            meta=meroitic.meta,
            parse=meroitic.parse,
            # decipherment-target corpus: splits are made by the Phase 5 protocol
            schemes=(),
        ),
        _stub_spec("linear_b_damos"),
        _stub_spec("mayan"),
        _stub_spec("libyco_berber"),
    ]
}


def ready_corpora() -> list[str]:
    return [name for name, spec in CORPORA.items() if spec.meta().kind != "stub"]
