"""Loud stubs for corpora that are planned but not yet ingestable.

Each raises NotIngestable with the concrete unblock path; census lists them
as planned so the gap is visible in every report (docs/data_gaps.md).
"""

from glyphos.data.ingest import CorpusMeta, NotIngestable

STUB_REASONS = {
    "greek_first1k": (
        "First1KGreek (~1 GB TEI XML) is deliberately deferred to a cluster-side "
        "ingest before Phase 3 (sealed Greek LM); see docs/data_gaps.md"
    ),
    "linear_b_damos": (
        "DAMOS/LiBER access method still unknown — loader interface reserved; "
        "Luo et al.'s Linear B-Greek cognate pairs ARE available as "
        "linearb_greek_cognates; see docs/data_gaps.md"
    ),
    "meroitic_rem": "frontier target (REM), not a Phase 1 blocker",
    "mayan": "frontier target, not a Phase 1 blocker",
    "libyco_berber": "frontier target, not a Phase 1 blocker",
}

_META = {
    "greek_first1k": CorpusMeta(
        name="greek_first1k",
        kind="stub",
        source="https://github.com/OpenGreekAndLatin/First1KGreek",
        license="CC-BY-SA-4.0",
        encoding="unicode polytonic Greek (TEI XML)",
        primary_field=None,
        notes=STUB_REASONS["greek_first1k"],
    ),
    "linear_b_damos": CorpusMeta(
        name="linear_b_damos",
        kind="stub",
        source="DAMOS (Oslo) / LiBER (CNR) — access TBD",
        license="TBD",
        encoding="unicode Linear B",
        primary_field=None,
        notes=STUB_REASONS["linear_b_damos"],
    ),
    "meroitic_rem": CorpusMeta(
        name="meroitic_rem",
        kind="stub",
        source="Répertoire d'Épigraphie Méroïtique — access TBD",
        license="TBD",
        encoding="transliteration",
        primary_field=None,
        notes=STUB_REASONS["meroitic_rem"],
    ),
    "mayan": CorpusMeta(
        name="mayan",
        kind="stub",
        source="TBD",
        license="TBD",
        encoding="TBD",
        primary_field=None,
        notes=STUB_REASONS["mayan"],
    ),
    "libyco_berber": CorpusMeta(
        name="libyco_berber",
        kind="stub",
        source="TBD",
        license="TBD",
        encoding="TBD",
        primary_field=None,
        notes=STUB_REASONS["libyco_berber"],
    ),
}


def meta_for(corpus: str) -> CorpusMeta:
    return _META[corpus]


def parse_for(corpus: str):
    raise NotIngestable(f"{corpus} is a stub: {STUB_REASONS[corpus]}")
