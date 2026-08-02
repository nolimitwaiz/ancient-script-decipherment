"""Loud stubs for corpora that are planned but not yet ingestable.

Each raises NotIngestable with the concrete unblock path; census lists them
as planned so the gap is visible in every report (docs/data_gaps.md).
Findings dated 2026-08-02 from a live source survey.
"""

from glyphos.data.ingest import CorpusMeta, NotIngestable

STUB_REASONS = {
    "linear_b_damos": (
        "DAMOS (Oslo) and LiBER (liber.cnr.it) are web databases without bulk export; "
        "LiBER exposes a CNR SPARQL endpoint (data.cnr.it) worth pursuing. Luo et al.'s "
        "linearb_greek_cognates already cover ladder rung 2; see docs/data_gaps.md"
    ),
    "mayan": (
        "no open bulk download found (2026-08): MHD requires an application (CSU Chico), "
        "mayacorpus.org (200k+ glyph blocks, MHD-sourced) is interactive-only, TWKM/Bonn "
        "publishes no public TEI dump — contact the projects; see docs/data_gaps.md"
    ),
    "libyco_berber": (
        "the LBI database is DEFUNCT (project ended after W. Pichler's death; "
        "lbi-project.org domain is now squatted). ~300 panels survive in Wayback "
        "snapshots of institutum-canarium.org/lbi-project (use pre-2020 captures); "
        "recover via archive scrape when needed; see docs/data_gaps.md"
    ),
}

_META = {
    "linear_b_damos": CorpusMeta(
        name="linear_b_damos",
        kind="stub",
        source="DAMOS (Oslo) / LiBER (CNR, liber.cnr.it) — no bulk export",
        license="TBD",
        encoding="unicode Linear B",
        primary_field=None,
        notes=STUB_REASONS["linear_b_damos"],
    ),
    "mayan": CorpusMeta(
        name="mayan",
        kind="stub",
        source="MHD (application) / mayacorpus.org (interactive) / TWKM Bonn (no dump)",
        license="TBD",
        encoding="glyph-block transliterations + images",
        primary_field=None,
        notes=STUB_REASONS["mayan"],
    ),
    "libyco_berber": CorpusMeta(
        name="libyco_berber",
        kind="stub",
        source="LBI database (defunct) via Wayback Machine snapshots",
        license="TBD",
        encoding="transliteration + panel drawings",
        primary_field=None,
        notes=STUB_REASONS["libyco_berber"],
    ),
}


def meta_for(corpus: str) -> CorpusMeta:
    return _META[corpus]


def parse_for(corpus: str):
    raise NotIngestable(f"{corpus} is a stub: {STUB_REASONS[corpus]}")
