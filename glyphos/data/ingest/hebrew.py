"""Hebrew Bible ingestion (known relative for the Ugaritic rung).

Source: OpenScriptures morphhb (WLC with morphology, CC BY 4.0) — chosen over
the ETCBC/BHSA text-fabric export for parse simplicity and clean licensing
(docs/decisions.md). One record per verse; doc = book. Text is the pointed
WLC surface form (concatenated <w> tokens); consonantal normalization for the
decipherment track happens downstream in align/, not at ingest.
"""

import xml.etree.ElementTree as ET
from collections.abc import Iterator

from glyphos.data.ingest import CorpusMeta, require_raw
from glyphos.data.schema import Record

CORPUS = "hebrew_morphhb"

FETCH_HINT = (
    "git clone --depth 1 https://github.com/openscriptures/morphhb "
    "$GLYPHOS_DATA_ROOT/hebrew_morphhb/raw/morphhb"
)

_OSIS = "{http://www.bibletechnologies.net/2003/OSIS/namespace}"


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="monolingual",
        source="https://github.com/openscriptures/morphhb",
        license="CC BY 4.0",
        encoding="unicode Hebrew (pointed, WLC)",
        primary_field="text",
        notes="one record per verse; doc = book; lemma/morph annotations not ingested in Phase 1",
    )


def parse() -> Iterator[Record]:
    wlc = require_raw(CORPUS, "morphhb/wlc", FETCH_HINT)
    for path in sorted(wlc.glob("*.xml")):
        if path.stem == "VerseMap":
            continue
        root = ET.parse(path).getroot()
        for verse in root.iter(f"{_OSIS}verse"):
            osis_id = verse.get("osisID")
            if not osis_id:
                continue
            words = [w.text.strip() for w in verse.iter(f"{_OSIS}w") if w.text and w.text.strip()]
            if not words:
                continue
            book = osis_id.split(".")[0]
            yield Record(
                corpus=CORPUS,
                doc_id=book,
                sent_id=osis_id,
                fields={"text": " ".join(words)},
                meta={},
            )
