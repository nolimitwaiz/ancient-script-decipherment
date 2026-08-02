"""First1KGreek ingestion (known relative for the Linear B rung; sealed Greek LM).

Source: github.com/OpenGreekAndLatin/First1KGreek (CC-BY-SA-4.0), ~1 GB of
EpiDoc TEI XML under data/<textgroup>/<work>/<file>.xml. Only `-grc` edition
files are ingested (the repo also carries Latin/English material). Records are
<p> and <l> elements from the TEI body that contain Greek script; doc = one
work (textgroup.work).
"""

import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from glyphos.data.ingest import CorpusMeta, NotIngestable, require_raw
from glyphos.data.schema import Record

CORPUS = "greek_first1k"

FETCH_HINT = (
    "git clone --depth 1 https://github.com/OpenGreekAndLatin/First1KGreek "
    "$GLYPHOS_DATA_ROOT/greek_first1k/raw/First1KGreek"
)

_TEI = "{http://www.tei-c.org/ns/1.0}"
_MIN_TOKENS = 3


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="monolingual",
        source="https://github.com/OpenGreekAndLatin/First1KGreek",
        license="CC-BY-SA-4.0",
        encoding="unicode polytonic Greek (EpiDoc TEI)",
        primary_field="text",
        notes="grc editions only; records = TEI <p>/<l> units containing Greek script; "
        "doc = textgroup.work",
    )


def _has_greek(text: str) -> bool:
    return any(0x0370 <= ord(ch) <= 0x03FF or 0x1F00 <= ord(ch) <= 0x1FFF for ch in text)


def _units(body) -> Iterator[str]:
    for el in body.iter():
        if el.tag in (f"{_TEI}p", f"{_TEI}l"):
            text = unicodedata.normalize("NFC", " ".join(" ".join(el.itertext()).split()))
            if len(text.split()) >= _MIN_TOKENS and _has_greek(text):
                yield text


def parse() -> Iterator[Record]:
    data = require_raw(CORPUS, "First1KGreek/data", FETCH_HINT)
    files = sorted(p for p in data.rglob("*.xml") if "grc" in p.stem)
    if not files:
        raise NotIngestable(f"no -grc TEI files under {data}; incomplete clone?")
    failures = []
    for path in files:
        rel = path.relative_to(data).parts
        doc_id = f"{rel[0]}.{rel[1]}" if len(rel) >= 3 else path.stem
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            failures.append(f"{path.relative_to(data)}: {exc}")
            continue
        body = root.find(f".//{_TEI}body")
        if body is None:
            continue
        for i, text in enumerate(_units(body)):
            yield Record(
                corpus=CORPUS,
                doc_id=doc_id,
                sent_id=f"{doc_id}.{path.stem}.{i:05d}",
                fields={"text": text},
                meta={},
            )
    if len(failures) > len(files) * 0.01:
        raise NotIngestable(
            f"{len(failures)}/{len(files)} TEI files failed to parse — investigate:\n  "
            + "\n  ".join(failures[:10])
        )
    if failures:
        print(f"[ingest] {CORPUS}: {len(failures)} malformed TEI file(s) skipped: {failures}")
