"""Meroitic ingestion — first frontier target with real machine-readable data.

Source: github.com/Joshua-Otten/Meroitic-Corpus (Otten & Anastasopoulos,
"Towards Ancient Meroitic Decipherment", ALP 2025) — the first machine-
readable Meroitic corpus, ASCII romanization of REM-derived texts (18k lines)
plus named narrative inscriptions (Hamadab stela of Amanirenas, Kalabsha
inscription of Kharamadoye, Tanyidamani).

SEALED-CONSTRAINT NOTE: the repo also ships pretrained word embeddings under
Embeddings/ — those are trained model parameters and are NEVER read by this
pipeline (only the text files are). License unstated upstream; cite the paper.
"""

from collections.abc import Iterator
from pathlib import Path

from glyphos.data.ingest import CorpusMeta, require_raw
from glyphos.data.schema import Record

CORPUS = "meroitic_rem"

FETCH_HINT = (
    "git clone --depth 1 https://github.com/Joshua-Otten/Meroitic-Corpus "
    "$GLYPHOS_DATA_ROOT/meroitic_rem/raw/Meroitic-Corpus"
)

# document name -> file under Data/
DOC_FILES = {
    "mero-corpus": "mero-corpus.txt",
    "hamadab-stela-amanirenas": "HamadabStelaOfAmanirenasNarrative.txt",
    "kalabsha-kharamadoye": "KalabshaMeroiticInscriptionOfKharamadoyeNarrative.txt",
    "tanyidamani": "TanyidamaniNarrative.txt",
}


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="monolingual",
        source="https://github.com/Joshua-Otten/Meroitic-Corpus (Otten & Anastasopoulos 2025)",
        license="unstated upstream — research use, cite the paper; REM texts are "
        "published scholarship",
        encoding="ASCII romanization of Meroitic (capitals encode special signs)",
        primary_field="text",
        notes="undeciphered-language frontier target; Embeddings/ dir is pretrained and "
        "quarantined (never read); auxiliary vocab/example files kept raw only",
    )


def _lines(path: Path) -> Iterator[str]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def parse() -> Iterator[Record]:
    data = require_raw(CORPUS, "Meroitic-Corpus/Data", FETCH_HINT)
    for doc_id, filename in DOC_FILES.items():
        path = data / filename
        if not path.exists():
            require_raw(CORPUS, f"Meroitic-Corpus/Data/{filename}", FETCH_HINT)
        for i, line in enumerate(_lines(path)):
            yield Record(
                corpus=CORPUS,
                doc_id=doc_id,
                sent_id=f"{doc_id}.{i:05d}",
                fields={"text": line},
                meta={},
            )
