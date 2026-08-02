"""Decipherment-ladder cognate data from Luo et al.'s NeuroDecipher release.

github.com/j-luo93/NeuroDecipher ships the exact datasets behind the
published results — Ugaritic-Hebrew cognates (transliterated, no special
symbols) and Linear B-Greek cognates (Unicode Linear B vs Greek, `|` between
Greek variants). Ingesting these makes ladder rungs 1 and 2 runnable in
Phase 5 without any custom scraping; the full DĀMOS/LiBER Linear B corpus
remains a separate gap.

.cog format: TSV, header row names the two sides, one cognate pair per line.
"""

from collections.abc import Iterator

from glyphos.data.ingest import CorpusMeta, require_raw
from glyphos.data.schema import Record

FETCH_HINT = (
    "git clone --depth 1 https://github.com/j-luo93/NeuroDecipher "
    "$GLYPHOS_DATA_ROOT/{corpus}/raw/NeuroDecipher"
)

FILES = {
    "ugaritic_hebrew_cognates": ("uga-heb.no_spe.cog", "uga", "heb"),
    "linearb_greek_cognates": ("linear_b-greek.cog", "linear_b", "greek"),
}

_SOURCE = "https://github.com/j-luo93/NeuroDecipher (Luo, Cao & Barzilay 2019 release)"


def meta_for(corpus: str) -> CorpusMeta:
    filename, src_name, tgt_name = FILES[corpus]
    return CorpusMeta(
        name=corpus,
        kind="cognate_pairs",
        source=_SOURCE,
        license="MIT (repository license; underlying lexica are published scholarship)",
        encoding=(
            "transliteration (both sides)"
            if corpus.startswith("ugaritic")
            else "unicode Linear B glyphs vs Greek"
        ),
        primary_field="src",
        notes=f"file {filename}; fields src={src_name}, tgt={tgt_name}; "
        "'|' separates target variants; splits for the ladder are made in Phase 5 protocol",
        extra={"src_name": src_name, "tgt_name": tgt_name},
    )


def parse_for(corpus: str) -> Iterator[Record]:
    filename, src_name, tgt_name = FILES[corpus]
    # both corpora share one clone; require it under the corpus's own raw dir
    path = require_raw(corpus, f"NeuroDecipher/data/{filename}", FETCH_HINT.format(corpus=corpus))
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if len(header) != 2:
            raise ValueError(f"{path}: expected 2-column TSV header, got {header!r}")
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{i + 2}: expected 2 columns, got {len(parts)}")
            yield Record(
                corpus=corpus,
                doc_id="cognates",
                sent_id=f"{corpus}-p{i:05d}",
                fields={"src": parts[0], "tgt": parts[1]},
                meta={"src_name": src_name, "tgt_name": tgt_name},
            )


def make_module(corpus: str):
    return corpus, (lambda: meta_for(corpus)), (lambda: parse_for(corpus))
