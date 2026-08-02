"""Coptic SCRIPTORIUM ingestion (known relative for the Egyptian rung).

Source: github.com/CopticScriptorium/corpora, sparse checkout of the
*_CONLLU/ directories (1.7k documents). We read only the CoNLL-U comment
headers (# newdoc id / # sent_id / # text / # text_en) — tokens are not
needed for Phase 1. Dialect is inferred from the top-level corpus directory
name (bohairic-* → bohairic, else sahidic — the collection's stated default).
"""

from collections.abc import Iterator

from glyphos.data.ingest import CorpusMeta, require_raw
from glyphos.data.schema import Record

CORPUS = "coptic_scriptorium"

FETCH_HINT = (
    "git clone --depth 1 --filter=blob:none --sparse "
    "https://github.com/CopticScriptorium/corpora "
    "$GLYPHOS_DATA_ROOT/coptic_scriptorium/raw/corpora && "
    "cd $GLYPHOS_DATA_ROOT/coptic_scriptorium/raw/corpora && "
    "git sparse-checkout set --no-cone '/README*' '**/*_CONLLU/**'"
)


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="monolingual",
        source="https://github.com/CopticScriptorium/corpora",
        license="CC-BY 4.0 (per corpus READMEs)",
        encoding="unicode Coptic",
        primary_field="text",
        translation_field="translation_en",
        translation_lang="en (partial coverage)",
        notes="dialect inferred from directory name; token annotations not ingested in Phase 1",
    )


def parse() -> Iterator[Record]:
    root = require_raw(CORPUS, "corpora", FETCH_HINT)
    conllu_files = sorted(root.rglob("*.conllu"))
    if not conllu_files:
        require_raw(CORPUS, "corpora/DOES_NOT_EXIST", FETCH_HINT)  # loud, with hint
    sent_counter = 0
    for path in conllu_files:
        subcorpus = path.relative_to(root).parts[0]
        dialect = "bohairic" if "bohairic" in subcorpus.lower() else "sahidic"
        doc_id = f"{subcorpus}/{path.stem}"
        header: dict[str, str] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("# newdoc id = "):
                    doc_id = line.removeprefix("# newdoc id = ").strip() or doc_id
                elif line.startswith("# sent_id = "):
                    header = {"sent_id": line.removeprefix("# sent_id = ").strip()}
                elif line.startswith("# text_en = "):
                    header["text_en"] = line.removeprefix("# text_en = ").strip()
                elif line.startswith("# text = "):
                    header["text"] = line.removeprefix("# text = ").strip()
                elif not line.strip() and header.get("text"):
                    fields = {"text": header["text"]}
                    if header.get("text_en"):
                        fields["translation_en"] = header["text_en"]
                    yield Record(
                        corpus=CORPUS,
                        doc_id=doc_id,
                        sent_id=header.get("sent_id") or f"{CORPUS}-s{sent_counter:07d}",
                        fields=fields,
                        meta={"subcorpus": subcorpus, "dialect": dialect},
                    )
                    sent_counter += 1
                    header = {}
        if header.get("text"):  # file ended without trailing blank line
            fields = {"text": header["text"]}
            if header.get("text_en"):
                fields["translation_en"] = header["text_en"]
            yield Record(
                corpus=CORPUS,
                doc_id=doc_id,
                sent_id=header.get("sent_id") or f"{CORPUS}-s{sent_counter:07d}",
                fields=fields,
                meta={"subcorpus": subcorpus, "dialect": dialect},
            )
            sent_counter += 1
