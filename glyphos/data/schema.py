"""Normalized corpus record schema (Phase 1).

Every ingested corpus becomes one `records.jsonl` of Record objects plus a
`manifest.json` carrying provenance and the content hash that serves as the
corpus's `data_version`. Field names inside `fields` are corpus-specific
(e.g. hieroglyphs/transliteration/translation_de for TLA, text for
monolingual corpora); `primary` names the field used for token counts and
rendering defaults.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Record:
    corpus: str
    doc_id: str
    sent_id: str
    fields: dict  # name -> text
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "corpus": self.corpus,
                "doc_id": self.doc_id,
                "sent_id": self.sent_id,
                "fields": self.fields,
                "meta": self.meta,
            },
            sort_keys=True,
            ensure_ascii=False,
        )

    @staticmethod
    def from_json(line: str) -> "Record":
        d = json.loads(line)
        return Record(
            corpus=d["corpus"],
            doc_id=d["doc_id"],
            sent_id=d["sent_id"],
            fields=d["fields"],
            meta=d.get("meta", {}),
        )


def write_records(records: Iterable[Record], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_json() + "\n")
            n += 1
    return n


def read_records(path: Path) -> Iterator[Record]:
    """Read via plain open() so the test-set guard audits reads of test files."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record.from_json(line)
