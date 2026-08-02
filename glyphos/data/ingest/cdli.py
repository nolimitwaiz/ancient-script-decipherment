"""CDLI cuneiform ingestion (the Akkadian/Sumerian corpus LogogramNLP drew on).

Source: the CDLI open-data release (github.com/cdli-gh/data, 2023-10 snapshot;
Git-LFS files fetched directly): `cdliatf_unblocked.atf` (135k tablets of ATF
transliteration) + `cdli_cat.csv` (catalogue: language, period, designation).

One record per tablet (the tablet IS the document): transliteration lines are
joined with " / ". Structure lines (@), state lines ($), comments (#) and
link lines (>>) are dropped; sign-level detail stays as-is in ATF notation.
"""

import csv
import re
from collections.abc import Iterator

from glyphos.data.ingest import CorpusMeta, NotIngestable, require_raw
from glyphos.data.schema import Record

CORPUS = "cuneiform_cdli"

FETCH_HINT = (
    "curl -L -o $GLYPHOS_DATA_ROOT/cuneiform_cdli/raw/cdliatf_unblocked.atf "
    "https://media.githubusercontent.com/media/cdli-gh/data/master/cdliatf_unblocked.atf && "
    "curl -L -o $GLYPHOS_DATA_ROOT/cuneiform_cdli/raw/cdli_cat.csv "
    "https://media.githubusercontent.com/media/cdli-gh/data/master/cdli_cat.csv"
)

_TEXT_LINE = re.compile(r"^\d+[a-z']*\.\s+(.*)$")
_PNUM_HEADER = re.compile(r"^&(P\d+)\s*=\s*(.*)$")


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="monolingual",
        source="https://github.com/cdli-gh/data (snapshot 2023-10)",
        license="CDLI open data (CC BY; see cdli.mpiwg-berlin.mpg.de)",
        encoding="ATF transliteration (Sumerian/Akkadian and others; language in meta)",
        primary_field="text",
        notes="one record per tablet; catalogue language/period attached from cdli_cat.csv",
    )


def _catalogue(path) -> dict[str, dict]:
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = {name: i for i, name in enumerate(header)}
        id_col = next((cols[c] for c in ("id_text", "id", "pnum", "p_number") if c in cols), None)
        if id_col is None:
            raise NotIngestable(f"cdli_cat.csv: no tablet-id column among {sorted(cols)[:40]}...")
        lang_col = cols.get("language")
        period_col = cols.get("period")
        out: dict[str, dict] = {}
        for row in reader:
            if len(row) <= id_col or not row[id_col].strip():
                continue
            raw_id = row[id_col].strip()
            pnum = (
                raw_id
                if raw_id.startswith("P")
                else f"P{int(raw_id):06d}"
                if raw_id.isdigit()
                else raw_id
            )
            info = {}
            if lang_col is not None and len(row) > lang_col and row[lang_col].strip():
                info["language"] = row[lang_col].strip()
            if period_col is not None and len(row) > period_col and row[period_col].strip():
                info["period"] = row[period_col].strip()
            out[pnum] = info
    return out


def parse() -> Iterator[Record]:
    atf = require_raw(CORPUS, "cdliatf_unblocked.atf", FETCH_HINT)
    cat_path = require_raw(CORPUS, "cdli_cat.csv", FETCH_HINT)
    catalogue = _catalogue(cat_path)

    pnum = None
    designation = ""
    lines: list[str] = []

    def _emit():
        if pnum is None or not lines:
            return None
        rec_meta = {"designation": designation, **catalogue.get(pnum, {})}
        return Record(
            corpus=CORPUS,
            doc_id=pnum,
            sent_id=pnum,
            fields={"text": " / ".join(lines)},
            meta=rec_meta,
        )

    with open(atf, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            header = _PNUM_HEADER.match(line)
            if header:
                rec = _emit()
                if rec is not None:
                    yield rec
                pnum, designation, lines = header.group(1), header.group(2).strip(), []
                continue
            m = _TEXT_LINE.match(line.strip())
            if m and pnum is not None:
                content = m.group(1).strip()
                if content:
                    lines.append(content)
    rec = _emit()
    if rec is not None:
        yield rec
