"""LogogramNLP (ACL 2024) — the published baseline we later compare against.

Phase 1 ingests this at INVENTORY level only: the release mixes images,
serialized metadata (.pth pickles — not opened here), and per-task files
whose formats matter in Phase 4/7 when we build the comparison hooks. The
inventory records per-script file counts and bytes so the census is honest
about what exists. Their cuneiform/Akkadian data is NOT in the repo checkout
(docs/data_gaps.md).
"""

from collections import Counter

from glyphos.data.ingest import CorpusMeta, require_raw

CORPUS = "logogram_nlp"

FETCH_HINT = (
    "git clone --depth 1 https://github.com/taineleau/logogramNLP "
    "$GLYPHOS_DATA_ROOT/logogram_nlp/raw/logogramNLP"
)


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="inventory",
        source="https://github.com/taineleau/logogramNLP",
        license="unstated in repo (contact authors before redistribution)",
        encoding="mixed: images + per-task text/metadata",
        primary_field=None,
        notes="inventory-only in Phase 1; task parsing lands with the Phase 7 comparison hooks; "
        ".pth pickles deliberately not opened",
    )


def inventory() -> dict:
    data = require_raw(CORPUS, "logogramNLP/data", FETCH_HINT)
    out: dict[str, dict] = {}
    for script_dir in sorted(p for p in data.iterdir() if p.is_dir()):
        exts: Counter[str] = Counter()
        total_bytes = 0
        n_files = 0
        for f in script_dir.rglob("*"):
            if f.is_file() and f.name != ".DS_Store":
                exts[f.suffix.lower() or "<none>"] += 1
                total_bytes += f.stat().st_size
                n_files += 1
        out[script_dir.name] = {
            "files": n_files,
            "bytes": total_bytes,
            "by_extension": dict(exts.most_common()),
        }
    return out
