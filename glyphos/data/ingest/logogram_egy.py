"""LogogramNLP Egyptian textlines — REAL hieroglyphic images (§ plan item 3).

Why this corpus matters more than its size suggests: every other pixel input
in this project is `render(unicode_text)`, a deterministic re-encoding that
contains no information the text does not already have — which makes the
pixel-vs-token comparison close to tautological. These are scholarly
hand-copied epigraphic linearts of actual inscriptions, so the image carries
what Unicode discards (drawn sign forms, spacing, layout, damage) and the
token arm presupposes a human already read the artifact.

Source: Chen et al., LogogramNLP (ACL 2024), `data/EGY` — 1,320 textlines,
each with transliteration (`tl`), an English translation (`ta`), a dating
string, and a PNG. Images live in `metadata/textline/` (the metadata's
`thot_img/` prefix is stale; basenames match).

doc_id = dating cohort, the same conservative convention used for TLA: real
text identifiers are not published with this release, and sentences from one
inscription share its dating, so cohort-held-out cannot leak a text across
partitions.
"""

import json
from collections.abc import Iterator
from pathlib import Path

from glyphos.data.ingest import CorpusMeta, NotIngestable, require_raw
from glyphos.data.schema import Record

CORPUS = "logogram_egy"

FETCH_HINT = (
    "git clone --depth 1 https://github.com/taineleau/logogramNLP "
    "$GLYPHOS_DATA_ROOT/logogram_nlp/raw/logogramNLP  # then symlink or copy "
    "data/EGY into $GLYPHOS_DATA_ROOT/logogram_egy/raw/EGY"
)


def meta() -> CorpusMeta:
    return CorpusMeta(
        name=CORPUS,
        kind="parallel",
        source="https://github.com/taineleau/logogramNLP (Chen et al., ACL 2024), data/EGY",
        license="unstated upstream — research use, cite the paper",
        encoding="hieroglyphic lineart PNG + transliteration",
        primary_field="image",
        translation_field="translation_en",
        translation_lang="en",
        notes="REAL scholarly linearts, not font renders; doc_id = dating cohort; "
        "image paths are relative to the corpus raw dir",
    )


def _entries(root: Path) -> list[dict]:
    with open(root / "metadata" / "metadata.json", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise NotIngestable(f"{CORPUS}: metadata.json is not a non-empty list")
    return data


def parse() -> Iterator[Record]:
    root = require_raw(CORPUS, "EGY", FETCH_HINT)
    textline_dir = root / "metadata" / "textline"
    entries = _entries(root)

    cohorts: dict[str, int] = {}
    missing = 0
    emitted = 0
    for i, e in enumerate(entries):
        translit = (e.get("tl") or "").strip()
        english = (e.get("ta") or "").strip()
        image_name = Path(e.get("image", "")).name
        if not (translit and english and image_name):
            continue
        image_path = textline_dir / image_name
        if not image_path.exists():
            missing += 1
            continue
        date = (e.get("date") or "unknown").strip()
        cohort = cohorts.setdefault(date, len(cohorts) + 1)
        yield Record(
            corpus=CORPUS,
            doc_id=f"{CORPUS}-d{cohort:03d}",
            sent_id=f"{CORPUS}-s{i:05d}",
            fields={
                "transliteration": translit,
                "translation_en": english,
                # stored relative to the corpus raw dir so the record stays
                # portable between this Mac and the cluster
                "image": str(image_path.relative_to(root.parent)),
            },
            meta={"date": date},
        )
        emitted += 1

    if emitted == 0:
        raise NotIngestable(f"{CORPUS}: no usable entries (missing images: {missing})")
    if missing:
        print(f"[ingest] {CORPUS}: {missing} entries skipped — image file absent")


def image_root() -> Path:
    """Directory that `fields['image']` paths are relative to."""
    from glyphos.data.ingest import raw_dir

    return raw_dir(CORPUS)
