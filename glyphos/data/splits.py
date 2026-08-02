"""Split schemes (§ Phase 1.3) — where the 29-47 BLEU contamination failure
mode gets designed out.

Schemes:
- random            weak diagnostic ONLY, never for headline numbers
- document_heldout  no doc_id crosses partitions (headline minimum, with dedup)
- period_heldout    whole period buckets held out (grouped by metadata)
- site_heldout      same engine, keyed on site metadata (no Phase 1 corpus
                    carries site metadata yet — see census)
- dedup             post-filter on any split: drop valid/test records whose
                    normalized target side is an exact or near duplicate of a
                    train target (formulaic funerary phrases are rampant)
- sign_heldout      hold out whole Gardiner/Unicode sign TYPES; test =
                    sentences containing a held-out sign (simulated
                    undeciphered signs)

All schemes are deterministic in (records order, seed). Nothing here ever
reads a frozen split; it only creates them.
"""

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from glyphos.data.schema import Record
from glyphos.utils.hashing import config_hash
from glyphos.utils.seed import derive_seed

PARTS = ("train", "valid", "test")
DEFAULT_RATIOS = {"train": 0.8, "valid": 0.1, "test": 0.1}


class SplitError(ValueError):
    pass


def _check_ratios(ratios: dict) -> None:
    if set(ratios) != set(PARTS):
        raise SplitError(f"ratios must define exactly {PARTS}, got {sorted(ratios)}")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise SplitError(f"ratios must sum to 1, got {ratios}")


def _assign_groups(
    groups: Sequence[str], ratios: dict, seed: int, weights: dict[str, int] | None = None
) -> dict[str, str]:
    """Deterministically assign whole groups to partitions, honoring ratios by
    record weight (number of records per group) rather than group count."""
    _check_ratios(ratios)
    rng = np.random.default_rng(derive_seed(seed, "group-assignment"))
    order = list(groups)
    rng.shuffle(order)
    weights = weights or dict.fromkeys(order, 1)
    total = sum(weights[g] for g in order)
    assignment: dict[str, str] = {}
    cum = 0
    bounds = {
        "train": ratios["train"] * total,
        "valid": (ratios["train"] + ratios["valid"]) * total,
    }
    for g in order:
        cum += weights[g]
        if cum <= bounds["train"] or not assignment:
            assignment[g] = "train"
        elif cum <= bounds["valid"]:
            assignment[g] = "valid"
        else:
            assignment[g] = "test"
    for part in PARTS:
        if part not in assignment.values():
            raise SplitError(
                f"partition {part!r} is empty: too few groups ({len(order)}) for {ratios}"
            )
    return assignment


def split_random(records: list[Record], ratios: dict, seed: int) -> dict[str, list[Record]]:
    """Sentence-level random split. Diagnostic only — leaks documents by design."""
    _check_ratios(ratios)
    rng = np.random.default_rng(derive_seed(seed, "random-split"))
    idx = np.arange(len(records))
    rng.shuffle(idx)
    n = len(records)
    n_train = round(n * ratios["train"])
    n_valid = round(n * ratios["valid"])
    out = {
        "train": [records[i] for i in idx[:n_train]],
        "valid": [records[i] for i in idx[n_train : n_train + n_valid]],
        "test": [records[i] for i in idx[n_train + n_valid :]],
    }
    _no_empty(out, "random")
    return out


def split_by_group(
    records: list[Record],
    group_fn: Callable[[Record], str],
    ratios: dict,
    seed: int,
) -> dict[str, list[Record]]:
    groups: dict[str, int] = {}
    for rec in records:
        g = group_fn(rec)
        groups[g] = groups.get(g, 0) + 1
    assignment = _assign_groups(sorted(groups), ratios, seed, weights=groups)
    out: dict[str, list[Record]] = {p: [] for p in PARTS}
    for rec in records:
        out[assignment[group_fn(rec)]].append(rec)
    _no_empty(out, "grouped")
    return out


def split_document(records: list[Record], ratios: dict, seed: int) -> dict[str, list[Record]]:
    return split_by_group(records, lambda r: r.doc_id, ratios, seed)


def split_metadata(
    records: list[Record], meta_key: str, ratios: dict, seed: int
) -> dict[str, list[Record]]:
    """period_heldout / site_heldout: whole metadata buckets held out."""
    missing = sum(1 for r in records if meta_key not in r.meta)
    if missing == len(records):
        raise SplitError(f"no record carries meta[{meta_key!r}]; scheme unavailable")
    return split_by_group(records, lambda r: str(r.meta.get(meta_key, "unknown")), ratios, seed)


def period_bucket(rec: Record) -> str:
    """Century bucket from dateNotBefore (e.g. -1580 -> 'c-16'); 'unknown' if absent."""
    raw = rec.meta.get("dateNotBefore")
    try:
        year = int(str(raw))
    except (TypeError, ValueError):
        return "unknown"
    return f"c{year // 100:+d}"


def split_period(records: list[Record], ratios: dict, seed: int) -> dict[str, list[Record]]:
    """period_heldout: whole century buckets held out."""
    buckets = {period_bucket(r) for r in records}
    if len(buckets - {"unknown"}) < 3:
        raise SplitError(
            f"period_heldout needs >= 3 dated century buckets, found {sorted(buckets)}"
        )
    return split_by_group(records, period_bucket, ratios, seed)


def _no_empty(splits: dict[str, list[Record]], label: str) -> None:
    empty = [p for p in PARTS if not splits[p]]
    if empty:
        raise SplitError(f"{label} split produced empty partition(s): {empty}")


# -- dedup ------------------------------------------------------------------


_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_target(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _PUNCT_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


def levenshtein_within(a: str, b: str, max_dist: int) -> bool:
    """True iff edit distance(a, b) <= max_dist. Banded DP with early exit."""
    if abs(len(a) - len(b)) > max_dist:
        return False
    if a == b:
        return True
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, cb in enumerate(b, start=1):
        cur = [j] + [0] * len(a)
        row_min = j
        for i, ca in enumerate(a, start=1):
            cur[i] = min(prev[i] + 1, cur[i - 1] + 1, prev[i - 1] + (ca != cb))
            row_min = min(row_min, cur[i])
        if row_min > max_dist:
            return False
        prev = cur
    return prev[len(a)] <= max_dist


@dataclass
class DedupReport:
    checked: int
    removed_exact: int
    removed_near: int

    @property
    def removed(self) -> int:
        return self.removed_exact + self.removed_near


def dedup_against_train(
    splits: dict[str, list[Record]],
    target_field: str,
    max_norm_dist: float = 0.1,
) -> tuple[dict[str, list[Record]], DedupReport]:
    """Drop valid/test records whose normalized target is an exact or near
    duplicate (normalized edit distance <= max_norm_dist) of any train target.

    Removal counts are ALWAYS reported — silent truncation would read as
    'covered everything' when it didn't.
    """
    train_norms = sorted(
        {normalize_target(r.fields[target_field]) for r in splits["train"]},
        key=len,
    )
    train_exact = set(train_norms)
    train_lens = np.array([len(t) for t in train_norms])

    # Bag-of-characters lower bound: ceil(L1/2) <= edit distance, so filtering
    # on it never drops a true near-duplicate — it only skips hopeless pairs.
    alphabet = {ch: i for i, ch in enumerate(sorted({c for t in train_norms for c in t}))}

    def _bag(text: str) -> np.ndarray:
        v = np.zeros(len(alphabet) + 1, dtype=np.int32)
        for ch in text:
            v[alphabet.get(ch, len(alphabet))] += 1
        return v

    train_bags = np.stack([_bag(t) for t in train_norms]) if train_norms else np.zeros((0, 1))

    checked = removed_exact = removed_near = 0
    out = {"train": list(splits["train"])}
    for part in ("valid", "test"):
        kept = []
        for rec in splits[part]:
            checked += 1
            norm = normalize_target(rec.fields[target_field])
            if norm in train_exact:
                removed_exact += 1
                continue
            max_dist = int(max_norm_dist * len(norm))
            near = False
            if max_dist > 0:
                # length window first (arrays are length-sorted), bag bound
                # only inside it — orders of magnitude fewer rows touched
                lo = int(np.searchsorted(train_lens, len(norm) - max_dist, side="left"))
                hi = int(np.searchsorted(train_lens, len(norm) + max_dist, side="right"))
                if lo < hi:
                    bag_lb = np.abs(train_bags[lo:hi] - _bag(norm)).sum(axis=1) // 2
                    for idx in np.flatnonzero(bag_lb <= max_dist):
                        if levenshtein_within(norm, train_norms[lo + int(idx)], max_dist):
                            near = True
                            break
            if near:
                removed_near += 1
            else:
                kept.append(rec)
        out[part] = kept
    report = DedupReport(checked=checked, removed_exact=removed_exact, removed_near=removed_near)
    _no_empty(out, "dedup")
    return out, report


# -- sign holdout -----------------------------------------------------------


HIEROGLYPH_RANGES = ((0x13000, 0x1345F),)  # Egyptian Hieroglyphs + Format Controls
_G_TAG_RE = re.compile(r"<g>([^<]+)</g>")


def extract_signs(text: str) -> set[str]:
    """Sign inventory of a hieroglyphic string: Unicode signs in the Egyptian
    block plus <g>Gardiner-code</g> tokens for signs outside Unicode."""
    signs = {ch for ch in text if any(lo <= ord(ch) <= hi for lo, hi in HIEROGLYPH_RANGES)}
    signs.update(_G_TAG_RE.findall(text))
    return signs


@dataclass
class SignHoldout:
    held_out_signs: list[str]
    splits: dict[str, list[Record]]
    n_test_with_sign: int


def split_sign_heldout(
    records: list[Record],
    sign_field: str,
    ratios: dict,
    seed: int,
    n_signs: int = 25,
    freq_band: tuple[float, float] = (0.25, 0.75),
) -> SignHoldout:
    """Hold out whole sign TYPES: test = sentences containing any held-out sign.

    Signs are sampled from the middle of the document-frequency ranking —
    top signs would send everything to test, bottom signs yield no test data.
    valid is carved from the remaining records by document.
    """
    doc_freq: dict[str, int] = {}
    per_record: list[set[str]] = []
    for rec in records:
        signs = extract_signs(rec.fields[sign_field])
        per_record.append(signs)
        for s in signs:
            doc_freq[s] = doc_freq.get(s, 0) + 1
    if not doc_freq:
        raise SplitError(f"no signs found in field {sign_field!r}; wrong corpus?")

    ranked = sorted(doc_freq, key=lambda s: (-doc_freq[s], s))
    lo = int(len(ranked) * freq_band[0])
    hi = int(len(ranked) * freq_band[1])
    band = ranked[lo:hi]
    if len(band) < n_signs:
        raise SplitError(
            f"frequency band has only {len(band)} sign types, need {n_signs}; "
            f"corpus has {len(ranked)} total"
        )
    rng = np.random.default_rng(derive_seed(seed, "sign-heldout"))
    held = sorted(rng.choice(band, size=n_signs, replace=False).tolist())
    held_set = set(held)

    test = [rec for rec, signs in zip(records, per_record, strict=True) if signs & held_set]
    rest = [rec for rec, signs in zip(records, per_record, strict=True) if not (signs & held_set)]
    if not test or not rest:
        raise SplitError("sign holdout produced an empty partition; adjust n_signs/freq_band")

    tv_ratio = ratios["train"] + ratios["valid"]
    sub = {
        "train": ratios["train"] / tv_ratio,
        "valid": ratios["valid"] / tv_ratio,
        "test": 0.0,
    }
    # carve valid from the sign-free remainder by document (no test here)
    groups: dict[str, int] = {}
    for rec in rest:
        groups[rec.doc_id] = groups.get(rec.doc_id, 0) + 1
    rng2 = np.random.default_rng(derive_seed(seed, "sign-heldout-valid"))
    order = sorted(groups)
    rng2.shuffle(order)
    total = sum(groups.values())
    cum, train_docs = 0, set()
    for g in order:
        cum += groups[g]
        if cum <= sub["train"] * total or not train_docs:
            train_docs.add(g)
    splits = {
        "train": [r for r in rest if r.doc_id in train_docs],
        "valid": [r for r in rest if r.doc_id not in train_docs],
        "test": test,
    }
    _no_empty(splits, "sign_heldout")
    return SignHoldout(held_out_signs=held, splits=splits, n_test_with_sign=len(test))


# -- versioning -------------------------------------------------------------


def split_version(splits: dict[str, list[Record]]) -> str:
    return config_hash({p: sorted(r.sent_id for r in splits[p]) for p in PARTS})
