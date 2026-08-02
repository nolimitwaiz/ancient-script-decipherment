# Phase 1 — data: acquisition, census, frozen splits (completed 2026-08-02)

`make check` green: lint + 80 tests + no-pretrained gate + smoke. Eight
corpora ingested with provenance and content hashes, 19 test partitions
frozen and guard-enforced, census committed. No model training occurred —
everything in this phase is download + file processing (heaviest single step:
Coptic dedup, 15 s of CPU).

## Census (full table: docs/census/census.md, regenerable via `prepare_data.py census`)

| corpus | sentences | docs | primary tokens | translation | data_version |
|---|---|---|---|---|---|
| tla_earlier_egyptian | 12,773 | 95¹ | 70,267 (hieroglyphs) | German | 62eace20bffb |
| tla_late_egyptian | 3,606 | 34¹ | 24,437 (hieroglyphs) | German | a287ffa107c2 |
| tla_demotic | 13,383 | 130¹ | 117,314 (translit) | German | 08b86674d85a |
| coptic_scriptorium | 52,105 | 1,490 | 736,585 | English (partial) | 4444354f3147 |
| hebrew_morphhb | 23,213 | 39 books | 306,785 | — | ffee0c50b1e9 |
| ugaritic_hebrew_cognates | 43,951 pairs | — | — | — | b555b738b83a |
| linearb_greek_cognates | 919 pairs | — | — | — | 5e3746066faf |
| logogram_nlp | inventory² | — | — | — | inventory |

¹ *Dating cohorts*, not true documents — see "TLA pseudo-documents" below.
² LNA 165 MB / EGY 10 MB / ZHO 138 MB, mixed images+metadata; per-task parsing
lands with the Phase 7 comparison hooks. Their cuneiform data is absent from
the repo (gap filed).

Sources verified live: TLA = the full official HuggingFace org (exactly three
premium exports, CC-BY-SA-4.0); Coptic SCRIPTORIUM via sparse checkout of the
1,717 CoNLL-U files; Hebrew = OpenScriptures morphhb (WLC, CC BY 4.0); both
decipherment rungs' cognate data from Luo et al.'s own NeuroDecipher release —
an unplanned win: rung 2 (Linear B→Greek) is now runnable without DĀMOS.

## TLA pseudo-documents (the honest caveat of this phase)

The premium exports carry NO text IDs, and empirically are not text-ordered
(contiguous same-date runs ≈ 1 sentence each). `doc_id` is therefore the
**dating cohort**: all sentences sharing the exact (dateNotBefore,
dateNotAfter, authors) triple. A real text's sentences always share these
values, so no text can straddle partitions — cohorts over-merge (95/34/130
groups), which errs strict, never leaky. Real text IDs remain an open gap;
obtaining them from TLA maintainers would upgrade cohort-held-out to true
document-held-out.

## Splits (seed 1234, ratios 80/10/10 by record weight over whole groups)

All materialized under `data/<corpus>/splits/<scheme>/v1/`, each with
`split_info.json`; **all 19 test partitions hashed into
`configs/frozen_splits.json` (committed) and verified**. The guard now
*enforces*: reading a frozen test file re-hashes it (tamper → hard error),
and any write-mode open of a frozen file is refused. Covered by tests both
ways.

Headline scheme = `dedup` (document/cohort-held-out + cross-split near-dup
removal, normalized edit distance ≤ 0.1, exact + banded-Levenshtein with a
bag-of-characters lower bound for speed). Removals — always reported, never
silent:

| corpus | dedup removed (exact + near) | test after dedup |
|---|---|---|
| tla_earlier_egyptian | 857 + 107 of 2,821 checked | 718 |
| tla_late_egyptian | 54 + 8 | 662 |
| tla_demotic | 171 + 54 | 1,283 |
| hebrew_morphhb | 50 + 9 | 3,227 |
| coptic_scriptorium | 1,597 + 249 | 4,541 |

The Earlier Egyptian rate (~34% of test/valid removed) quantifies exactly the
formulaic-duplication problem the 2026 contamination audit flagged: a third of
a naive test set is near-verbatim in train.

Scheme availability: `sign_heldout` (25 mid-frequency sign types held out
entirely; hieroglyphic corpora only) — Earlier: 197 test sentences, Late: 335.
`period_heldout` (century buckets): Earlier + Demotic only — Late Egyptian has
just 8 buckets (dropped with a note). `site_heldout`: no Phase 1 corpus
carries site metadata (spec's "where metadata allows" — it doesn't, recorded).
`random` exists as the labeled-diagnostic only.

## Infrastructure added

- `glyphos/data/schema.py` — normalized Record + jsonl IO (guard-audited reads).
- `glyphos/data/ingest/` — per-corpus modules (tla, coptic, hebrew, cognates,
  logogram, stubs); every raw-tree miss raises with the exact fetch command.
- `glyphos/data/registry.py` — all 13 corpora (8 ready, 5 loud stubs) with
  their applicable schemes declared.
- `glyphos/data/splits.py` — grouped/random/period/sign/dedup engines,
  deterministic in (records, seed).
- `glyphos/data/freeze.py` + guard wiring — the freeze manifest and its
  enforcement (reentrancy-latched so hash-on-read can't recurse).
- `glyphos/data/census.py` + `scripts/prepare_data.py`
  (ingest/census/split/freeze/verify).
- 31 new tests (80 total).

## Environment note

The Desktop/iCloud venv problem RECURRED overnight (6,760 files re-flagged
hidden; Python skips hidden .pth files → imports silently break). Durable fix
applied: `.venv` is now a symlink to `~/.venvs/glyphos` (outside iCloud), same
pattern as `data → ~/dev/glyphos-data`. Both symlinks are gitignored; nothing
inside iCloud holds mutable tool state anymore.

## Next: Phase 2 — rendering engine

PangoCairo + Noto (incl. Egyptian Hieroglyphs), 24×24 windows / stride 12,
degradation suite with `damage_level`, multiscript SSL stream; `<g>…</g>`
sign-markup handling. Requires cairo/pango system libraries — install path for
this Mac to be decided at phase start (no Homebrew present).
