# RESULTS.md — ledger-backed results index

Rules for this file (see CONVENTIONS.md):
- Every number cites its ledger `run_id` (and from Phase 6 on, a 95% block-
  bootstrap CI over documents).
- Every winner cites its family's `tried` count from `glyphos-ledger report` —
  distributions, not maxima.
- Headline MT/restoration numbers come from `document_heldout + dedup` splits
  only; `random`-split numbers are diagnostics and are labeled as such.
- Tables here are regenerated, not hand-edited, once `scripts/evaluate.py`
  lands (Phase 7).

## Phase 0 — smoke pipeline (infrastructure demonstration, not science)

Toy substitution-cipher decipherment via frequency-rank matching
(12 symbols, 240 sentences, document-held-out split, seed 1337):

| metric | value | run |
|---|---|---|
| key_accuracy | 1.000 | `20260802-002751-6a0a99` (family `phase0-smoke-toy-decipherment`, tried: 1) |
| valid_token_accuracy | 1.000 | same run |
| test_token_accuracy | 1.000 | same run |

Guard audit: 1 locked-test read logged, attributed to the run. Regression
floor asserted by `make smoke`: key_accuracy ≥ 0.5.

## Phase 1+ — (empty until the phase completes)

## Phase 1 — corpus census & split integrity (data facts, not model results)

Ingested (full table: docs/census/census.md): TLA Earlier Egyptian 12,773 /
Late 3,606 / Demotic 13,383 parallel sentences (→German); Coptic SCRIPTORIUM
52,105; Hebrew (WLC) 23,213 verses; Ugaritic–Hebrew 43,951 + Linear B–Greek
919 cognate pairs; LogogramNLP inventoried. 19 test partitions frozen
(configs/frozen_splits.json) and verified; guard enforces hash-on-read +
write refusal.

Contamination quantified — dedup (near-dup vs train, headline scheme) removed
from test/valid: Earlier Egyptian 964/2,821 (~34%), Coptic 1,846, Demotic
225, Late 62, Hebrew 59. Naive splits on this data overstate test size and
inflate scores; headline numbers use `dedup` splits only.
