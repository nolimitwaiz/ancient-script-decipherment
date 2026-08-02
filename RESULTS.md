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
