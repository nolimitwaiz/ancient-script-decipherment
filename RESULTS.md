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

### Phase 1 second pass (frontier downloads)

greek_first1k 392,478 passages / 28.8M tokens; cuneiform_cdli 132,210
tablets / 8.9M tokens (language+period metadata); meroitic_rem 18,103 lines
(first machine-readable Meroitic corpus). 26 frozen test partitions total,
all verified. Exact-dedup removals: Greek 617, CDLI 1,891.

## Phase 3 — trained models (CLSP, 2026-08-04/05)

All early-stopped on held-out cross-entropy; all preregistered in the ledger.

| model | run family | steps | best held-out CE |
|---|---|---|---|
| Hebrew sealed char-LM | sealed-lm-hebrew | 13,000 | 1.5109 |
| Coptic sealed char-LM | sealed-lm-coptic | 15,000 | 1.2582 |
| Greek sealed char-LM | sealed-lm-greek | 42,000 | 1.5994 |
| BPE translation control (Egy→De) | mt-tla-earlier-bpe | 12,000 | 5.5635 |

NOT YET headline-grade: single seed each (headline requires ≥3) and no test-set
evaluation has been run — these are valid/held-out numbers only. The frozen
test partitions remain untouched, as designed.

Pixel-model kill-gate (200 steps, run 1709819): eval CE 6.3034 — pipeline
proven on GPU; full 3-seed runs pending.

### Sealed-LM hypotheses evaluated against count baselines (2026-08-05)

Preregistered claim: each sealed LM beats a unigram baseline substantially.
Baselines fit on the same train partition, scored on the same valid slice
(nats/char; frozen test partitions untouched):

| corpus | unigram | bigram | sealed LM | bits/char | vs bigram |
|---|---|---|---|---|---|
| hebrew_morphhb | 3.577 | 2.561 | **1.511** | 2.18 | −41.0% |
| coptic_scriptorium | 3.013 | 2.507 | **1.258** | 1.82 | −49.8% |
| greek_first1k | 3.760 | 2.806 | **1.599** | 2.31 | −43.0% |

Hypotheses CONFIRMED: all three cut cross-entropy by ~58% vs unigram and
41–50% vs a bigram model — they learned real orthographic structure, not
frequency. At 1.8–2.3 bits/char these are credible small-corpus character LMs,
adequate as Phase 5 plausibility scorers.
