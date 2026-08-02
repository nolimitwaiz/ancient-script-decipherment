# CONTEXT.md — what GLYPHOS is and why

## Mission

A from-scratch pipeline for **visual modeling, restoration, translation, and
constrained decipherment of ancient scripts**, validated by *artificial
decipherment*: hide the key of an already-deciphered script (Ugaritic, Linear
B, Egyptian→Coptic), run the pipeline, measure how much of the key it recovers.
Graduate research project at JHU CLSP; advisor Philipp Koehn.

## Three design lineages

1. **Pixel representations (Salesky/Koehn).** Text rendered to images →
   overlapping 24×24 windows (stride 12, 10pt @ 120 DPI, PangoCairo + Noto) →
   thin conv block → deep-encoder/shallow-decoder Transformer (12/3, d=512).
   Script-level parameter sharing; transfer to unseen scripts. Our twist: the
   renderer doubles as an unbounded self-supervised data generator, which is
   the from-scratch replacement for pretrained pixel models.
2. **Constrained decipherment (Luo & Barzilay).** Decipherment as matching
   between a lost script and a known relative: character/sign embeddings,
   entropic-OT/min-cost-flow cognate alignment with monotonicity, sparsity,
   and frequency-matching constraints, IPA phonetic priors — plus our novel
   **visual prior** from pixel-encoder embeddings of sign images.
3. **Quant epistemics.** Hierarchical shrinkage, block bootstrap CIs,
   Ledoit-Wolf covariance shrinkage, permutation-null significance with
   constraint-respecting nulls, conformal prediction sets, and a mandatory
   experiment ledger with multiple-testing accounting. Every claim must
   survive this layer; a 2026 contamination audit found published hieroglyph
   MT results inflated by 29–47 BLEU by leaky splits — that failure mode is
   designed out at the split level (document-held-out + dedup minimum).

## Why from scratch (the scientific core)

If any component was pretrained on the open web, it may have seen the
scholarship on the very scripts we "decipher" — lookahead bias that voids the
validation. So: **no pretrained weights anywhere** (CI-enforced), sealed
known-relative LMs trained in-repo on documented corpora, and an `openbook/`
quarantine for future explicitly-contaminated comparisons.

## Primary corpora (Phase 1)

| Corpus | Role | Source |
|---|---|---|
| TLA Earlier/Late Egyptian (hieroglyphs + translit + German) | primary translation + restoration | HuggingFace org `thesaurus-linguae-aegyptiae` |
| Coptic SCRIPTORIUM (~2.3M words) | known relative for Egyptian rung | GitHub `CopticScriptorium/corpora` |
| LogogramNLP (Linear A, hieroglyphic, Cuneiform, Bamboo) | published baseline to beat | GitHub `taineleau/logogramNLP` |
| Ugaritic + Hebrew Bible | decipherment rung 1 | Luo et al. release / ETCBC-BHSA |
| Linear B (DĀMOS/LiBER) + Greek (Perseus/First1KGreek) | decipherment rung 2 | access TBD → docs/data_gaps.md |
| Meroitic, Mayan, Libyco-Berber | frontier stubs | not Phase 1 blockers |

Headline task: Egyptian→German translation; restoration (Ithaca-style) with
conformal candidate sets; decipherment ladder Ugaritic→Hebrew, Linear
B→Greek, Egyptian→Coptic.

## Hardware & stack

Dev: M5 Pro MacBook, CPU/MPS smoke only. Training: ~3× A100, CLSP SLURM, DDP,
bf16, 50M–400M params. Python 3.12 via `uv`; PyTorch with explicit training
loops (no Lightning); plain YAML + dataclasses (no hydra); pytest; wandb off
by default (local JSONL is the record); pycairo/PangoCairo + Noto fonts;
hand-rolled Sinkhorn (~100 lines, POT only as a test-time cross-check).

## Roadmap (details: docs/roadmap.md)

Phase 0 scaffold/ledger ✅ → 1 data+census+frozen splits ✅ → 2 renderer+degradation
→ 3 models (pixel, BPE control, SSL pretrain, sealed LMs) → 4 translation+
restoration+aux heads → 5 OT decipherment core+ladder → 6 quant layer →
7 evaluation packaging+sweeps → 8 GUI/openbook scaffolds (deferred).
