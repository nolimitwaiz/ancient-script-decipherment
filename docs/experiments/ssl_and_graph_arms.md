# Experiment designs: SSL objectives (MAE / I-JEPA / DINO) and the graph decipherment arm

Written 2026-08-06, after the first pixel-vs-BPE comparison came back
**negative** for pixels (BPE 5.616 ± 0.071 vs pixel 5.975 ± 0.098 valid CE,
≈4.2 seed-sd). That result is what makes these arms decidable rather than
decorative: each one below exists to attack a specific, named weakness, and
each carries a numeric success criterion fixed *before* it runs.

Hard constraints unchanged: every objective is implemented and trained
in-repo. No pretrained checkpoints (DINOv2/v3, I-JEPA releases) enter the
sealed pipeline — they are permitted only as labelled open-book comparisons in
`contrib/openbook/` (Phase 8).

---

## Shared decision criterion for the SSL arms (A1–A3)

The pixel arm's deficit to close is **Δ = 0.359 nats** (random-init pixel
5.975 → BPE 5.616), at 3 seeds, same split, same target vocab.

| outcome | rule |
|---|---|
| **Success** | warm-started pixel closes ≥ 50% of Δ (valid CE ≤ 5.796) at 3 seeds |
| **Strong success** | pixel ≤ BPE (≤ 5.616): pixels win once pretrained |
| **Failure** | < 25% of Δ closed; record and stop that objective |

All three arms share the identical downstream recipe (same config, same seeds,
same steps) so the *only* variable is the pretraining objective. Pretraining
corpus is the multiscript render stream (TLA ×2 + Coptic + Hebrew + Greek).

---

## A1 — MAE-style masked-window reconstruction (baseline SSL, already built)

- **Status**: implemented (`glyphos/models/ssl.py`), kill-gate passed
  (loss 1.233 → 0.042). Full 80k-step run pending.
- **Hypothesis (preregistered)**: masked-window pixel reconstruction over the
  multiscript stream learns transferable stroke features; warm-starting the
  pixel translator closes ≥50% of Δ.
- **Why it is the reference arm**: exact from-scratch analog of the pretrained
  PIXEL/ViT-MAE encoder used by the published LogogramNLP baseline, so the
  comparison isolates *data provenance* rather than objective family.
- **Ledger family**: `ssl-pretrain-multiscript` → `mt-pixel-ssl-mae`.

## A2 — I-JEPA-style latent prediction

- **Idea**: stop predicting pixels. Encode a *context* span of windows; a small
  predictor then predicts the **representations** of held-out target spans,
  supplied by an EMA target encoder (stop-gradient). Loss is L2 in latent
  space, not pixel space.
- **Why it could beat A1 here**: pixel reconstruction spends capacity on ink
  texture, paper grain and the very degradation noise our Phase-2 suite injects
  on purpose. Latent prediction is free to encode *sign identity* instead —
  which is what both translation and restoration actually consume.
- **Implementation** (~60 lines on the existing encoder):
  `context_ratio 0.6`, 4 target spans of 2–6 windows, predictor = 2-layer
  Transformer with learned positional queries, EMA momentum 0.996 → 1.0
  (cosine), targets L2-normalised.
- **Failure mode + detector (mandatory)**: representation collapse. Log
  per-batch embedding variance and the effective rank (participation ratio) of
  the target encoder's outputs every 500 steps; **abort the run** if mean
  variance falls below 10% of its step-500 value. A collapsed encoder can post
  a beautiful loss curve and carry zero information — without this detector the
  arm is unfalsifiable.
- **Secondary probe**: frozen-encoder linear probe for Gardiner-class
  prediction, reported alongside downstream CE.
- **Ledger family**: `ssl-pretrain-ijepa` → `mt-pixel-ssl-ijepa`.

## A3 — DINO-style self-distillation

- **Idea**: two views of the same rendered strip → student and EMA teacher →
  both project onto K=4096 prototypes → cross-entropy of student against a
  centred, sharpened teacher. No negatives, no reconstruction.
- **Why it fits this project specifically**: DINO needs heavy, *semantically
  faithful* augmentation and lots of data — and we own both. Our degradation
  suite (erosion, damage-shaped occlusion, stone/papyrus texture, jitter) is a
  domain-true augmentation pipeline rather than generic colour jitter, and the
  renderer is an unbounded sample generator. Multi-crop maps naturally onto
  strips: 2 global crops (long window spans) + 6 local crops (short spans).
- **Distinctive prediction**: DINO-style features should be the most
  *damage-robust* of the three, because invariance to our degradations is
  trained in explicitly. Therefore this arm is judged not only on translation
  CE but on the **damage-level sweep** (Phase 7.4): restoration recall@k at
  `damage_level` ∈ {0, 0.25, 0.5, 0.75, 1.0}.
- **Failure modes + detectors**: collapse (monitor teacher output entropy —
  abort if it falls below 0.5 × uniform), and temperature/centring sensitivity
  (the known finicky part). Because it has the most hyperparameters, it is
  **gated**: run A3 only if A1 or A2 clears the 25% bar, else the tuning cost
  is not justified.
- **Ledger family**: `ssl-pretrain-dino` → `mt-pixel-ssl-dino`.

---

## B — The graph arm: Ventris's grid, formalised (GIN + optimal transport)

This is the arm with the most upside, and it belongs to the **decipherment**
core (Phase 5), not to translation.

**The insight it operationalises.** Ventris cracked Linear B before knowing any
sound values by tabulating which signs occurred in which contexts: signs
sharing a consonant or a vowel pattern together. That is structural matching on
a co-occurrence graph. Formalise it:

- **Lost-script graph** `G_X`: nodes = sign types; edge weights = co-occurrence
  and adjacency (bigram transition) statistics, Ledoit–Wolf shrunk (Phase 6.3
  already requires this — raw counts from a few thousand lines are exactly the
  ill-conditioned case).
- **Known-relative graph** `G_Y`: nodes = characters/phonemes of Hebrew /
  Greek / Coptic, identical construction from the sealed corpora.
- **Node features carry NO identity** — only structural descriptors (degree,
  bucketed frequency rank, word-initial/medial/final distribution, entropy of
  neighbours). Identity features would let the model memorise instead of
  matching structure, which would quietly void the whole test.
- **Matching**: GIN node embeddings → entropic OT (Sinkhorn) with the existing
  cost terms (sparsity, frequency prior, phonetic prior, visual prior).

**Ablation ladder** (each rung must beat the one below it):

| rung | method | learning |
|---|---|---|
| 0 | frequency-rank matching | none (existing null baseline) |
| 1 | Gromov–Wasserstein on raw co-occurrence graphs | none |
| 2 | GIN embeddings + Sinkhorn OT | learned structure |
| 3 | rung 2 + phonetic (IPA) prior | + linguistics |
| 4 | rung 3 + visual prior (pixel-encoder cosine) | + our images |

Rung 1 matters a lot: **Gromov–Wasserstein needs no neural network at all**. If
GW matches GIN, the GNN is decoration and we say so.

- **Validation**: artificial decipherment, Ugaritic→Hebrew first (43,951
  cognate pairs available), then Linear B→Greek, then Egyptian→Coptic. Score =
  sign-mapping accuracy + cognate precision/recall against the hidden key.
- **Significance is mandatory**: every claim carries the Phase-6
  permutation-null p-value over ≥1,000 constraint-respecting random mappings,
  plus the wrong-relative negative control (Ugaritic→Greek must *not* fire).
- **Hypothesis (preregistered)**: rung 2 beats rung 0 by ≥15 points of
  sign-mapping accuracy on Ugaritic→Hebrew, with permutation p < 0.01, and the
  wrong-relative control stays non-significant.
- **Known limitation to state up front**: GIN's expressive power is bounded by
  the 1-WL test, so structurally regular graphs are indistinguishable to it.
  Sign co-occurrence graphs are weighted and highly irregular, so this is
  unlikely to bind — but if rung 2 fails while rung 1 succeeds, 1-WL
  degeneracy is the first hypothesis to check.
- **Cost**: small graphs (30–1,000 nodes). This arm is **CPU-cheap and does not
  compete with the GPU queue** — it can be built while translation runs.
- **Ledger family**: `decipher-graph-{gw,gin}-{uga-heb,linb-grc,egy-cop}`.

### B2 (stretch) — glyph stroke graphs

Extract stroke/junction topology directly from the **Noto font outlines** (the
vector data is already on disk — no vectorisation research needed), giving each
sign a shape graph. Isomorphism-flavoured matching then strengthens the visual
prior for genetically related scripts (Egyptian → hieratic → demotic → Coptic).
Only worth building if rung 4 shows the visual prior carrying weight.

---

## Sequencing and cost

| order | arm | blocks on | GPU cost |
|---|---|---|---|
| 1 | A1 full SSL + warm-started pixel (3 seeds) | nothing | ~4 GPU-days |
| 2 | **B rungs 0–2** (graph decipherment) | nothing — CPU | negligible |
| 3 | A2 I-JEPA (3 seeds) | A1 finishing | ~4 GPU-days |
| 4 | B rungs 3–4 (+priors) | B2, sealed LMs (done) | negligible |
| 5 | A3 DINO | gated on A1/A2 clearing 25% | ~6 GPU-days |

Rationale: arm B costs almost nothing and targets the project's actual thesis
(decipherment), so it proceeds in parallel with the GPU work rather than
queueing behind it. A3 is deliberately last — it is the most hyperparameter-
sensitive and the least differentiated from A1/A2 until we know pretraining
helps at all here.
