# Phase 3 — models: from-scratch stack, trained on the cluster (2026-08-05)

Four model families implemented from scratch, smoke-verified on CPU, and
trained for real on CLSP GPUs. Every parameter in every model was randomly
initialized and trained inside this repo; `make check-no-pretrained` gates the
sealed tree on every commit.

## What exists

| component | file | notes |
|---|---|---|
| Transformer core | `models/transformer.py` | pre-LN blocks, sinusoidal positions, Salesky defaults (12 enc / 3 dec, d=512, FF 4096, 4 heads) |
| Pixel front end | `models/pixel_encoder.py` | 24×24 window → Conv2D 3×3 → BatchNorm → ReLU → linear → d_model |
| Seq2seq (both arms) | `models/seq2seq.py` | pixel and BPE front ends share one core → architecture-matched by construction; tied output embedding; label smoothing 0.2 |
| Sealed char LM | `models/char_lm.py` | decoder-only causal LM + in-repo `CharVocab` |
| Masked-window SSL | `models/ssl.py` | MAE-family reconstruction + encoder warm-start into the pixel model |
| Tokenizer | `models/tokenizer.py` | SentencePiece **trainer** run on our corpora (never a downloaded tokenizer) |
| Training loop | `train/loop.py` | explicit PyTorch: Adam, linear warmup → inv-sqrt, grad accum, clipping, bf16-on-CUDA with fp32 fallback, DDP from env, early stopping, rank-0 checkpoints |
| Runner | `scripts/train.py` | config-driven, ledger-preregistered, `--kill-gate`, `--seed` override |

Full-size pixel model: ~97M parameters (spec band 50–400M). Smoke: all four
families train 50 CPU steps and their losses fall, inside `make smoke` (~5 s).

## Trained models (CLSP, all early-stopped, all ledgered)

| model | job | steps | wall | best held-out CE |
|---|---|---|---|---|
| sealed-lm-hebrew | 1707434 | 13,000 | ~4 h | **1.5109** |
| sealed-lm-coptic | 1707436 | 15,000 | ~5 h | **1.2582** |
| sealed-lm-greek | 1707435 | 42,000 | 21.4 h | **1.5994** |
| mt-tla-earlier-bpe (control) | 1707462 | 12,000 | 9.7 h | **5.5635** |

Sealed LMs are ~10.7M params each and are the ONLY language models the
decipherment track may consume (`docs/sealed_models.md`). The BPE control is
the line the pixel model must beat.

Kill-gates (200 steps each, required before any long run):
`char_lm` 1707332 loss 4.75→2.43 · `translation_bpe` 1707437 9.48→6.68 ·
`translation_pixel` 1709819 **9.65→6.48, eval CE 6.30** — the first time
rendered hieroglyphs went through the pixel encoder on a GPU.

## Cluster hardening (what the kill-gates caught)

Kill-gates paid for themselves four times over; each failure cost seconds, not
hours:

1. **torch/driver mismatch** — torch 2.13 silently fell back to CPU on older
   nodes (49 min of fake "GPU" training). Linux now pins **torch 2.4.1+cu118**,
   which runs on every driver generation in the fleet; macOS keeps 2.13 for
   local smoke.
2. **No silent fallbacks** — `gpu.sbatch` now asserts `torch.cuda.is_available()`
   before launching; a GPU job dies loudly rather than training on CPU.
3. **Broken node** — `c04` throws `CUDA unknown error` on contact; excluded
   from all jobs (report to CLSP admins).
4. **Renderer bug** — paragraph-length Greek asked cairo for a surface wider
   than it will allocate. `render_strip` now truncates to
   `max_strip_width(cfg)` before allocating; nothing is lost because
   `slice_windows` already caps at `max_windows`. Regression test added.

Cluster environments: `uv` venv (training) and a micromamba env with
cairo/pango + torch (rendering jobs); ancient-script Noto fonts installed and
hieroglyph rendering verified identical to the Mac.

## Deviations / open items

- Headline runs still owe 3 seeds each (`--seed` override now exists).
- SSL pretraining kill-gate in progress; full 80k-step run and the
  SSL-warm-started pixel model follow.
- Sealed German LM (target-side rescoring) not yet trained — only needed if
  Phase 5 rescoring calls for it.

## Next: Phase 4

Translation metrics (chrF primary, BLEU secondary, COMET excluded by design),
Ithaca-style restoration with damage-shaped masks from the Phase 2 suite,
conformal candidate sets, and the auxiliary heads — all evaluated on the
frozen test partitions for the first time.
