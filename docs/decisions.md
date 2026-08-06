# decisions.md — running log of decisions and one-line rationales

Format: `YYYY-MM-DD · decision — rationale`. Newest at the bottom of each
phase block. Anything touching the hard constraints is NOT decided here alone.

## Phase 0 (2026-08-01)

- Repo lives at `~/dev/glyphos` — established local convention for research
  repos, and off iCloud (dataless-file eviction has corrupted corpora before).
- Python 3.12 pinned via `.python-version` — newest version with guaranteed
  wheels for the full future stack (torch, pycairo) as of Aug 2026.
- `uv` + hatchling; dev deps as a PEP 735 dependency group — spec-mandated uv,
  lockfile committed for CI `--frozen` reproducibility later.
- Plain YAML + frozen dataclasses over hydra — spec preference; strict
  unknown-key errors make silent typo'd overrides impossible.
- Ledger = two-event append-only JSONL (`register` → `complete`) at
  `runs/ledger.jsonl` — registration-before-results makes preregistration
  structural, not procedural; the file is committed (it IS the record) with
  git `merge=union` so Mac + cluster appends never conflict.
- `n_variants_tried_so_far_in_this_family` computed at registration time from
  the ledger itself — the multiple-testing count cannot be curated after the
  fact.
- Smoke runs log to the REAL ledger — "every run, no exceptions" is only
  credible if the trivial ones are in there too; family naming keeps them
  separable in reports.
- Test-set guard patches `builtins.open` AND `io.open` (pathlib routes through
  the latter) and is log-only in Phase 0 — spec asks for logging; blocking
  semantics arrive with frozen-split manifests in Phase 1.
- Guard trigger = path under `data_root` with a `test` dir component (or
  `test.*` filename), plus a generic `.../data/.../test/...` fallback for
  cluster paths — requires the data-dir contract, which is now normative.
- Smoke = toy substitution-cipher decipherment by frequency-rank matching —
  smallest real instance of the artificial-decipherment protocol; exercises
  ledger, guard, doc-held-out splits, and hashing in the exact shape later
  phases must follow. Toy data isolated under `runs/smoke/data/`.
- torch deliberately NOT a dependency yet — Phase 0/1 need none of it; keeps
  `uv sync` seconds-fast; arrives with Phase 3 models.
- No TeX on this Mac (no MacTeX/brew) → installed `tectonic` 0.17 to
  `~/.local/bin` — single user-level binary, no admin, CTAN packages cached on
  first use.
- Report freshness gate = mtime check + `\ReportStamp` string extracted from
  the PDF via pypdf — pure-Python replacement for missing pdftotext; stamp
  bumping on edit is the convention that makes staleness detectable.
- COMET excluded from headline metrics (chrF primary) — a learned metric is a
  pretrained model; also insensitive to the contextual distinctions we target.
- Kendall tau implemented by hand (tau-a, O(n²) over configs) — avoids a scipy
  dependency before Phase 5 actually needs scipy for min-cost flow.

## Post-Phase-0 (2026-08-01, same day)

- Repo moved `~/dev/glyphos` → `~/Desktop/glyphos` — at Waiz's request.
  Supersedes the off-iCloud placement rationale above; the caveat stands.
- Incident during the move: the venv recreated on Desktop came up with every
  file flagged macOS-hidden (`UF_HIDDEN`, likely an iCloud file-provider race
  during folder ingestion); Python ≥3.12 silently skips hidden `.pth` files,
  so the editable install vanished and pytest could not import `glyphos`.
  Fix: `chflags -R nohidden .venv`; a second fresh `uv sync` produced zero
  hidden files. Mitigation if it recurs: venv outside iCloud via
  `UV_PROJECT_ENVIRONMENT`, or move the repo back.

## Phase 1 (2026-08-02)

- iCloud venv breakage RECURRED (6,760 files re-flagged hidden overnight) —
  correcting Phase 0's "did not recur" note. Durable fix: `.venv` is a symlink
  to `~/.venvs/glyphos`; rule of thumb recorded: no mutable tool state inside
  iCloud-managed paths.
- Corpora live at `~/dev/glyphos-data` behind a repo-root `data` symlink —
  keeps the data-dir contract intact on Mac and cluster while keeping bytes
  off iCloud.
- TLA doc_id = non-contiguous dating cohort (dateNotBefore, dateNotAfter,
  authors) — exports have no text IDs and are not text-ordered (contiguous
  date runs ≈ 1 sentence); cohorts over-merge, which errs strict for
  held-out splitting, never leaky.
- Hebrew Bible source = OpenScriptures morphhb (WLC) over ETCBC/BHSA —
  trivially parseable OSIS XML, clean CC BY 4.0; BHSA's text-fabric adds a
  dependency for no Phase 1 gain.
- Ugaritic AND Linear B cognate data taken from Luo et al.'s NeuroDecipher
  release — exact published-comparison data; rung 2 unblocked without DĀMOS.
- LogogramNLP ingested inventory-only; its .pth pickles are never opened
  (untrusted pickles); per-task parsing deferred to the Phase 7 hooks.
- processed/ layout simplified to flat records.jsonl + manifest.json carrying
  data_version (hash) — content-hash-named dirs were self-referential churn.
- Dedup target field: translation side for parallel corpora, primary text for
  monolingual ones (partial translations like Coptic text_en can't anchor
  dedup).
- Dedup candidate filter: numpy bag-of-characters L1/2 lower bound (a true
  edit-distance lower bound → exact results) — Hebrew went from >10 min to
  2.6 s.
- tla_late_egyptian: period_heldout dropped (8 century buckets can't fill
  3 partitions meaningfully); recorded in registry with comment.
- sign_heldout: 25 sign types sampled from the 25–75% document-frequency band
  (top signs would send everything to test; rare signs yield no test).
- First1KGreek (~1 GB TEI) deferred to cluster-side ingest before Phase 3 —
  needed only for the sealed Greek LM; stub + gap entry meanwhile.
- Freeze manifest lives at configs/frozen_splits.json (committed) — data/ is
  never committed, so split identity must live in git; guard enforces
  hash-on-read and refuses write-mode opens of frozen files.

## Phase 1 addendum (2026-08-02, frontier downloads)

- Cuneiform source = CDLI open-data snapshot (2023-10) — the canonical open
  corpus and what LogogramNLP built on; ancient-text staleness is a non-issue.
  Record granularity = one tablet (the tablet IS the document).
- Greek deferral reversed: First1KGreek parsed locally in 15 s — the "defer to
  cluster" call was based on an overestimate; corrected rather than kept.
- Meroitic corpus = Otten & Anastasopoulos 2025 release; its pretrained
  embeddings are quarantined — the sealed constraint covers data releases
  that happen to contain trained parameters.
- period_heldout generalized: numeric dating (TLA) → century buckets;
  categorical period strings (CDLI) → whole-value groups.
- Mayan/Libyco-Berber stay stubs on evidence (application-gated / defunct DB),
  with concrete unblock paths recorded in data_gaps — not for lack of trying.
- Greek/CDLI dedup = exact-only (threshold flag --dedup-max-norm-dist 0) —
  near-dup at this scale costs hours for marginal benefit on LM corpora;
  TLA headline splits keep the full near-dup pass. Parameter recorded in
  every split_info.json.
- Long-running local steps run under `caffeinate -i` — system sleep froze a
  10-hour wall-clock job at 38 CPU-minutes; wall time is not compute time.
- Compute policy extended (Waiz, 2026-08-02): the cluster is the default for
  heavy CPU work too, not just GPU — SLURM CPU jobs (cpu.sbatch, 16 cores)
  for data prep and rendering; GPU requests are typeless (any free card).
  Cluster harness at scripts/cluster/ (sync, env bootstrap, job templates);
  blocked only on one-time interactive key authorization.

## Phase 2 start (2026-08-02)

- Render environment = micromamba (user-level, ~/micromamba-envs/glyphos-render):
  pycairo + pygobject + pango + harfbuzz (typelib was the missing piece) +
  numpy. Ancient-script Noto fonts (Egyptian Hieroglyphs, Coptic, Linear B,
  Cuneiform, Meroitic) installed to ~/.local/share/fonts — conda fontconfig
  does NOT scan ~/Library/Fonts. All six target scripts confirmed rendering
  ink at 10pt. Open item: unify render env with the uv venv (two interpreters
  is temporary; cluster setup will install pycairo against system cairo).

## Phase 3/7 planning (2026-08-05)

- SSL objective = masked-window reconstruction (MAE family) for the headline:
  exact from-scratch analog of PIXEL/ViT-MAE (the LogogramNLP comparison),
  and pixel-level detail IS the signal for glyphs. JEPA-style latent
  prediction and DINO-style self-distillation (both implemented from scratch)
  are queued as Phase 7 ablation arms — objectives are unconstrained, weights
  are not.
- DINOv2/v3 CHECKPOINTS are banned from the sealed pipeline (web-pretrained =
  lookahead contamination; CI gate enforces). Permitted one day only as an
  explicitly-labeled open-book comparison in contrib/openbook/ (Phase 8).

## Phase 3 findings (2026-08-05)

- Sealed-LM preregistered hypotheses CONFIRMED against computed unigram AND
  bigram baselines (glyphos/eval/lm_baselines.py). Baselines are now code, not
  an assumption — any future LM claim must clear them.
- KNOWN LIMITATION in the in-flight headline runs: target subword vocab was
  requested at 10k, and SentencePiece yields 8,893 pieces for only ~142k
  training subword tokens = 16 tokens/vocab slot (healthy >50). A 2k vocab
  gives 90 tokens/slot. Both arms share the target vocab, so the pixel-vs-BPE
  comparison stays fair, but both are handicapped in absolute terms.
  NOT fixed by cancelling: per-token perplexity is NOT comparable across
  vocabularies, so vocab must be selected on a vocab-independent metric.
  Phase 4 selects it by chrF on validation, then the headline is rerun at the
  chosen size — recorded as a separate ledger family so nothing is
  cherry-picked.
- 2026-08-06 INCIDENT: sync.sh pushed the whole repo including runs/, so the
  local ledger overwrote the cluster's compacted ledger and destroyed 12
  events. Fixes: (a) sync.sh never pushes runs/ — the cluster ledger is
  authoritative for cluster runs and is merged on pull; (b)
  scripts/recover_ledger.py reconstructs lost entries from checkpoint
  config.json (written at run start) + slurm stdout, marking every recovered
  run `reconstructed=true` with its evidence cited. Reconstruction is NOT
  preregistration and is labelled as such.
