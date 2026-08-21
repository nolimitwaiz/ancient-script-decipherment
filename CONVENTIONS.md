# Rules of work for Ancient Script Decipherment

Read CONTEXT.md for what this project is; read MEMORY.md for where it currently
stands. This file is the rulebook. These rules override defaults.

## Hard constraints (§1 of the project spec — NEVER violate)

1. **No pretrained model weights.** No HuggingFace checkpoints, no downloaded
   encoders, no pretrained tokenizers from other projects, no `torch.hub`.
   Every parameter is randomly initialized and trained in this repo. Pretrained
   **datasets** are allowed; pretrained **models** are not. The only future
   exception is `contrib/openbook/` (explicitly-labeled contaminated
   comparisons, never headline results). `make check-no-pretrained` enforces
   this; do not weaken the gate — extend its pattern list when new loading APIs
   appear.
2. **Ledger before results.** Every training/eval run — including failed,
   abandoned, and smoke runs — is registered in `runs/ledger.jsonl` with a
   non-empty `hypothesis` BEFORE results are looked at, and closed with a
   terminal status. Use `Ledger.run(...)` (Python) or `glyphos-ledger
   register/complete` (shell/SLURM). Never edit or delete ledger lines.
3. **Locked test sets.** Test splits are created once (Phase 1), hashed, and
   never touched by tuning decisions. All entry points call
   `guard.install_guard()` first thing; corpus-reading library code uses
   `guard.guarded_open`. The audit trail lives in `runs/test_access.jsonl`.
4. **Reproducibility.** Explicit seed in every config (`set_seed`), 3+ seeds
   for any headline number (`require_seeds`), `config_hash`/`data_version`/
   `split_version` recorded per run, deterministic dataloaders where feasible.

## Compute policy

- The CLSP cluster (SLURM) is the DEFAULT for anything heavy — CPU work
  included: multi-core data prep, bulk rendering, corpus processing run as
  SLURM CPU jobs (`scripts/cluster/cpu.sbatch`), not on the Mac. This Mac:
  editing, unit tests, `make smoke`, and sub-minute glue only. No model
  training here, ever.
- GPU policy on the cluster: request GPUs TYPELESS (`--gres=gpu:N`) so
  SLURM allocates whatever is free — A100 or otherwise; NEVER queue-wait on a
  specific card. Multiple GPUs (DDP) when it helps; pin a type only when a
  job genuinely requires it.
  Training code must therefore run on any CUDA generation (bf16 with fp32
  fallback where unsupported) and any world size >= 1.
- Everything runs through `uv` (`uv sync`, `uv run ...`). Never pip-install
  into system Python.

## Phase gates (working style §11.1)

Finish a phase → write `docs/phaseN.md` → `make check` green → only then
proceed. Blockers are stubbed loudly and logged in `docs/data_gaps.md`; the
build never stalls and never silently falls back. Before any multi-hour run:
one budget-unit kill-gate run first, expected outcome preregistered in the
ledger `hypothesis`.

## Commits

- Author: Waiz Khan <waizkhan008@gmail.com>, sole author. **Never add
  AI co-author trailers, "Generated with" footers, or any AI attribution.**
- Small commits, descriptive messages, tests alongside features.
- Coverage target ≥70% on `data/`, `render/`, `quant/`; models are covered by
  smoke tests.

## Documentation discipline

- Decisions with rationale → `docs/decisions.md` (one line each). Ask nothing
  you can decide — except anything touching the hard constraints above.
- Dataset blockers → `docs/data_gaps.md`.
- Results → `RESULTS.md`, every number backed by a ledger `run_id`, with CIs
  once `quant/` exists (Phase 6). Never a best value without its family's
  `tried` count (`glyphos-ledger report`).
- Session state → update `MEMORY.md` at the end of every working session.
- **Reports:** every phase gets `docs/phaseN.md` AND a LaTeX twin under
  `docs/reports/phaseN/` with a committed PDF. After ANY `.tex` edit: bump that
  file's `\ReportStamp` (e.g. `phase0-v1` → `phase0-v2`) and run
  `make reports` — it rebuilds ALL report PDFs in one pass and then runs the
  staleness gate (`scripts/check_reports_fresh.py`). Never rebuild a single
  PDF in isolation; never commit a `.tex` edit without its rebuilt PDF.
  Engine on this Mac: `tectonic` (in `~/.local/bin`).

## Metrics policy

chrF primary for MT, BLEU secondary; COMET excluded from headlines (learned
metric = pretrained model). Restoration: top-k recall, per-frequency-bin
breakdown, ECE, conformal sets. Decipherment claims require the permutation-
null p-value (Phase 6) — no exceptions once the harness exists.
