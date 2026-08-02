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
