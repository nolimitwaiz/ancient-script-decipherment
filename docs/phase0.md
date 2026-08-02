# Phase 0 — scaffold, ledger, reproducibility (completed 2026-08-01)

Everything the spec's §2 requires exists, is tested, and is green:
`make check` = ruff lint + 49 pytest tests + no-pretrained gate + end-to-end
smoke, total wall time ≈ 5 s (smoke itself 0.02 s against a 5-minute budget).

## What exists

### 1. Repo scaffold
Layout per spec (configs/, glyphos/{data,render,models,tasks,align,quant,eval,
ledger,utils}, scripts/, tests/, docs/, contrib/openbook/), `uv` project pinned
to Python 3.12, ruff (E/F/W/I/UP/B/SIM/RUF, line 100), pytest, Makefile,
GitHub Actions CI (lint/test/gate/smoke). Phase-owned packages that arrive
later (render, models, quant, eval) carry docstring stubs stating what lands
when. Entry-point stubs (`prepare_data.py`, `train.py`, `evaluate.py`,
`align.py`) fail loudly with exit 2 rather than pretending.

### 2. Experiment ledger (`glyphos/ledger/`)
Append-only JSONL at `runs/ledger.jsonl` (committed; git `merge=union`).
Two-event lifecycle makes preregistration structural:

- `register` — REQUIRES non-empty `hypothesis`; records run_id, timestamp,
  git_hash (auto-detected, `-dirty` aware), phase, family, config_hash,
  data_version, split_version, seed, selection_metric, and
  `n_variants_tried_so_far_in_this_family` (counted from the ledger at
  registration — the multiple-testing count cannot be curated later).
- `complete` — exactly one terminal event per run:
  completed/failed/abandoned + `all_metrics`. Double-completion is an error;
  corrupt lines are hard errors; nothing is ever rewritten.
- `Ledger.run(...)` context manager guarantees crashed runs land in the
  ledger as `failed` with the exception in notes.
- CLI `glyphos-ledger {report,list,show,register,complete}` — register/
  complete exist so SLURM shell wrappers carry identical discipline.
- `report` implements the multiple-testing view: per family — variants tried,
  status counts, full distribution of the selection metric (best/median/worst
  + every value), and rank stability across seeds (mean pairwise Kendall
  tau-a between per-seed config rankings; implemented by hand, no scipy).

### 3. Seed management (`glyphos/utils/seed.py`)
`set_seed` covers python/numpy/torch (torch import-guarded until Phase 3);
`derive_seed(base, *labels)` for stable component seeds;
`require_seeds(seeds, minimum=3)` is the headline-number enforcement point.

### 4. Locked-test-set guard (`glyphos/data/guard.py`)
`install_guard()` patches `builtins.open` AND `io.open` (pathlib's
`Path.open` resolves through the latter), so any read of a path under
`data/**/test/` (or `test.*` file) is appended to `runs/test_access.jsonl`
with timestamp, active ledger run_id, path, mode, pid, and caller file:line.
Audit writes use raw `os.open/os.write` — recursion-proof. Log-only per spec;
freeze enforcement arrives with Phase 1 split manifests. A generic
`.../data/.../test/...` fallback covers cluster paths outside the repo.

### 5. Smoke pipeline (`make smoke`)
The smallest real instance of the artificial-decipherment protocol, in the
exact lifecycle shape all later phases must follow (config → guard → seed →
data hashes → register-with-hypothesis → fit/select/test → terminal event →
report): a 12-letter substitution cipher over a Zipfian toy language is
recovered by frequency-rank matching (`glyphos/align/freqmatch.py`, which
stays as the Phase 5 null baseline). Document-held-out split; test read goes
through the patched `open` and the pipeline **asserts its own audit entry
exists**. Toy data isolated under `runs/smoke/data/`; runs logged to the real
ledger (every run, no exceptions).

Result (seed 1337, deterministic): `key_accuracy = 1.000`,
`valid/test_token_accuracy = 1.000`, 1 audited test read, regression floor
0.5 asserted. See RESULTS.md.

### 6. Hard-constraint gate (`make check-no-pretrained`)
Greps `glyphos/ scripts/ tests/` for pretrained-loading APIs
(`from_pretrained`, `AutoModel*`, `torch.hub`, `hf_hub_download`,
`snapshot_download`, `transformers.pipeline`, `timm.create_model`);
`contrib/openbook/` is the only exclusion. Tested both directions: clean repo
passes, a planted violation fails, openbook is exempt. Wired into CI.

### 7. Report build system (`make reports`)
Rebuilds ALL `docs/reports/**/*.tex` in one pass (latexmk > tectonic >
pdflatex; this machine: tectonic 0.17, installed to `~/.local/bin` since no
TeX existed), then `check_reports_fresh.py` verifies each PDF exists, is not
older than its source, and contains the source's `\ReportStamp` (extracted
with pypdf). Convention: bump the stamp on every substantive edit.

## Tests

49 tests, all state-isolated (tmp data root/runs dir per test, guard
uninstalled after each): ledger lifecycle + corruption + counter, report
distributions + rank stability (agreement and disagreement cases), guard
(both open routes, writes ignored, run attribution, restore-on-uninstall),
seeds, hashing (order-invariance, content sensitivity), strict configs, toy
corpus determinism + split disjointness + cipher consistency, freq-matching
correctness + beats-chance, CLI round-trip, and the no-pretrained gate
(clean/violation/exempt). `uv run pytest`: 49 passed in 0.24 s.

## Deviations from spec

None of substance. Two additions: smoke logs to the real ledger (stricter
reading of "no exceptions"), and the report/PDF discipline requested on top of
the spec (stamp + rebuild-all + freshness gate). Torch intentionally deferred
to Phase 3 dependencies.

## Next: Phase 1

Ingest TLA (all subsets, provenance), Coptic SCRIPTORIUM, LogogramNLP,
Ugaritic+Hebrew; census table; six split schemes with frozen, hashed test
partitions; guard freeze enforcement. Known gaps already filed in
docs/data_gaps.md (DĀMOS/LiBER access, Ugaritic source location).
