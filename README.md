# GLYPHOS

From-scratch visual decipherment research stack: pixel-representation MT
(Salesky/Koehn lineage), OT/min-cost-flow decipherment (Luo & Barzilay lineage),
and a quant-derived statistical epistemics layer — validated by *artificial
decipherment* (hide the key of an already-deciphered script, measure recovery).

Graduate research project, JHU CLSP (advisor: Philipp Koehn).

## Hard constraints (see CONVENTIONS.md — never violate)

1. **No pretrained model weights, anywhere.** Every parameter is randomly
   initialized and trained in this repo (`make check-no-pretrained` enforces it).
   Pretrained *datasets* are allowed; pretrained *models* are not.
2. **Every run is preregistered in the experiment ledger** — hypothesis first,
   results second — including failed and abandoned runs.
3. **Locked test sets.** Test splits are frozen and every read is audit-logged.
4. **Reproducibility.** Explicit seeds (3+ for headline numbers), YAML configs,
   content-hashed data versions, deterministic where feasible.

## Quickstart

```bash
uv sync          # env (Python 3.12, pinned via .python-version)
make check       # lint + tests + no-pretrained gate + end-to-end smoke
make smoke       # toy pipeline on CPU (<5 min budget, currently ~1 s)
uv run glyphos-ledger report   # multiple-testing-aware experiment report
make reports     # rebuild ALL LaTeX report PDFs + staleness gate
```

Development is local (CPU smoke + unit tests only); all real training runs on
the CLSP cluster (3x A100, SLURM).

## Layout

```
configs/          YAML configs (data, model, train, eval, align)
glyphos/
  data/           loaders, splits, dedup, census, locked-test-set guard
  render/         text->image engine + degradation suite        (Phase 2)
  models/         pixel encoder, transformer, BPE baseline, sealed LMs (Phase 3)
  tasks/          translation, restoration, auxiliary heads     (Phase 4)
  align/          OT/min-cost-flow decipherment core, priors    (Phase 5)
  quant/          shrinkage, bootstrap, permutation nulls, conformal (Phase 6)
  eval/           metrics, calibration, artificial-decipherment protocol
  ledger/         append-only experiment ledger + report CLI
scripts/          entry points + CI gates + report builder
tests/            pytest suite (state-isolated; guard-safe)
docs/             phase reports (md + LaTeX + PDF), decisions, gaps, roadmap
contrib/openbook/ the ONLY place pretrained models may ever live (empty)
runs/             ledger.jsonl (committed, union-merged) + local artifacts
data/             corpora (never committed; census hashes are)
```

Orientation files: `CONTEXT.md` (what/why), `CONVENTIONS.md` (rules of work),
`MEMORY.md` (current state), `RESULTS.md` (ledger-backed results index).
