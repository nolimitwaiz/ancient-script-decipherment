# architecture.md — dataset pipeline & system dataflow

## Dataflow

```mermaid
flowchart TD
    subgraph sources [Phase 1 - sources]
        TLA[TLA HF datasets<br/>hieroglyphs+translit+DE]
        COP[Coptic SCRIPTORIUM]
        LOG[LogogramNLP]
        UGA[Ugaritic + Hebrew]
        LINB[Linear B + Greek - stub]
    end
    sources --> ING[ingest: normalize + provenance<br/>data_version = content hash]
    ING --> CEN[census: counts, licenses,<br/>encodings, translation targets]
    ING --> SPL[splits: random / document_heldout /<br/>dedup / period / site / sign_heldout<br/>test partitions hashed + FROZEN]
    SPL -->|test reads audited| GUARD[(guard log<br/>runs/test_access.jsonl)]
    SPL --> REN[Phase 2 render engine<br/>PangoCairo+Noto, 24x24 win, stride 12<br/>+ degradation suite, damage_level]
    REN --> STREAM[multiscript SSL stream<br/>unbounded, we own the generator]
    REN --> MODELS
    SPL --> MODELS[Phase 3 models - all from scratch:<br/>pixel enc-dec 12/3 d512 · BPE control<br/>masked-window SSL · sealed char-LMs]
    MODELS --> TASKS[Phase 4 tasks:<br/>Egyptian->German MT · restoration<br/>+ conformal sets · aux heads]
    MODELS --> ALIGN[Phase 5 decipherment:<br/>Sinkhorn OT + min-cost flow<br/>monotonicity/sparsity/freq priors<br/>+ IPA prior + visual prior]
    TASKS --> QUANT[Phase 6 quant layer:<br/>shrinkage · block bootstrap ·<br/>Ledoit-Wolf · permutation nulls]
    ALIGN --> QUANT
    QUANT --> EVAL[Phase 7 evaluate.py:<br/>regenerates every table<br/>from ledger + frozen tests]
    LEDGER[(runs/ledger.jsonl<br/>register BEFORE results)] -.-> MODELS
    LEDGER -.-> TASKS
    LEDGER -.-> ALIGN
    EVAL --> DOCS[docs/phaseN.md · RESULTS.md ·<br/>LaTeX+PDF reports]
```

## Data directory contract (normative — the guard depends on it)

```
data/
  <corpus>/                      # e.g. tla_earlier_egyptian, coptic_scriptorium
    raw/                         # exactly as downloaded, never modified
    processed/<data_version>/    # normalized jsonl; dir hash = data_version
    splits/<scheme>/<version>/   # scheme in {random, document_heldout, dedup,
      train/                     #   period_heldout, site_heldout, sign_heldout}
      valid/
      test/                      # <- every read below a test/ dir is audited
    private/                     # keys/answer material (artificial decipherment)
```

- `data/` is gitignored; identity lives in content hashes recorded in the
  census and in every ledger entry (`data_version`, `split_version`).
- Test partitions are frozen at split time; Phase 1 adds a manifest the guard
  will enforce against (Phase 0 guard is audit-only).
- Render cache (Phase 2) is keyed by `(text_hash, render_config_hash)` and is
  regenerable — never data of record.

## Run lifecycle (every phase, same shape — established by the smoke pipeline)

1. resolve config (strict dataclass) → `config_hash`
2. `guard.install_guard()`; `set_seed(cfg.seed)`
3. materialize/verify data → `data_version`, `split_version`
4. `ledger.register(hypothesis, family, ...)` — BEFORE any result
5. train / fit on `train`, select on `valid`
6. score on locked `test` (audited), attach `all_metrics`
7. terminal ledger event (`completed`/`failed`/`abandoned`)
8. `glyphos-ledger report` — winner only reportable with its family
   distribution

Reference implementation: `glyphos/tasks/smoke.py`.
