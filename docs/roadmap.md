# roadmap.md — phase plan and status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, experiment ledger, seeds, test-set guard, smoke, gates | ✅ done 2026-08-01 (docs/phase0.md) |
| 1 | Data: TLA/Coptic/LogogramNLP/Ugaritic-Hebrew ingest, census, 6 split schemes, frozen+hashed tests | ✅ done 2026-08-02 (docs/phase1.md) |
| 2 | Render engine (PangoCairo/Noto, 24×24/12), degradation suite, multiscript SSL stream | ⬜ next |
| 3 | Models from scratch: pixel enc-dec (12/3, d512), BPE control, masked-window SSL, sealed char-LMs | ⬜ |
| 4 | Tasks + evaluation: chrF/BLEU, restoration (damage-shaped masks), conformal sets, document bootstrap CIs | 🔵 IN PROGRESS — metrics/restoration/conformal/bootstrap built + tested; model-side wiring next |
| 5 | Decipherment core: Sinkhorn OT + min-cost flow, monotonicity/sparsity/freq + IPA + visual priors; ladder Ugaritic→Hebrew, Linear B→Greek, Egyptian→Coptic | ⬜ |
| 6 | Quant layer: hierarchical shrinkage, block bootstrap, Ledoit-Wolf, permutation nulls + wrong-relative negative control, multiple-testing report, tail reporting | ⬜ |
| 7 | Evaluation packaging: evaluate.py regenerates all tables; capacity/data-efficiency/damage sweeps; LogogramNLP comparison hooks | ⬜ |
| 3b | SSL arms: A1 MAE (running) → **A2 I-JEPA / A3 DINO GATED** on Phase 4 evidence that translation is worth optimizing at this scale | ⏸ gated (docs/experiments/ssl_and_graph_arms.md) |
| 5b | **Graph decipherment arm** (runs in PARALLEL — CPU only, and it is the thesis): co-occurrence graphs → Gromov-Wasserstein (rung 1) → GIN + Sinkhorn OT (rung 2) → + phonetic/visual priors. Ventris's grid method, formalised. CPU-cheap, runs in parallel | ⬜ next (same doc) |
| 8 | Deferred scaffolds: GUI backend skeleton, contrib/openbook, frontier-script stubs | ⬜ (openbook README exists) |

Gate protocol: a phase is done when its `docs/phaseN.md` + LaTeX/PDF report
exist and `make check` is green. Blockers → stub + docs/data_gaps.md, keep
moving.
