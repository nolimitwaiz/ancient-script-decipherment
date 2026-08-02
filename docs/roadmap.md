# roadmap.md — phase plan and status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, experiment ledger, seeds, test-set guard, smoke, gates | ✅ done 2026-08-01 (docs/phase0.md) |
| 1 | Data: TLA/Coptic/LogogramNLP/Ugaritic-Hebrew ingest, census, 6 split schemes, frozen+hashed tests | ✅ done 2026-08-02 (docs/phase1.md) |
| 2 | Render engine (PangoCairo/Noto, 24×24/12), degradation suite, multiscript SSL stream | ⬜ next |
| 3 | Models from scratch: pixel enc-dec (12/3, d512), BPE control, masked-window SSL, sealed char-LMs | ⬜ |
| 4 | Tasks: Egyptian→German MT (chrF primary), Ithaca-style restoration + conformal sets, aux heads | ⬜ |
| 5 | Decipherment core: Sinkhorn OT + min-cost flow, monotonicity/sparsity/freq + IPA + visual priors; ladder Ugaritic→Hebrew, Linear B→Greek, Egyptian→Coptic | ⬜ |
| 6 | Quant layer: hierarchical shrinkage, block bootstrap, Ledoit-Wolf, permutation nulls + wrong-relative negative control, multiple-testing report, tail reporting | ⬜ |
| 7 | Evaluation packaging: evaluate.py regenerates all tables; capacity/data-efficiency/damage sweeps; LogogramNLP comparison hooks | ⬜ |
| 8 | Deferred scaffolds: GUI backend skeleton, contrib/openbook, frontier-script stubs | ⬜ (openbook README exists) |

Gate protocol: a phase is done when its `docs/phaseN.md` + LaTeX/PDF report
exist and `make check` is green. Blockers → stub + docs/data_gaps.md, keep
moving.
