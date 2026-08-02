# data_gaps.md — known blockers and stubs (fail loudly, never stall)

Format: status · gap · what exists instead · unblock path.

## Open

- **OPEN · Linear B (DĀMOS / LiBER) access method unknown.** Phase 1 will ship
  the loader interface with a stubbed download that raises with this file's
  URL. Unblock: inspect DĀMOS (Oslo) and LiBER (CNR) export options; likely
  scraping-with-permission or a request to maintainers.
- **OPEN · Ugaritic corpus provenance.** Spec says to check Luo et al. 2019's
  released code/data first (their GitHub) before falling back to other
  transliterated sources. Unblock: locate release during Phase 1 ingest.
- **OPEN · Hebrew Bible export choice.** ETCBC/BHSA (text-fabric format) vs
  other openly licensed machine-readable texts; license and transliteration
  scheme need a Phase 1 decision entry.
- **OPEN · Frontier scripts (Meroitic REM, Mayan, Libyco-Berber).** Loader
  stubs only; explicitly not Phase 1 blockers.
- **OPEN · CI has no LaTeX.** `make reports` runs locally (tectonic); CI runs
  lint/test/gate/smoke only. Acceptable: PDFs are committed and the freshness
  gate runs locally. Revisit if reports start drifting.

## Resolved

- **RESOLVED 2026-08-01 · No TeX toolchain on the dev Mac.** Installed
  tectonic 0.17.0 to `~/.local/bin` (user-level, no admin).

## Phase 1 additions (2026-08-02)

- **OPEN · TLA text IDs.** Premium exports carry no text/document IDs and are
  not text-ordered; doc-held-out uses dating cohorts (conservative). Unblock:
  ask TLA maintainers for an export with text IDs (or map sentences via the
  TLA web API), then mint new split tags.
- **OPEN · LogogramNLP cuneiform/Akkadian absent** from the repo's data/
  (only LNA/EGY/ZHO). Unblock: check the paper's release scripts for external
  download links when building Phase 7 comparison hooks.
- **OPEN · Coptic translation coverage.** text_en exists for only part of
  SCRIPTORIUM; fine for known-relative use, insufficient for a Coptic MT task.
- **RESOLVED 2026-08-02 · Ugaritic corpus provenance** — Luo et al.'s
  NeuroDecipher release ships uga-heb cognate data (43,951 pairs) AND
  linear_b-greek.cog (919 pairs): ladder rungs 1 and 2 both have their
  published-comparison data. Full DĀMOS/LiBER corpus remains OPEN above.
- **RESOLVED 2026-08-02 · Hebrew Bible export choice** — OpenScriptures
  morphhb (WLC, CC BY 4.0); see decisions.
