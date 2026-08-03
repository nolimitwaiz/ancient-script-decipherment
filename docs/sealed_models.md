# Sealed known-relative language models (§ Phase 3.4)

The decipherment track (Phase 5+) may consume ONLY the language models listed
here. Every one is a character-level Transformer (~10–25M params at the
default CharLMConfig) trained from scratch in this repo; the exact training
data is pinned by corpus `data_version` so contamination claims are checkable.

| model | training corpus | data_version | split | status |
|---|---|---|---|---|
| sealed-hebrew | hebrew_morphhb (WLC verses) | ffee0c50b1e9 | dedup/v1 train | pending cluster run |
| sealed-greek | greek_first1k (grc passages) | 0bf0e2da23cb | dedup/v1 train | pending cluster run |
| sealed-coptic | coptic_scriptorium | 4444354f3147 | dedup/v1 train | pending cluster run |
| sealed-german | TLA translation_de fields ONLY (target side) | 62eace20bffb + a287ffa107c2 + 08b86674d85a | dedup/v1 train | pending cluster run |

Rules:
- Training text comes exclusively from the listed corpus/split train
  partitions — never from test/valid, never from outside the repo's data
  pipeline.
- Checkpoints are stored with their ledger run_id; a sealed model without a
  ledger entry does not exist as far as Phase 5 is concerned.
- The quarantined Meroitic release embeddings (see meroitic_rem notes) are
  NOT sealed models and are never loaded.
