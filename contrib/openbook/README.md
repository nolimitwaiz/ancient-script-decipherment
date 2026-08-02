# contrib/openbook — the only place pretrained models may ever live

**Sealed vs. open-book:** the entire scientific pipeline (`glyphos/`, `scripts/`,
`tests/`) is *sealed*: every model parameter is randomly initialized and trained
inside this repo, on data whose provenance is recorded in the census. This is what
makes the artificial-decipherment validation meaningful — a component pretrained
on the open web may have seen Ugaritic/Linear B scholarship, which is lookahead
bias, and any "decipherment" it produces is void.

This directory is the future home of explicitly *open-book* comparisons
(e.g. "how does a pretrained-PIXEL translator compare to our sealed one?"),
where contamination is acknowledged and the numbers are labeled as such.
They are context, never headline results.

Rules:
- Nothing under `glyphos/`, `scripts/`, or `tests/` may import from here.
- `make check-no-pretrained` greps the sealed tree and fails the build on any
  pretrained-weight loading API; this directory is its only exclusion.
- Currently empty by design (Phase 8 scaffold).
