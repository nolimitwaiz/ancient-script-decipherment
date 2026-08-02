#!/usr/bin/env bash
# Rebuild ALL report PDFs in one pass — never a single document in isolation,
# so no committed PDF can go stale relative to its .tex source.
# Toolchain preference: latexmk > tectonic > pdflatex. Fails loudly if none.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

mains=$(find docs/reports -name '*.tex' -exec grep -l '\\documentclass' {} \; 2>/dev/null | sort)
if [ -z "$mains" ]; then
    echo "build_reports: no report .tex files under docs/reports/ yet"
    exit 0
fi

if command -v latexmk >/dev/null 2>&1; then
    tool=latexmk
elif command -v tectonic >/dev/null 2>&1; then
    tool=tectonic
elif command -v pdflatex >/dev/null 2>&1; then
    tool=pdflatex
else
    echo "ERROR: no LaTeX engine found (latexmk/tectonic/pdflatex)." >&2
    echo "Reports MUST be rebuilt after every .tex edit; install tectonic:" >&2
    echo "  https://tectonic-typesetting.github.io" >&2
    exit 1
fi

echo "build_reports: using $tool"
for tex in $mains; do
    dir=$(dirname "$tex")
    base=$(basename "$tex")
    echo "  -> $tex"
    case $tool in
        latexmk)  (cd "$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error "$base" >/dev/null) ;;
        tectonic) (cd "$dir" && tectonic --chatter minimal "$base") ;;
        pdflatex) (cd "$dir" && pdflatex -interaction=nonstopmode -halt-on-error "$base" >/dev/null \
                             && pdflatex -interaction=nonstopmode -halt-on-error "$base" >/dev/null) ;;
    esac
done
echo "build_reports: rebuilt all $(echo "$mains" | wc -l | tr -d ' ') report(s)"
