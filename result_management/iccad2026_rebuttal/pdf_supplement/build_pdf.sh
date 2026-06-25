#!/usr/bin/env bash
# Build the rebuttal PDF supplement.
#
# Mode (C, hybrid): tables and the algorithm are individual .tex files;
# the body, captions, and structure live in supplement.md. pandoc converts
# the markdown to a LaTeX document and passes \input{...} directives
# through to xelatex, which inlines each .tex fragment.
#
# Usage:
#   ./build_pdf.sh                          # produces supplement.pdf
#   ./build_pdf.sh --keep-tex               # also keeps the intermediate .tex
#
# Requires: pandoc, xelatex (texlive-xetex), texlive package for
# booktabs + algorithm2e + graphicx + hyperref.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KEEP_TEX=0
for arg in "$@"; do
    case "$arg" in
        --keep-tex) KEEP_TEX=1 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

OUT_PDF="supplement.pdf"
OUT_TEX="supplement.tex"

# Sanity check the fragments exist
for f in A1_algorithm_extraction.tex \
         T1_input_parity.tex \
         T2_fg_vs_sa_seq.tex \
         T3_constraint_lut.tex \
         T4_all_corners.tex \
         T5_alignment_ablation.tex \
         T6_maml_effects.tex \
         assets/fg_vs_sa_seq.png; do
    if [ ! -e "$f" ]; then
        echo "missing fragment: $f" >&2
        exit 1
    fi
done

# pandoc: markdown + \input{...} -> latex -> xelatex -> PDF
pandoc supplement.md \
    --from markdown \
    --pdf-engine=xelatex \
    --output "$OUT_PDF"

if [ "$KEEP_TEX" -eq 1 ]; then
    pandoc supplement.md \
        --from markdown \
        --standalone \
        --output "$OUT_TEX"
    echo "wrote $OUT_TEX (intermediate)"
fi

echo "wrote $OUT_PDF"
