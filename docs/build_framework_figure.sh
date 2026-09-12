#!/usr/bin/env bash

set -euo pipefail

if ! command -v latexmk >/dev/null 2>&1; then
    echo "latexmk is required to build docs/assets/framework_overview.tex" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_DIR="${ROOT_DIR}/docs/assets"
STEM="framework_overview"
SOURCE_TEX="${ASSET_DIR}/${STEM}.tex"
OUTPUT_PDF="${ASSET_DIR}/${STEM}.pdf"
OUTPUT_PNG="${ASSET_DIR}/${STEM}.png"

if [[ ! -f "${SOURCE_TEX}" ]]; then
    echo "Missing source file: ${SOURCE_TEX}" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

cp "${SOURCE_TEX}" "${WORK_DIR}/${STEM}.tex"

pushd "${WORK_DIR}" >/dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error "${STEM}.tex" >/dev/null
if command -v gs >/dev/null 2>&1; then
    gs \
        -dSAFER \
        -dBATCH \
        -dNOPAUSE \
        -dTextAlphaBits=4 \
        -dGraphicsAlphaBits=4 \
        -sDEVICE=pngalpha \
        -r216 \
        -sOutputFile="${STEM}.png" \
        "${STEM}.pdf" >/dev/null
elif command -v pdftoppm >/dev/null 2>&1; then
    pdftoppm -png -singlefile -r 216 "${STEM}.pdf" "${STEM}" >/dev/null
elif command -v sips >/dev/null 2>&1; then
    sips -s format png "${STEM}.pdf" --out "${STEM}.png" >/dev/null
else
    echo "Unable to generate PNG output: install Ghostscript, pdftoppm, or use macOS sips." >&2
    exit 1
fi
popd >/dev/null

cp "${WORK_DIR}/${STEM}.pdf" "${OUTPUT_PDF}"
cp "${WORK_DIR}/${STEM}.png" "${OUTPUT_PNG}"
