#!/bin/bash
# Copy generated publication tables/figures into the manuscript folder and
# record checksums. Run after `python -m src.analysis.report_pub`.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=experiments/analysis/v3/report_pub
DST=paper/revision_v3/generated
rm -rf "$DST"; mkdir -p "$DST"
cp "$SRC"/*.tex "$SRC"/*.pdf "$SRC"/manifest.json "$DST"/
(cd "$DST" && shasum -a 256 *.tex *.pdf > SHA256SUMS)
echo "synced $(ls "$DST" | wc -l) files"
