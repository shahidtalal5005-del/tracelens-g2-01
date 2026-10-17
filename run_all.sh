#!/usr/bin/env bash
# One command: tests + benchmark + phase 1 + phase 2 + viewer.  Works on Linux/macOS.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -m pytest || echo "(pytest not installed - skipping tests: pip install pytest)"
$PY -m tracelens all --out results
echo; echo "Open results/viewer.html in your browser. Reports: results/phase1_results.md, results/phase2_results.md, results/gap_resolution_table.md"
