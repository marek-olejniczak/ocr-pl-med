#!/usr/bin/env bash
# Cały pipeline: inferencja (GPU, kontener) + analizy (CPU, host).
#   bash training/ocr/error_analysis/run_all.sh
#   PYTHON=.venv/bin/python SKIP_INFER=1 bash training/ocr/error_analysis/run_all.sh
set -euo pipefail

TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

[ "${SKIP_INFER:-0}" = 1 ] || bash "$TOOL/infer.sh"
"$PYTHON" "$TOOL/analyze.py"
"$PYTHON" "$TOOL/analyze_ngram.py"
"$PYTHON" "$TOOL/diacritics.py" model-input
echo "== gotowe: $TOOL/out, $TOOL/out_ngram"
