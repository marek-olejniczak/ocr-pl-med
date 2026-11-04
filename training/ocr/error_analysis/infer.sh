#!/usr/bin/env bash
# Krok 1: inferencja adaptera Surya na handlabeled -> data/predictions.jsonl + data/metrics.json.
# Reużywa hp_search/eval_cer.py (parity z benchmarkiem).
#
#   bash training/ocr/error_analysis/infer.sh
#   ADAPTER=training/results/ocr/surya/<run>/checkpoint-N GPU=1 bash training/ocr/error_analysis/infer.sh
set -euo pipefail

TOOL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$TOOL/../../.." && pwd)"
REL="${TOOL#"$REPO"/}"   # ścieżka narzędzia względem repo (= względem /app w kontenerze)
ADAPTER="${ADAPTER:-training/results/ocr/surya/v2_200k-lora-r64-mega/adapter}"
GPU="${GPU:-0}"
BATCH="${BATCH:-16}"

mkdir -p "$TOOL/data"
cd "$REPO"

echo "== inferencja: adapter=$ADAPTER  GPU=$GPU  batch=$BATCH  $(date -Is)"
docker compose -f training/docker-compose.yml run --rm \
    -e "CUDA_VISIBLE_DEVICES=$GPU" \
    surya-training python training/ocr/surya/hp_search/eval_cer.py \
        --adapter "$ADAPTER" \
        --output-json "$REL/data/metrics.json" \
        --dump-predictions "$REL/data/predictions.jsonl" \
        --batch-size "$BATCH"

echo "== gotowe: $TOOL/data/predictions.jsonl  $(date -Is)"
