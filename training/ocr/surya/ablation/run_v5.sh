#!/usr/bin/env bash
# Generator v5 in one go: two trainings on v5_200k (10k and 25k steps), then
# the benchmark of both on the real test set v2.
# Run inside tmux, from anywhere:
#   bash ~/ocr-pl-med/training/ocr/surya/ablation/run_v5.sh
# Needs dataset/train/ocr/v5_200k (dvc pull) and benchmark/dane/real_v2
# (prepare_real_v2.sh). A finished training is skipped.
set -u
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$(pwd)
A=training/ocr/surya/ablation

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

echo "== v5 start $(date -Is)"
[[ -f benchmark/dane/real_v2/labels.csv ]] || bash "$A/prepare_real_v2.sh" || exit 1

echo "== etap 1/2: 2 treningi"
ONLY="v5_200k v5_200k_25k" bash "$A/run_ablation_queue.sh" 2>&1 | tee -a training/results/ocr/surya/ablation_queue.log

echo "== etap 2/2: benchmark 2 modeli na testsecie v2"
( cd benchmark && python -u autorunner.py --config "$ROOT/$A/experiments_v5.yaml" 2>&1 | tee wyniki/v5.log )

echo "== v5 koniec $(date -Is)"
ls -d benchmark/wyniki/ocr-v5-real-v2_* 2>/dev/null
