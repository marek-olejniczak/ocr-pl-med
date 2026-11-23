#!/usr/bin/env bash
# Generator v4 in one go: two trainings on v4_200k (10k and 25k steps), then
# the benchmark of seven models on the real notes v1.2.
# Run inside tmux, from anywhere:
#   bash ~/ocr-pl-med/training/ocr/surya/ablation/run_v4.sh
# Needs dataset/train/ocr/v4_200k and dataset/test/real/v1.2 (dvc pull).
# A finished training is skipped, so after an interruption just start it again.
set -u
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$(pwd)
A=training/ocr/surya/ablation

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

echo "== v4 start $(date -Is)"

echo "== etap 1/3: testset v1.2"
bash "$A/prepare_real_v12.sh" || exit 1

echo "== etap 2/3: 2 treningi"
ONLY="v4_200k v4_200k_25k" bash "$A/run_ablation_queue.sh" 2>&1 | tee -a training/results/ocr/surya/ablation_queue.log

echo "== etap 3/3: benchmark 7 modeli na prawdziwych notatkach v1.2"
( cd benchmark && python -u autorunner.py --config "$ROOT/$A/experiments_v4.yaml" 2>&1 | tee wyniki/v4.log )

echo "== v4 koniec $(date -Is)"
ls -d benchmark/wyniki/ocr-v4_* 2>/dev/null
