#!/usr/bin/env bash
# Ablation round 2 in one go: robustness benchmark, nine trainings, family benchmark.
# Run inside tmux, from anywhere:
#   bash ~/ocr-pl-med/training/ocr/surya/ablation/run_round2.sh
# Each stage logs on its own; a failing stage does not stop the next one.
set -u
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$(pwd)
A=training/ocr/surya/ablation

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

echo "== runda 2 start $(date -Is)"

echo "== etap 1/3: odczyt zbiorow kontrolnych (v1, v2)"
( cd benchmark && python -u autorunner.py --config "$ROOT/$A/experiments_odpornosc.yaml" 2>&1 | tee wyniki/runda2_odpornosc.log )

echo "== etap 2/3: 9 treningow"
bash "$A/run_ablation_queue.sh" 2>&1 | tee -a training/results/ocr/surya/ablation_queue.log

echo "== etap 3/3: benchmark 10 modeli na prawdziwych notatkach"
( cd benchmark && python -u autorunner.py --config "$ROOT/$A/experiments_rodziny.yaml" 2>&1 | tee wyniki/runda2_rodziny.log )

echo "== runda 2 koniec $(date -Is)"
ls -d benchmark/wyniki/ocr-odpornosc_* benchmark/wyniki/ocr-ablacja-rodziny_* 2>/dev/null
