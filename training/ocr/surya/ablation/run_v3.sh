#!/usr/bin/env bash
# Generator v3 pilot in one go: seven trainings, then the benchmark of eight models.
# Run inside tmux, from anywhere:
#   bash ~/ocr-pl-med/training/ocr/surya/ablation/run_v3.sh
# Needs dataset/train/ocr/v3_base_200k, v3_200k and v3_no_*_200k (dvc pull).
# A finished training is skipped, so after an interruption just start it again.
set -u
cd "$(dirname "$0")/../../../.." || exit 1
ROOT=$(pwd)
A=training/ocr/surya/ablation

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

echo "== v3 start $(date -Is)"

echo "== etap 1/2: 7 treningow"
ONLY="v3_base_200k v3_200k v3_no_cursive_joins_200k v3_no_glyph_variants_200k v3_no_diacritic_placement_200k v3_base_200k_s43 v3_200k_s43" \
    bash "$A/run_ablation_queue.sh" 2>&1 | tee -a training/results/ocr/surya/ablation_queue.log

echo "== etap 2/2: benchmark 8 modeli na prawdziwych notatkach"
( cd benchmark && python -u autorunner.py --config "$ROOT/$A/experiments_v3.yaml" 2>&1 | tee wyniki/v3_pilot.log )

echo "== v3 koniec $(date -Is)"
ls -d benchmark/wyniki/ocr-v3-pilot_* 2>/dev/null
