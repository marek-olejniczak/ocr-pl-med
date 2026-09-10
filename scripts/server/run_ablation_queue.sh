#!/usr/bin/env bash
# Train the OCR ablation LoRAs one after another on the RTX 3090.
#
# Run from the root of the ocr-pl-med checkout (branch `server`), inside tmux:
#     tmux new -s ablacja
#     bash scripts/server/run_ablation_queue.sh 2>&1 | tee training/results/ocr/surya/ablation_queue.log
#
# Hyperparameters are exactly those of the existing model surya_lora_ocr800k
# (adapter_config + meta.json): LoRA r=64, alpha=64, dropout 0.2, q/v/o_proj,
# lr 2e-5, weight decay 0.05, cosine, warmup 0.1, bf16, batch 8. Only the
# data and the step budget change between runs:
#     v2_800k  -> 40 000 steps (same as the baseline; this is the real model)
#     *_200k   -> 10 000 steps (quarter budget, same for every ablation variant)
#
# Each run is skipped if its adapter/ already exists, so the script can be
# restarted after an interruption. A failed run does not stop the queue.

set -u
cd "$(dirname "$0")/../.." || exit 1

DATA_ROOT="${DATA_ROOT:-dataset}"                       # dataset/<name>/images_shards + labels.jsonl
PROCESSED="${PROCESSED:-training/data/processed}"       # converted Surya-format data
RESULTS="${RESULTS:-training/results/ocr/surya}"
COMPOSE=(docker compose -f training/docker-compose.yml run --rm surya-training)
REPORT_TO="${REPORT_TO:-}"                              # set to "wandb" to log

# name  max_steps  eval_steps
RUNS=(
  "v2_800k   40000 5000"
  "v2_200k   10000 2500"
  "noA_200k  10000 2500"
  "noB_200k  10000 2500"
  "noC_200k  10000 2500"
)
# Restrict with: ONLY="noA_200k noB_200k" bash scripts/server/run_ablation_queue.sh
ONLY="${ONLY:-}"

mkdir -p "$RESULTS"
echo "== ablation queue start $(date -Is)"

for spec in "${RUNS[@]}"; do
  read -r name steps eval_steps <<<"$spec"
  if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then
    continue
  fi
  out="$RESULTS/$name-lora-r64"
  if [[ -d "$out/adapter" ]]; then
    echo "== $name: adapter exists, skipping"
    continue
  fi
  if [[ ! -f "$DATA_ROOT/$name/labels.jsonl" ]]; then
    echo "== $name: no data at $DATA_ROOT/$name (dvc pull?), skipping"
    continue
  fi

  echo "== $name: convert $(date -Is)"
  if [[ ! -f "$PROCESSED/$name/train/metadata.jsonl" ]]; then
    "${COMPOSE[@]}" python training/ocr/surya/convert_data.py ocr800k \
        --input "$DATA_ROOT/$name" --output "$PROCESSED/$name" \
        || { echo "== $name: convert FAILED"; continue; }
  fi

  echo "== $name: train $steps steps $(date -Is)"
  extra=()
  [[ -n "$REPORT_TO" ]] && extra=(--report-to "$REPORT_TO")
  "${COMPOSE[@]}" python training/ocr/surya/cli.py \
      --train-metadata "$PROCESSED/$name/train/metadata.jsonl" \
      --train-images-dir "$PROCESSED/$name/train/images" \
      --val-metadata "$PROCESSED/$name/val/metadata.jsonl" \
      --val-images-dir "$PROCESSED/$name/val/images" \
      --lora --lora-rank 64 --lora-alpha 64 --lora-dropout 0.2 \
      --lora-target-modules q_proj v_proj o_proj \
      --max-steps "$steps" --batch-size 8 --learning-rate 2e-5 --weight-decay 0.05 \
      --warmup-ratio 0.1 --lr-scheduler-type cosine \
      --evaluation-strategy steps --eval-steps "$eval_steps" --save-total-limit 3 \
      --run-name "$name-lora-r64" --output-dir "$out" "${extra[@]}" \
      || { echo "== $name: train FAILED"; continue; }
  echo "== $name: done $(date -Is)"
done

echo "== ablation queue end $(date -Is)"
