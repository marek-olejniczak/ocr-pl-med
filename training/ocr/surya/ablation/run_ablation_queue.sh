#!/usr/bin/env bash
# Train the OCR ablation LoRAs one after another on the RTX 3090.
#
# Run from the root of the ocr-pl-med checkout (branch `server`), inside tmux:
#     tmux new -s ablacja
#     bash training/ocr/surya/ablation/run_ablation_queue.sh 2>&1 | tee training/results/ocr/surya/ablation_queue.log
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
cd "$(dirname "$0")/../../../.." || exit 1

DATA_ROOT="${DATA_ROOT:-dataset}"                       # dataset/<name>/images_shards + labels.jsonl
PROCESSED="${PROCESSED:-training/data/processed}"       # converted Surya-format data
RESULTS="${RESULTS:-training/results/ocr/surya}"
COMPOSE=(docker compose -f training/docker-compose.yml run --rm surya-training)
REPORT_TO="${REPORT_TO:-}"                              # set to "wandb" to log

# name  max_steps  eval_steps  [seed  [dataset, if not the name]]
RUNS=(
  "v2_800k   40000 5000"
  "v2_200k   10000 2500"
  "noA_200k  10000 2500"
  "noB_200k  10000 2500"
  "noC_200k  10000 2500"
  # second round (2026-10): leave one family out, reference v2_200k
  "no_short_words_200k 10000 2500"
  "no_caps_200k 10000 2500"
  "no_arrows_bullets_200k 10000 2500"
  "no_anatomy_vocab_200k 10000 2500"
  "no_neighbour_glyphs_200k 10000 2500"
  "no_grid_paper_200k 10000 2500"
  "no_morphology_200k 10000 2500"
  "no_elastic_200k 10000 2500"
  "no_phone_photo_200k 10000 2500"
  # generator v3 pilot (2026-11). The five sets share every line except the
  # glyph families; v3_base_200k is the reference. Seed 42, and the reference
  # and full v3 once more with seed 43 to see how far two trainings on the
  # same data drift apart.
  "v3_base_200k 10000 2500 42"
  "v3_200k 10000 2500 42"
  "v3_no_cursive_joins_200k 10000 2500 42"
  "v3_no_glyph_variants_200k 10000 2500 42"
  "v3_no_diacritic_placement_200k 10000 2500 42"
  "v3_base_200k_s43 10000 2500 43 v3_base_200k"
  "v3_200k_s43 10000 2500 43 v3_200k"
)
# Restrict with: ONLY="noA_200k noB_200k" bash training/ocr/surya/ablation/run_ablation_queue.sh
ONLY="${ONLY:-}"

mkdir -p "$RESULTS"
echo "== ablation queue start $(date -Is)"

for spec in "${RUNS[@]}"; do
  read -r name steps eval_steps seed data <<<"$spec"
  data="${data:-$name}"
  if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then
    continue
  fi
  out="$RESULTS/$name-lora-r64"
  if [[ -d "$out/adapter" ]]; then
    echo "== $name: adapter exists, skipping"
    continue
  fi
  if [[ ! -f "$DATA_ROOT/$data/labels.jsonl" ]]; then
    echo "== $name: no data at $DATA_ROOT/$data (dvc pull?), skipping"
    continue
  fi

  # A tar cut short in transit makes convert_data.py skip the whole shard
  # ("Pominieto niekompletny shard") and train on a fraction of the data
  # without ever failing. Check the archives before spending a day on them.
  bad=0
  for archive in "$DATA_ROOT/$data/images_shards/"*.tar; do
    tar -tf "$archive" >/dev/null 2>&1 || { echo "== $name: USZKODZONY shard $archive"; bad=1; }
  done
  if [[ $bad -eq 1 ]]; then
    echo "== $name: skipping, re-copy the shards first"
    continue
  fi

  echo "== $name: convert $(date -Is)"
  if [[ ! -f "$PROCESSED/$data/train/metadata.jsonl" ]]; then
    "${COMPOSE[@]}" python training/ocr/surya/convert_data.py ocr800k \
        --input "$DATA_ROOT/$data" --output "$PROCESSED/$data" \
        || { echo "== $name: convert FAILED"; continue; }
  fi

  # Converted data left over from an earlier, broken run is reused silently,
  # so compare what landed on disk with what the labels promise.
  want=$(wc -l < "$DATA_ROOT/$data/labels.jsonl")
  got=$(( $(wc -l < "$PROCESSED/$data/train/metadata.jsonl") + $(wc -l < "$PROCESSED/$data/val/metadata.jsonl") ))
  if (( got < want * 99 / 100 )); then
    echo "== $name: NIEPELNE dane - $got z $want probek; usun $PROCESSED/$data i powtorz"
    continue
  fi
  echo "== $name: probek gotowych: $got/$want"

  echo "== $name: train $steps steps $(date -Is)"
  extra=()
  [[ -n "$REPORT_TO" ]] && extra=(--report-to "$REPORT_TO")
  [[ -n "$seed" ]] && extra+=(--seed "$seed")
  "${COMPOSE[@]}" python training/ocr/surya/cli.py \
      --train-metadata "$PROCESSED/$data/train/metadata.jsonl" \
      --train-images-dir "$PROCESSED/$data/train/images" \
      --val-metadata "$PROCESSED/$data/val/metadata.jsonl" \
      --val-images-dir "$PROCESSED/$data/val/images" \
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
