#!/usr/bin/env bash
# Pełny, wyczerpujący trening Surya LoRA na v2_200k — 8 epok, duży batch.
#
# Uruchom z roota checkoutu, w tmux z WANDB_API_KEY w środowisku:
#     tmux new -s surya
#     export WANDB_API_KEY=...            # jeśli nie ma go już w sesji
#     bash training/ocr/surya/mega/run_mega.sh 2>&1 \
#         | tee training/results/ocr/surya/mega/queue.log
#
# USTALENIA (zmierzone na 2x RTX A6000):
#   * Trening na JEDNEJ karcie (CUDA_VISIBLE_DEVICES=0). Gdy kontener widzi
#     obie, HF Trainer włącza DataParallel, a custom collator zwija
#     image_embeddings PŁASKO (nie per-sample) -> DataParallel tnie je źle
#     i leci `masked_scatter_size_check` / CUDA device-side assert.
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True: przy batch 96 peak
#     VRAM spada z ~48 GB do ~37 GB (bez tego 96 jest na granicy OOM).
#   * batch 96 = 84 próbek/s (najlepszy zmierzony; 64 -> 58, 128 -> 77).
#   * eval CER na DRUGIEJ karcie (GPU 1) przez subprocess .../hp_search/eval_cer.py;
#     wybór checkpointu po `eval_cer` (eval_loss jest w tym zadaniu zawodny).
#
# 189 809 próbek train / batch 96 = 1977 kroków na epokę -> 8 epok = 15 800 kroków.
# Szacowany czas: ~5 h treningu + ~0.8 h evalów.

set -euo pipefail
cd "$(dirname "$0")/../../../.."

DATA="${DATA:-training/data/processed/v2_200k}"
OUT="${OUT:-training/results/ocr/surya/v2_200k-lora-r64-mega}"
DATASET="${DATASET:-dataset/v2_200k}"

COMPOSE=(docker compose -f training/docker-compose.yml run --rm \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    surya-training)

mkdir -p "$(dirname "$OUT")"

if [[ ! -f "$DATA/train/metadata.jsonl" ]]; then
    echo "== konwersja $DATASET -> $DATA $(date -Is)"
    "${COMPOSE[@]}" python training/ocr/surya/convert_data.py ocr800k \
        --input "$DATASET" --output "$DATA"
fi

for f in "$DATA/train/metadata.jsonl" "$DATA/val/metadata.jsonl" \
         benchmark/dane/handlabeled/labels.csv; do
    if [[ ! -e "$f" ]]; then
        echo "== BŁĄD: brak $f" >&2
        exit 1
    fi
done

echo "== trening start $(date -Is)"
"${COMPOSE[@]}" python training/ocr/surya/cli.py \
    --train-metadata "$DATA/train/metadata.jsonl" \
    --train-images-dir "$DATA/train/images" \
    --val-metadata "$DATA/val/metadata.jsonl" \
    --val-images-dir "$DATA/val/images" \
    --lora --lora-rank 64 --lora-alpha 64 --lora-dropout 0.2 \
    --lora-target-modules q_proj v_proj o_proj \
    --batch-size 96 --max-steps 15800 \
    --learning-rate 2e-5 --weight-decay 0.05 \
    --warmup-ratio 0.1 --lr-scheduler-type cosine --bf16 \
    --evaluation-strategy steps --eval-steps 1977 --save-total-limit 100 \
    --cer-callback --cer-eval-gpu 1 --cer-eval-batch-size 16 \
    --report-to wandb --run-name surya-v2_200k-lora-r64-mega \
    --output-dir "$OUT"

echo "== trening koniec $(date -Is)"
