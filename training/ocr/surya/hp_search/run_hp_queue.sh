#!/usr/bin/env bash
# Sweep hiperparametrów LoRA Suryi na v2_200k (runda 1a: learning rate × rank).
#
# Run z roota checkoutu na serwerze, w tmux:
#     tmux new -s hp
#     bash training/ocr/surya/hp_search/run_hp_queue.sh 2>&1 \
#         | tee -a training/results/ocr/surya/hp/queue.log
#
# Wszystko poza lr / rank / alpha / dropout jest IDENTYCZNE z recepturą ablacji
# (training/ocr/surya/ablation/run_ablation_queue.sh): dane v2_200k, 10 000
# kroków, batch 8, wd 0.05, cosine, warmup 0.1, bf16, target q/v/o_proj,
# ewaluacja co 2500 kroków. Dzięki temu różnice w CER można przypisać
# zmienianym parametrom, a nie reszcie przepisu.
#
# Baseline do porównania (już zmierzony, benchmark `ocr-ablacja-v2`):
#     v2_200k-lora-r64 = lr 2e-5 / r64 / a64 / dropout 0.2
#     CER 0.127243   WER 0.479250   EMA 0.262500   (2160 linii, poziom "line")
# Trenowany tą samą ścieżką i z tym samym budżetem 10k kroków, więc nadaje się
# na punkt odniesienia dla wszystkich 9 runów. Nie ma go w kolejce (jest już
# policzony) — leży w training/results/ocr/surya/v2_200k-lora-r64/adapter.
# Te trzy liczby to wzorzec dla smoke-testu parity (patrz hp_search/README.md).
#
# --- Runda 1b: baseline + DOKŁADNIE jedna zmiana ---------------------------------
# Runda 1a nie dała odpowiedzi, bo każdy jej run zmieniał naraz lr, alpha (r -> 2r)
# i dropout (0.2 -> 0.1) względem baseline'u — degradacji nie da się przypisać
# żadnemu z tych parametrów. W 1b każdy wiersz to baseline z jedną podmienioną
# wartością, więc różnica w CER ma jednoznaczne przypisanie:
#   hp1b_seed1234  — ten sam config, inny seed  -> miara szumu między runami
#   hp1b_d0.1      — dropout 0.1 (baseline 0.2) -> izoluje dropout
#   hp1b_d0.3      — dropout 0.3                -> strona przeciwna (więcej regularyzacji)
#   hp1b_a128      — alpha 128 (baseline 64)    -> izoluje alpha = 2r
#   hp1b_lr5e-5    — lr 5e-5 (baseline 2e-5)    -> izoluje lr
#   hp1b_steps20k  — 20 000 kroków (baseline 10 000) -> jedyna nietknięta dźwignia
# 6 runów ≈ 3,1 h. hp1b_steps20k to jedyny zakład w górę (~1 h) — można go
# pominąć przez ONLY= (albo skomentować wiersz), reszta to diagnostyka.
#
# hp1b_* wymaga `--seed` w cli.py (commit z tą kolejką go dokłada).
#
# Każdy run jest pomijany, jeśli adapter już istnieje, a eval — jeśli wynik
# już jest, więc kolejkę można bezpiecznie restartować po przerwaniu.
# Restrict: ONLY="hp_lr1e-4_r64" bash training/ocr/surya/hp_search/run_hp_queue.sh
# Dopisz predykcje per linia: DUMP_PREDICTIONS=1 bash ...

set -u
cd "$(dirname "$0")/../../../.." || exit 1

DATA_ROOT="${DATA_ROOT:-dataset}"                       # dataset/<name>/images_shards + labels.jsonl
PROCESSED="${PROCESSED:-training/data/processed}"       # converted Surya-format data
RESULTS="${RESULTS:-training/results/ocr/surya/hp}"     # adaptery + results/<name>.json
COMPOSE=(docker compose -f training/docker-compose.yml run --rm surya-training)
REPORT_TO="${REPORT_TO:-}"                              # set to "wandb" to log
DUMP_PREDICTIONS="${DUMP_PREDICTIONS:-}"

DATASET="${DATASET:-v2_200k}"
STEPS="${STEPS:-10000}"
EVAL_STEPS="${EVAL_STEPS:-2500}"
LABELS_CSV="${LABELS_CSV:-benchmark/dane/handlabeled/labels.csv}"
IMAGES_DIR="${IMAGES_DIR:-benchmark/dane/handlabeled/images}"

# name             lr     rank alpha dropout [seed] [steps]
#   alpha = 2*rank, dropout = domyślne PEFT (runda 1a)
#   "-" w seed/steps = wartość domyślna (42 / $STEPS)
RUNS=(
  # --- runda 1a: lr x rank (wszystkie adaptery już policzone, kolejka je pomija) ---
  "hp_lr5e-5_r32   5e-5   32   64    0.1"
  "hp_lr5e-5_r64   5e-5   64   128   0.1"
  "hp_lr5e-5_r128  5e-5   128  256   0.1"
  "hp_lr1e-4_r32   1e-4   32   64    0.1"
  "hp_lr1e-4_r64   1e-4   64   128   0.1"
  "hp_lr1e-4_r128  1e-4   128  256   0.1"
  "hp_lr2e-4_r32   2e-4   32   64    0.1"
  "hp_lr2e-4_r64   2e-4   64   128   0.1"
  "hp_lr2e-4_r128  2e-4   128  256   0.1"
  # --- runda 1b: baseline (lr 2e-5 / r64 / a64 / d0.2) + jedna zmiana ---
  "hp1b_seed1234   2e-5   64   64    0.2   1234  -"
  "hp1b_d0.1       2e-5   64   64    0.1   -     -"
  "hp1b_d0.3       2e-5   64   64    0.3   -     -"
  "hp1b_a128       2e-5   64   128   0.2   -     -"
  "hp1b_lr5e-5     5e-5   64   64    0.2   -     -"
  "hp1b_steps20k   2e-5   64   64    0.2   -     20000"
  # Opcjonalnie (druga noga pary replikatów — różni się od baseline'u z 1a tylko
  # tym, że init LoRA jest seedowany, więc para seed42/seed1234 daje czysty szum):
  # "hp1b_seed42     2e-5   64   64    0.2   -     -"
)
ONLY="${ONLY:-}"

mkdir -p "$RESULTS/results"
echo "== hp queue start $(date -Is)  (dataset=$DATASET, steps=$STEPS)"

if [[ ! -f "$LABELS_CSV" ]]; then
  echo "== BŁĄD: brak $LABELS_CSV — bez handlabeled nie ma czym porównywać runów."
  echo "   (benchmark/dane/ jest w .gitignore, więc musi istnieć w checkoutcie na serwerze)"
  exit 1
fi

for spec in "${RUNS[@]}"; do
  read -r name lr rank alpha dropout seed steps <<<"$spec"
  seed="${seed:--}"
  steps="${steps:--}"
  [[ "$steps" == "-" ]] && steps="$STEPS"
  seed_extra=()
  [[ "$seed" != "-" ]] && seed_extra=(--seed "$seed")
  if [[ -n "$ONLY" && " $ONLY " != *" $name "* ]]; then
    continue
  fi
  out="$RESULTS/$name"
  result_json="$RESULTS/results/$name.json"

  if [[ ! -d "$out/adapter" ]]; then
    if [[ ! -f "$DATA_ROOT/$DATASET/labels.jsonl" ]]; then
      echo "== $name: brak danych w $DATA_ROOT/$DATASET (dvc pull?), pomijam"
      continue
    fi
    if [[ ! -f "$PROCESSED/$DATASET/train/metadata.jsonl" ]]; then
      echo "== $name: konwersja $DATASET $(date -Is)"
      "${COMPOSE[@]}" python training/ocr/surya/convert_data.py ocr800k \
          --input "$DATA_ROOT/$DATASET" --output "$PROCESSED/$DATASET" \
          || { echo "== $name: konwersja NIE UDAŁA SIĘ"; continue; }
    fi

    echo "== $name: trening (lr=$lr rank=$rank alpha=$alpha dropout=$dropout seed=$seed steps=$steps) $(date -Is)"
    extra=()
    [[ -n "$REPORT_TO" ]] && extra=(--report-to "$REPORT_TO")
    "${COMPOSE[@]}" python training/ocr/surya/cli.py \
        --train-metadata "$PROCESSED/$DATASET/train/metadata.jsonl" \
        --train-images-dir "$PROCESSED/$DATASET/train/images" \
        --val-metadata "$PROCESSED/$DATASET/val/metadata.jsonl" \
        --val-images-dir "$PROCESSED/$DATASET/val/images" \
        --lora --lora-rank "$rank" --lora-alpha "$alpha" --lora-dropout "$dropout" \
        --lora-target-modules q_proj v_proj o_proj \
        --max-steps "$steps" --batch-size 8 --learning-rate "$lr" --weight-decay 0.05 \
        --warmup-ratio 0.1 --lr-scheduler-type cosine \
        --evaluation-strategy steps --eval-steps "$EVAL_STEPS" --save-total-limit 3 \
        --run-name "$name" --output-dir "$out" \
        "${seed_extra[@]}" "${extra[@]}" \
        || { echo "== $name: trening NIE UDAŁ SIĘ"; continue; }
  else
    echo "== $name: adapter już jest, pomijam trening"
  fi

  if [[ -f "$result_json" ]]; then
    echo "== $name: wynik już policzony, pomijam eval"
    continue
  fi

  echo "== $name: eval CER na handlabeled $(date -Is)"
  eval_extra=()
  [[ -n "$DUMP_PREDICTIONS" ]] && eval_extra=(--dump-predictions "$RESULTS/results/$name.pred.jsonl")
  "${COMPOSE[@]}" python training/ocr/surya/hp_search/eval_cer.py \
      --adapter "$out/adapter" \
      --labels-csv "$LABELS_CSV" --images-dir "$IMAGES_DIR" \
      --output-json "$result_json" "${eval_extra[@]}" \
      || { echo "== $name: eval NIE UDAŁ SIĘ (adapter zostaje, kolejka leci dalej)"; continue; }

  echo "== $name: gotowe $(date -Is)"
done

echo "== hp queue: tabela $(date -Is)"
"${COMPOSE[@]}" python training/ocr/surya/hp_search/summarize.py "$RESULTS/results" \
    || echo "== podsumowanie nie udało się (wyniki są w $RESULTS/results/)"

echo "== hp queue end $(date -Is)"
