#!/usr/bin/env bash
# Unpack the round-two evaluation sets from dataset/ (pulled with DVC) into
# benchmark/dane/, where the benchmark configs expect them.
#   dvc pull dataset/robustness dataset/v2_val10k
#   bash training/ocr/surya/ablation/prepare_eval_data.sh
set -euo pipefail
cd "$(dirname "$0")/../../../.."

for name in robustness v2_val10k; do
  src="dataset/$name"
  dst="benchmark/dane/$name"
  if [[ ! -f "$src/labels.csv" ]]; then
    echo "== brak $src - najpierw: dvc pull $src"
    exit 1
  fi
  mkdir -p "$dst/images"
  for archive in "$src"/images_shards/*.tar; do
    tar -tf "$archive" >/dev/null 2>&1 || { echo "== USZKODZONY $archive - skopiuj go ponownie"; exit 1; }
    tar -xf "$archive" -C "$dst/images"
  done
  cp "$src/labels.csv" "$src/labels.jsonl" "$dst/"
  n_img=$(find "$dst/images" -type f | wc -l)
  n_lab=$(( $(wc -l < "$dst/labels.csv") - 1 ))
  echo "== $name: obrazow $n_img, etykiet $n_lab"
  if [[ "$n_img" -ne "$n_lab" ]]; then
    echo "== NIEZGODNOSC: liczba obrazow i etykiet sie rozni"
    exit 1
  fi
done
echo "== gotowe, mozna odpalac benchmark odpornosci"
