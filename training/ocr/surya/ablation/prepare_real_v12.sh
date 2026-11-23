#!/usr/bin/env bash
# Lay out the real-notes test set v1.2 the way the OCR benchmark reads it.
#   dvc pull dataset/test/real/v1.2
#   bash training/ocr/surya/ablation/prepare_real_v12.sh
# -> benchmark/dane/real_v1.2/images/*.jpg, labels.csv (file_name,label),
#    labels_dev.csv and labels_test.csv (the split column of manifest.csv)
set -euo pipefail
cd "$(dirname "$0")/../../../.."
SRC="${SRC:-dataset/test/real/v1.2}"
DST="${DST:-benchmark/dane/real_v1.2}"
PY="${PY:-python3}"
[[ -f "$SRC/crops.csv" ]] || { echo "== brak $SRC - najpierw: dvc pull $SRC"; exit 1; }
mkdir -p "$DST/images"
find "$SRC/crops" -type f -name '*.jpg' -exec cp -u {} "$DST/images/" \;
"$PY" - "$SRC" "$DST" <<'PY'
import csv, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
split = {r["page_id"]: r["split"] for r in csv.DictReader(open(src / "manifest.csv", encoding="utf-8"))}
rows = list(csv.DictReader(open(src / "crops.csv", encoding="utf-8")))
parts = {"": [], "_dev": [], "_test": []}
for r in rows:
    row = {"file_name": Path(r["file_name"]).name, "label": r["label"]}
    if not (dst / "images" / row["file_name"]).exists():
        sys.exit(f"brak obrazu {row['file_name']}")
    parts[""].append(row)
    parts["_" + split[r["line_id"].rsplit("_l", 1)[0]]].append(row)
if len({r["file_name"] for r in parts[""]}) != len(parts[""]):
    sys.exit("powtorzone nazwy wycinkow")
for suffix, part in parts.items():
    with open(dst / f"labels{suffix}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_name", "label"])
        w.writeheader()
        w.writerows(part)
    print(f"== labels{suffix}.csv: {len(part)} linii")
PY
echo "== obrazow: $(find "$DST/images" -type f | wc -l)"
