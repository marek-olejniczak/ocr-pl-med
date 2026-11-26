#!/usr/bin/env bash
# Lay out the real test set v2 the way the OCR benchmark reads it.
#   dvc pull dataset/test/real/v2
#   bash training/ocr/surya/ablation/prepare_real_v2.sh
# -> benchmark/dane/real_v2/images/*.jpg and labels.csv (file_name,label)
# crops.csv holds only the lines usable for OCR (no #ignore, #unreadable, [?]),
# and the set has no dev/test split, so nothing is filtered here.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
SRC="${SRC:-dataset/test/real/v2}"
DST="${DST:-benchmark/dane/real_v2}"
PY="${PY:-python3}"
[[ -f "$SRC/crops.csv" ]] || { echo "== brak $SRC - najpierw: dvc pull $SRC"; exit 1; }
mkdir -p "$DST/images"
find "$SRC/crops" -type f -name '*.jpg' -exec cp -u {} "$DST/images/" \;
"$PY" - "$SRC" "$DST" <<'PY'
import csv, sys
from pathlib import Path
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
rows = [{"file_name": Path(r["file_name"]).name, "label": r["label"]}
        for r in csv.DictReader(open(src / "crops.csv", encoding="utf-8"))]
if len({r["file_name"] for r in rows}) != len(rows):
    sys.exit("powtorzone nazwy wycinkow")
for r in rows:
    if not (dst / "images" / r["file_name"]).exists():
        sys.exit(f"brak obrazu {r['file_name']}")
    if r["label"].startswith("#") or "[?]" in r["label"]:
        sys.exit(f"tag w crops.csv: {r['file_name']}")
with open(dst / "labels.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["file_name", "label"])
    w.writeheader()
    w.writerows(rows)
print(f"== labels.csv: {len(rows)} linii")
PY
