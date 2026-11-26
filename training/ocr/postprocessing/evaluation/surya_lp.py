"""Ticket 20 krok C, część serwerowa: log P(tekst | obraz) z Suryi dla próśb z rescore_cands (teacher forcing).

W kontenerze surya-training (WORKDIR /app = repo), na GPU:
    docker compose -f training/docker-compose.yml run --rm -e PYTHONPATH=/app/training/ocr/postprocessing surya-training \
        python training/ocr/postprocessing/evaluation/surya_lp.py --adapter <adapter> --inp req.jsonl --out resC.jsonl
Wejście: {"id", "image" (względem /app), "texts"}; wyjście: {"id", "scores": [{"lp"}...]}.
"""

import argparse
import json
import time
from pathlib import Path

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--root", default="/app")
a = ap.parse_args()

from peft import PeftModel  # noqa: E402
from surya.foundation import FoundationPredictor  # noqa: E402

from pp.surya_score import SuryaScorer  # noqa: E402

fp = FoundationPredictor()
fp.model = PeftModel.from_pretrained(fp.model, a.adapter).eval()
scorer = SuryaScorer(fp)
t0, n = time.time(), 0
with open(a.inp, encoding="utf-8") as fi, open(a.out, "w", encoding="utf-8") as fo:
    for line in fi:
        q = json.loads(line)
        lps = scorer.score(Image.open(Path(a.root) / q["image"]), q["texts"])
        fo.write(json.dumps({"id": q["id"], "scores": [{"lp": x} for x in lps]}) + "\n")
        n += len(q["texts"])
print(f"koniec: {n} tekstów, {time.time() - t0:.0f}s")
