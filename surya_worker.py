"""
surya_worker.py — lokalny serwis OCR: Surya 0.17.1 + adapter LoRA zespołu.

Adapter był trenowany na surya-ocr==0.17.1 / transformers 4.56.1 (gałąź `server`),
a Surya wymaga Pythona >=3.10, więc działa w osobnym venv (.venv-surya17),
a labeling_tool łączy się z nim po HTTP (patrz SuryaLoraRecognizer w auto_annotate.py).

Uruchomienie ręczne:
    .venv-surya17/bin/python surya_worker.py

API:
    GET  /health            -> {"status": "ok", "adapter": "..."}
    POST /recognize_batch   {"images_base64": [png, ...]} -> {"texts": [...]}
"""

from __future__ import annotations

import base64
import html
import io
import os
import re
from pathlib import Path

HERE = Path(__file__).parent.resolve()
ADAPTER_DIR = Path(os.environ.get("SURYA_ADAPTER_DIR", HERE / "models" / "surya_lora_adapter"))
PORT = int(os.environ.get("SURYA_WORKER_PORT", "5055"))
BATCH_SIZE = int(os.environ.get("SURYA_BATCH_SIZE", "8"))

# Surya czyta ustawienia ze środowiska przy imporcie.
os.environ.setdefault("TORCH_DEVICE", os.environ.get("SURYA_DEVICE", "mps"))

from flask import Flask, jsonify, request  # noqa: E402
from PIL import Image  # noqa: E402

app = Flask(__name__)
_predictor = None

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")


def _clean(raw: str) -> str:
    """Surya zwraca lekki HTML; wejście to pojedyncza linia, więc wszystko po <br> to halucynacja."""
    text = _BR_RE.split(raw, maxsplit=1)[0]
    return html.unescape(_TAG_RE.sub("", text)).strip()


def _get_predictor():
    global _predictor
    if _predictor is None:
        from peft import PeftModel
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        foundation = FoundationPredictor()
        foundation.model = PeftModel.from_pretrained(foundation.model, str(ADAPTER_DIR))
        _predictor = RecognitionPredictor(foundation)
    return _predictor


@app.get("/health")
def health():
    return jsonify({"status": "ok", "adapter": str(ADAPTER_DIR), "loaded": _predictor is not None})


@app.post("/recognize_batch")
def recognize_batch():
    payload = request.get_json(force=True)
    images = []
    for b64 in payload.get("images_base64", []):
        images.append(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))
    if not images:
        return jsonify({"texts": []})

    results = _get_predictor()(
        images,
        task_names=["ocr_without_boxes"] * len(images),
        # wejście to gotowe wycinki linii — jeden bbox na cały obraz
        bboxes=[[[0, 0, im.width, im.height]] for im in images],
        math_mode=False,
        recognition_batch_size=BATCH_SIZE,
        sort_lines=False,
        return_words=False,
    )
    texts = [
        " ".join(t for t in (_clean(str(line.text)) for line in (r.text_lines or [])) if t)
        for r in results
    ]
    return jsonify({"texts": texts})


if __name__ == "__main__":
    _get_predictor()  # ładuj przy starcie, nie przy pierwszym żądaniu
    app.run(host="127.0.0.1", port=PORT, threaded=False)
