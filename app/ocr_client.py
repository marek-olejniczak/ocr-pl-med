"""Client for the OCR benchmark services (benchmark/docker/*/app.py).

Every service exposes the same contract, so the OCR model is just a URL
(tesseract-pol :8007, trocr :8002, ... - see benchmark/docker-compose.yml):

    GET  /health
    POST /load           {"options": {...}}
    POST /predict_batch  {"images_base64": [...], "options": {...}}
                         -> {"texts": [...]}
"""

import base64
import io

import requests


class OCRClient:
    def __init__(self, url, timeout=300):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def health(self):
        r = requests.get(f"{self.url}/health", timeout=5)
        r.raise_for_status()
        return r.json()

    def load(self, options=None):
        # warm-up: first predict otherwise pays the model init cost
        r = requests.post(f"{self.url}/load",
                          json={"options": options or {}},
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def predict_batch(self, crops, options=None):
        """crops: list of PIL images -> list of recognized texts."""
        if not crops:
            return []
        payload = {"images_base64": [self._encode(c) for c in crops],
                   "options": options or {}}
        r = requests.post(f"{self.url}/predict_batch", json=payload,
                          timeout=self.timeout)
        r.raise_for_status()
        return r.json()["texts"]

    @staticmethod
    def _encode(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()
