from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterable

import requests

from modele.base_wrapper import HTRModelWrapper


class ModelClient:
    """Prosty klient HTTP do serwisu modelu OCR."""

    def __init__(self, url: str, timeout_seconds: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)

    def health(self) -> dict:
        response = requests.get(f"{self.url}/health", timeout=self.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Niepoprawna odpowiedz /health (oczekiwano obiektu JSON).")
        return payload

    def predict(self, image_path: str, options: dict | None = None) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        image_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        payload = {
            "image_base64": image_b64,
            "options": options or {},
        }

        response = requests.post(
            f"{self.url}/predict",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Niepoprawna odpowiedz /predict (oczekiwano obiektu JSON).")

        text = data.get("text", "")
        if not isinstance(text, str):
            return str(text).strip()
        return text.strip()

    def predict_batch(self, image_paths: Iterable[str], options: dict | None = None) -> list[str]:
        return [self.predict(path, options=options) for path in image_paths]


class HTTPModelWrapper(HTRModelWrapper):
    """Adapter klienta HTTP do kontraktu HTRModelWrapper."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout_seconds: float = 60.0,
        options: dict | None = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.client = ModelClient(url=base_url, timeout_seconds=timeout_seconds)
        self.options = options or {}

    def predict(self, image_path: str) -> str:
        return self.client.predict(image_path=image_path, options=self.options)

    def predict_batch(self, image_paths: Iterable[str]) -> list[str]:
        return self.client.predict_batch(image_paths=image_paths, options=self.options)
