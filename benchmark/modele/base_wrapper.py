from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List


class HTRModelWrapper(ABC):
    """Bazowa klasa dla modeli HTR używanych w benchmarku."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def predict(self, image_path: str) -> str:
        """Przyjmuje sciezke do obrazu i zwraca rozpoznany tekst."""

    def predict_batch(self, image_paths: Iterable[str]) -> List[str]:
        """Domyslna implementacja batch inference oparta na wywolaniach `predict`."""
        return [self.predict(path) for path in image_paths]