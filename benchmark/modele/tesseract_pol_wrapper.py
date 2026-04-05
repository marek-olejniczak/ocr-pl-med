from __future__ import annotations

from pathlib import Path

import pytesseract
from PIL import Image

from modele.base_wrapper import HTRModelWrapper


class TesseractPolWrapper(HTRModelWrapper):
    """Wrapper dla Tesseract OCR z jezykiem polskim."""

    def __init__(
        self,
        language: str = "pol",
        psm: int = 7,
        oem: int = 1,
        tesseract_cmd: str | None = None,
    ) -> None:
        super().__init__(model_name="Tesseract_POL")
        self.language = language
        self.psm = psm
        self.oem = oem
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._validate_tesseract_installation()

    @staticmethod
    def _validate_tesseract_installation() -> None:
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "Nie znaleziono binarki 'tesseract'. Zainstaluj Tesseract w systemie lub "
                "podaj sciezke parametrem --tesseract-cmd."
            ) from exc

    def predict(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Nie znaleziono obrazu: {image_path}")

        config = f"--oem {self.oem} --psm {self.psm}"
        with Image.open(path) as image:
            text = pytesseract.image_to_string(image, lang=self.language, config=config)
        return text.strip()
