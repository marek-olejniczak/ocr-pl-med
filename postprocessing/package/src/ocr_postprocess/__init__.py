"""Korektor wyjścia OCR (Surya) dla polskich rękopisów medycznych. Instrukcja podłączenia: README.md pakietu."""

__all__ = ["load", "Corrector", "DEFAULT_BUNDLE"]
__version__ = "1.0.0"


def __getattr__(name: str):  # leniwie: `ocr_postprocess.surya_score` ma działać bez zależności korektora
    if name in ("load", "DEFAULT_BUNDLE"):
        from ocr_postprocess import bundle
        return getattr(bundle, name)
    if name == "Corrector":
        from ocr_postprocess.correct import Corrector
        return Corrector
    raise AttributeError(f"module 'ocr_postprocess' has no attribute {name!r}")
