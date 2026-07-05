"""Document -> text pipeline: detect lines, crop them, OCR each crop.

Deliberately UI-free: ui.py (gradio) is a thin wrapper and a future
FastAPI/React front can reuse this module unchanged.
"""

from dataclasses import dataclass

PAD_PX = 4  # small margin around each crop so ascenders/descenders survive


@dataclass
class LineResult:
    bbox: tuple   # x, y, w, h in original image coordinates
    score: float
    text: str


def sort_reading_order(lines):
    """Top-to-bottom by y-center. Single-column assumption, which holds for
    the current documents; column handling would slot in here."""
    return sorted(lines, key=lambda l: l.y + l.h / 2)


def crop_line(image, line, pad=PAD_PX):
    x1 = max(0, int(line.x) - pad)
    y1 = max(0, int(line.y) - pad)
    x2 = min(image.width, int(line.x + line.w) + pad)
    y2 = min(image.height, int(line.y + line.h) + pad)
    return image.crop((x1, y1, x2, y2))


def run(image, detector, ocr, pad=PAD_PX, preprocess=None):
    """image: PIL.Image -> (list[LineResult] in reading order, full text).

    With `preprocess` set, detection runs on the preprocessed image while
    crops are cut from the original - valid because preprocessing is
    geometry-preserving (see preprocess.py)."""
    det_input = preprocess(image) if preprocess else image
    lines = sort_reading_order(detector.detect(det_input))
    crops = [crop_line(image, ln, pad) for ln in lines]
    texts = ocr.predict_batch(crops)
    results = [LineResult((ln.x, ln.y, ln.w, ln.h), ln.score, txt)
               for ln, txt in zip(lines, texts)]
    return results, "\n".join(texts)
