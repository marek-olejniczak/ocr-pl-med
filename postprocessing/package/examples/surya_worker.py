"""Przykład podłączenia korektora do procesu, który robi OCR Suryą. Do skopiowania i przerobienia.

Pokazuje wszystko, co aplikacja musi zrobić po swojej stronie:
  - jeden FoundationPredictor z adapterem: używa go i OCR, i ocena kandydatów (model raz w pamięci),
  - korektor ładowany w tle przy starcie,
  - czyszczenie wyjścia Suryi takie samo jak w benchmarku (na nim mierzono wyniki),
  - cała strona jednym wywołaniem, te same wycinki linii do OCR i do oceny,
  - błąd korekty nie psuje OCR: ocena Suryi -> statyczny -> surowy tekst.

    pip install "./postprocessing/package[surya]" && ocr-postprocess setup
    TORCH_DEVICE=cuda python surya_worker.py --adapter KATALOG linia1.jpg linia2.jpg
"""

from __future__ import annotations

import argparse
import html
import logging
import re
import threading

from PIL import Image

log = logging.getLogger("ocr")

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"</?[a-zA-Z][^<>]*>")


def clean(raw: str) -> str:
    """Jak benchmark/docker/surya/app.py: wejście to jedna linia, więc wszystko po <br> to halucynacja."""
    return html.unescape(_TAG_RE.sub("", _BR_RE.split(raw, maxsplit=1)[0])).strip()


class Ocr:
    def __init__(self, adapter: str, bundle: str | None = None, batch_size: int = 8):
        from peft import PeftModel
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        from ocr_postprocess.surya_score import SuryaScorer

        foundation = FoundationPredictor()  # urządzenie: zmienna TORCH_DEVICE albo autodetekcja Suryi
        foundation.model = PeftModel.from_pretrained(foundation.model, adapter).eval()
        self.recognition = RecognitionPredictor(foundation)
        self.scorer = SuryaScorer(foundation, max_batch=batch_size)
        self.batch_size = batch_size
        self.lock = threading.Lock()  # jeden model: OCR i ocena nie mogą iść równolegle
        self._corrector = None
        self._loader = threading.Thread(target=self._load_corrector, args=(bundle,), daemon=True)
        self._loader.start()

    def _load_corrector(self, bundle):
        try:
            from ocr_postprocess import load
            self._corrector = load(bundle)
        except Exception:
            log.exception("korektor niedostępny, aplikacja działa bez korekty")

    def read(self, crops: list[Image.Image]) -> list[str]:
        results = self.recognition(
            crops, task_names=["ocr_without_boxes"] * len(crops),
            bboxes=[[[0, 0, im.width, im.height]] for im in crops],  # wycinek = jedna linia
            math_mode=False, recognition_batch_size=self.batch_size, sort_lines=False, return_words=False)
        return [" ".join(t for t in (clean(str(line.text)) for line in (r.text_lines or [])) if t) for r in results]

    def page(self, crops: list[Image.Image], correction: str = "surya") -> list[dict]:
        """Wycinki linii jednej strony -> [{"raw", "text", "changes", "mode"}]; correction: surya | static | off."""
        crops = [c.convert("RGB") for c in crops]
        if not crops:
            return []
        with self.lock:
            raw = self.read(crops)
            if correction != "off":
                self._loader.join()  # tylko pierwsze wywołanie może tu chwilę poczekać
            modes = {"surya": ["surya", "static"], "static": ["static"], "off": []}[correction]
            for mode in modes if self._corrector else []:
                try:
                    kw = {"images": crops, "scorer": self.scorer} if mode == "surya" else {}
                    fixed = self._corrector.correct_lines_detailed(raw, **kw)
                    return [{"raw": r, "text": f["text"], "changes": f["changes"], "mode": mode}
                            for r, f in zip(raw, fixed)]
                except Exception:
                    log.exception("korekta (%s) nie powiodła się", mode)
        return [{"raw": r, "text": r, "changes": [], "mode": "off"} for r in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--correction", default="surya", choices=["surya", "static", "off"])
    ap.add_argument("images", nargs="+", help="wycinki linii")
    a = ap.parse_args()
    ocr = Ocr(a.adapter)
    for path, r in zip(a.images, ocr.page([Image.open(p) for p in a.images], a.correction)):
        print(f"{path} [{r['mode']}]\n  OCR:     {r['raw']}\n  korekta: {r['text']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
