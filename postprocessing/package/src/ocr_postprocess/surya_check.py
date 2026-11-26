"""Test zgodności trybu z oceną Suryi na danej maszynie (CUDA / MPS / CPU), bez naszych danych.

    ocr-postprocess check-surya --adapter KATALOG_ADAPTERA        # urządzenie: zmienna TORCH_DEVICE

Rysuje linię tekstu i sprawdza dwie rzeczy, które mogą po cichu zawieść na innym sprzęcie:
  1. ocena liczona w partii (różne długości, dopełnienie) zgadza się z oceną pojedynczą (maska uwagi, dtype);
  2. tekst z obrazu dostaje wyższe log P niż przekręcony.
Mierzy też czas oceny. Nie sprawdza jakości adaptera.
"""

from __future__ import annotations

import time

TEXT = "Kość udowa i rzepka"
WRONG = "Rość udawa i rzeqka"
TOL = 1.0  # nat; na RTX 3090 (bf16) maks. 0.55 w 1% tekstów


def render(text: str):
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=40)
    w = int(font.getlength(text)) + 40
    img = Image.new("RGB", (w, 72), "white")
    ImageDraw.Draw(img).text((20, 12), text, fill="black", font=font)
    return img


def compare_scores(batched: list[float], single: list[float]) -> list[str]:
    """Problemy z ocenami [tekst z obrazu, przekręcony, krótszy]; pusta lista = zgodne."""
    problems = []
    worst = max(abs(a - b) for a, b in zip(batched, single))
    if not worst <= TOL:  # także NaN
        problems.append(f"ocena w partii różni się od pojedynczej o {worst:.3f} nat (limit {TOL})")
    if not batched[0] > batched[1]:
        problems.append(f"przekręcony tekst nie dostał niższej oceny ({batched[0]:.2f} vs {batched[1]:.2f})")
    return problems


def run(adapter: str | None) -> int:
    from surya.foundation import FoundationPredictor

    from ocr_postprocess.surya_score import SuryaScorer

    fp = FoundationPredictor()
    if adapter:
        from peft import PeftModel
        fp.model = PeftModel.from_pretrained(fp.model, adapter).eval()
    scorer = SuryaScorer(fp)
    print(f"urządzenie: {scorer.device}, dtype: {scorer.dtype}, adapter: {adapter or 'brak (model bazowy)'}")
    img, texts = render(TEXT), [TEXT, WRONG, "Kość"]
    t0 = time.time()
    batched = scorer.score(img, texts)
    first = time.time() - t0
    single = [scorer.score(img, [t])[0] for t in texts]
    t0 = time.time()
    n = len(scorer.score(img, [TEXT] * 16))
    rate = n / (time.time() - t0)
    for t, b, s in zip(texts, batched, single):
        print(f"  {t!r:28s} w partii {b:9.3f}   pojedynczo {s:9.3f}")
    print(f"pierwsza ocena {first:.1f} s, potem ok. {rate:.0f} wariantów/s "
          f"(strona 30 linii to ok. 150 wariantów: ok. {150 / rate:.0f} s; prawdziwe linie są większe, więc bywa wolniej)")
    problems = compare_scores(batched, single)
    for p in problems:
        print("BŁĄD:", p)
    print("tryb z oceną Suryi:", "NIE UŻYWAĆ na tej maszynie, zostać przy statycznym" if problems else "OK")
    return 1 if problems else 0
