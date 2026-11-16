"""Opcja 0 (ticket 01): higiena tekstu wyjściowego. Bez NFKC, bez zmian wielkości liter.

Transformacje „tylko dla metryki” (casefold, bez interpunkcji) są w evaluation/metrics.py
(`normalize_aux`) i nigdy nie dotykają tekstu korektora.

`deloop` (runda 2): Surya v3 na obrazach spoza dystrybucji (RxHandBD) zapętla się: `Maring Maring …`,
`- - - …`, segmenty `<br>`. Zmiany tylko przy ewidentnej degeneracji (≥ 3 powtórzenia z rzędu), więc
naturalne „bardzo bardzo” zostaje.
"""

from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
BR = re.compile(r"\s*<br\s*/?>\s*", re.IGNORECASE)
MIN_REPEATS = 3
MAX_NGRAM = 3


def deloop(text: str) -> str:
    """Scala identyczne segmenty `<br>` i zwija n-gramy (n ≤ 3) powtórzone ≥ 3 razy z rzędu do jednego."""
    if "<br" in text.lower():
        seen, parts = set(), []
        for p in BR.split(text):
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                parts.append(p)
        text = " ".join(parts)
    toks = text.split()
    for n in range(1, MAX_NGRAM + 1):
        out, i = [], 0
        while i < len(toks):
            g = toks[i:i + n]
            k = 1
            while len(g) == n and toks[i + k * n:i + (k + 1) * n] == g:
                k += 1
            if k >= MIN_REPEATS:
                out += g
                i += k * n
            else:
                out.append(toks[i])
                i += 1
        toks = out
    return " ".join(toks)


def hygiene(text: str, loops: bool = True) -> str:
    t = unicodedata.normalize("NFC", text).translate(ZERO_WIDTH)
    t = " ".join(t.split())
    return deloop(t) if loops else t
