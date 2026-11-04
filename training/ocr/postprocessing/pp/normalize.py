"""Opcja 0 (ticket 01): higiena tekstu wyjściowego. Bez NFKC, bez zmian wielkości liter.

Transformacje „tylko dla metryki” (casefold, bez interpunkcji) są w evaluation/metrics.py
(`normalize_aux`) i nigdy nie dotykają tekstu korektora.
"""

from __future__ import annotations

import unicodedata

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))


def hygiene(text: str) -> str:
    t = unicodedata.normalize("NFC", text).translate(ZERO_WIDTH)
    return " ".join(t.split())
