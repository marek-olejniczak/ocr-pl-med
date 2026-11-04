"""Słownik form (P1: minimalny loader do oracle; P2 doda kategorie, NKJP, indeks długości).

Formy są przechowywane małymi literami. Pierwsze wczytanie PoliMorf (~8.5 mln wierszy)
zapisuje listę unikalnych form w $PP_CACHE, kolejne czytają już tę listę.
"""

from __future__ import annotations

import gzip
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fetch_resources import cache_dir, resource_path  # noqa: E402

POLIMORF = "polimorf-20261018.tab.gz"


def fold(s: str) -> str:
    """Zdejmuje diakrytyki (ł→l osobno, bo nie rozkłada się w NFD)."""
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=None)
def polimorf_forms() -> frozenset[str]:
    cached = cache_dir() / "forms-polimorf-20261018.txt"
    if cached.exists():
        return frozenset(cached.read_text(encoding="utf-8").split("\n"))
    forms = set()
    with gzip.open(resource_path(POLIMORF), "rt", encoding="utf-8") as f:
        for line in f:
            parts = line.split("\t", 2)
            if len(parts) >= 3 and not line.startswith("#"):
                forms.add(parts[0].lower())
    forms.discard("")
    cached.write_text("\n".join(sorted(forms)), encoding="utf-8")
    return frozenset(forms)
