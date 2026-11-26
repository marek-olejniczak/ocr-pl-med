"""Słownik form (ticket 05, 04): PoliMorf 2026-10-18 (+ sjp „odmiany” jako ablacja).

Dwa zbiory (ticket 04, „dwa słowniki”):
  known(w)   – czy token jest znany i ma zostać bez zmian (wszystkie formy PoliMorf, skróty, sjp)
  target(w)  – czy forma może być kandydatem korekty: formy pospolite + formy wyłącznie
               własne (nazwiska, geografia…) tylko przy liczności NKJP ≥ proper_min_count
Formy są małymi literami. Pierwsze wczytanie zapisuje przetworzoną listę w $PP_CACHE.
"""

from __future__ import annotations

import gzip
import io
import re
import unicodedata
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ocr_postprocess.resources import cache_dir, resource_path

POLIMORF = "polimorf-20261018.tab.gz"
SJP = "sjp-odm-20261101.zip"
COMMON_CATEGORIES = {"", "nazwa_pospolita"}
LETTER_RUN = re.compile(r"[^\W\d_]+")
FRAGMENT = 16  # człon hasła wieloczłonowego / z dywizem: tylko do known()
DOMAIN_TERMS = Path(__file__).resolve().parent / "data" / "ta_terms.tsv"
MIN_DOMAIN_LEN = 4
DOMAIN_LANGS = {"en", "la", "pl"}

# Skróty anatomiczne z TA98 (User Guide, ticket 04), małymi literami, bez kropki.
TA98_ABBREV = {"a", "aa", "lig", "ligg", "m", "mm", "n", "nn", "r", "rr", "v", "vv"}


def fold(s: str) -> str:
    """Zdejmuje diakrytyki (ł→l osobno, bo nie rozkłada się w NFD)."""
    s = s.replace("ł", "l").replace("Ł", "L")
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _polimorf_cache() -> Path:
    return cache_dir() / "polimorf-20261018.kinds.tsv"


def _build_polimorf() -> None:
    kind: dict[str, int] = defaultdict(int)  # bit 1 pospolita, 2 własna, 4 brev:pun, 8 brev:npun, 16 człon
    with gzip.open(resource_path(POLIMORF), "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            form, tag = parts[0].lower(), parts[2]
            cat = parts[3] if len(parts) > 3 else ""
            if not LETTER_RUN.fullmatch(form):
                for run in LETTER_RUN.findall(form):
                    kind[run] |= FRAGMENT
            if tag.startswith("brev"):
                kind[form] |= 4 if tag.startswith("brev:pun") else 8
            elif cat in COMMON_CATEGORIES:
                kind[form] |= 1
            else:
                kind[form] |= 2
    tmp = _polimorf_cache().with_suffix(".part")
    with open(tmp, "w", encoding="utf-8") as out:
        for form in sorted(kind):
            out.write(f"{form}\t{kind[form]}\n")
    tmp.replace(_polimorf_cache())


@lru_cache(maxsize=None)
def polimorf_kinds() -> dict[str, int]:
    if not _polimorf_cache().exists():
        _build_polimorf()
    d = {}
    with open(_polimorf_cache(), encoding="utf-8") as f:
        for line in f:
            form, k = line.rstrip("\n").split("\t")
            d[form] = int(k)
    return d


@lru_cache(maxsize=None)
def polimorf_forms() -> frozenset[str]:
    """Wszystkie formy PoliMorf (P1: oracle)."""
    return frozenset(polimorf_kinds())


@lru_cache(maxsize=None)
def sjp_forms() -> frozenset[str]:
    cached = cache_dir() / "sjp-odm-20261101.runs.txt"
    if cached.exists():
        return frozenset(cached.read_text(encoding="utf-8").split("\n"))
    forms = set()
    with zipfile.ZipFile(resource_path(SJP)) as z, z.open("odm.txt") as raw:
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            for w in line.split(","):
                forms.update(r.lower() for r in LETTER_RUN.findall(w))
    cached.write_text("\n".join(sorted(forms)), encoding="utf-8")
    return frozenset(forms)


def domain_terms(langs: Iterable[str], path: Path = DOMAIN_TERMS) -> set[str]:
    """Terminy anatomiczne z wybranych języków (en, la, pl), do `Lexicon(extra_known=...)`.

    Słowo należy do wyniku, jeśli ma etykietę w co najmniej jednym z języków: `femur` zostaje przy
    samym `la`, a `muscle` (tylko en) nie. Pusty wybór daje pusty zbiór, czyli listę wyłączoną.
    """
    want = set(langs)
    if want - DOMAIN_LANGS:
        raise ValueError(f"nieznane języki: {sorted(want - DOMAIN_LANGS)}")
    out = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, word_langs = line.rstrip("\n").split("\t")
            if want & set(word_langs.split(",")):
                out.add(word)
    return out


class Lexicon:
    def __init__(self, use_sjp: bool = False, proper_min_count: int = 5, counts: dict[str, int] | None = None,
                 extra_known: set[str] | None = None):
        kinds = polimorf_kinds()
        self.common = {w for w, k in kinds.items() if k & 1}
        proper_only = {w for w, k in kinds.items() if k & 2 and not k & 1}
        self.abbrev = {w for w, k in kinds.items() if k & 12} | TA98_ABBREV
        self.abbrev_dot = {w for w, k in kinds.items() if k & 4}
        self.proper_only = proper_only
        self.fragments = {w for w, k in kinds.items() if k == FRAGMENT}
        if counts is None:
            from ocr_postprocess.lm import nkjp_counts
            counts = nkjp_counts()
        self.proper_target = {w for w in proper_only if counts.get(w, 0) >= proper_min_count}
        self.sjp = sjp_forms() - self.common if use_sjp else frozenset()
        self.extra_known = set(extra_known or ())
        self._by_len: dict[int, list[str]] | None = None
        self._by_fold: dict[str, list[str]] | None = None

    def known(self, w: str) -> bool:
        return (w in self.common or w in self.proper_only or w in self.abbrev or w in self.fragments
                or w in self.sjp or w in self.extra_known)

    def target(self, w: str) -> bool:
        return w in self.common or w in self.proper_target or w in self.sjp

    def targets(self):
        yield from self.common
        yield from self.proper_target - self.common
        yield from set(self.sjp) - self.common - self.proper_target

    def by_length(self) -> dict[int, list[str]]:
        if self._by_len is None:
            d: dict[int, list[str]] = defaultdict(list)
            for w in self.targets():
                d[len(w)].append(w)
            self._by_len = dict(d)
        return self._by_len

    def by_fold(self) -> dict[str, list[str]]:
        if self._by_fold is None:
            d: dict[str, list[str]] = defaultdict(list)
            for w in self.targets():
                d[fold(w)].append(w)
            self._by_fold = dict(d)
        return self._by_fold
