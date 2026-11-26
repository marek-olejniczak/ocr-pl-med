"""Skompilowany indeks słownika i częstości: budowany raz (ok. 30 s), potem ładowanie w ok. 1–2 s zamiast ok. 23 s.

Zawiera dokładnie to, czego korektor używa w czasie działania: `known`, `target`, `abbrev`, kubełki długości i grupy
wariantów bez diakrytyków dla prefiltra, `logp`/`count` z modelu unigramowego. Wyniki są identyczne z `Lexicon` +
`UnigramLM` (sprawdza `tests/test_compiled.py` i porównanie na v1.2 w README wyników).

Format: katalog z plikami tekstowymi (słowa rozdzielone \\n, UTF-8), tablicami numpy i `meta.json`.
Słowa są posortowane po (długość, słowo), więc kubełek długości to wycinek listy, a wyszukiwanie to bisect.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from ocr_postprocess.lexicon import Lexicon, domain_terms, fold

FORMAT = 1


def index_key(domain_langs: Iterable[str]) -> str:
    """Klucz indeksu: format, języki listy TA, rozmiary plików zasobów (zmiana zasobu = nowy indeks)."""
    from ocr_postprocess.resources import cache_dir
    sig = [f"format={FORMAT}", "domain=" + ",".join(sorted(domain_langs))]
    for name in ("polimorf-20261018.tab.gz", "nkjp-1grams.gz"):
        p = cache_dir() / name
        sig.append(f"{name}={p.stat().st_size if p.exists() else 0}")
    from ocr_postprocess.lexicon import DOMAIN_TERMS
    sig.append(f"terms={DOMAIN_TERMS.stat().st_size if DOMAIN_TERMS.exists() else 0}")
    return hashlib.sha256("|".join(sig).encode()).hexdigest()[:12]


def default_dir(domain_langs: Iterable[str]) -> Path:
    from ocr_postprocess.resources import cache_dir
    return cache_dir() / "compiled" / index_key(domain_langs)


def _write_words(path: Path, words: list[str]) -> None:
    path.write_bytes("\n".join(words).encode("utf-8"))


def _read_words(path: Path) -> list[str]:
    return path.read_bytes().decode("utf-8").split("\n")


def build(out: Path, domain_langs: Iterable[str] = ()) -> Path:
    """Z `Lexicon(extra_known=TA)` i `UnigramLM` (jak `bundle.load(compiled=False)`)."""
    from ocr_postprocess.lm import UnigramLM

    langs = list(domain_langs)
    lex = Lexicon(extra_known=domain_terms(langs))
    return write_index(out, lex, UnigramLM(lex.targets()), langs)


def write_index(out: Path, lex, lm, langs: list[str]) -> Path:
    """Zapis indeksu z gotowych obiektów (`Lexicon`, `UnigramLM` albo zgodnych)."""
    targets = sorted(set(lex.targets()), key=lambda w: (len(w), w))
    known = sorted(lex.common | lex.proper_only | lex.abbrev | lex.fragments | set(lex.sjp) | lex.extra_known)
    lens: dict[int, list[int]] = {}
    for i, w in enumerate(targets):
        lens.setdefault(len(w), [i, i])[1] = i + 1
    folds = [fold(w) for w in targets]
    order = sorted(range(len(targets)), key=folds.__getitem__)
    out.mkdir(parents=True, exist_ok=True)
    _write_words(out / "targets.txt", targets)
    _write_words(out / "known.txt", known)
    _write_words(out / "folds.txt", [folds[i] for i in order])
    np.save(out / "fold_perm.npy", np.asarray(order, dtype=np.int32))
    np.save(out / "counts.npy", np.asarray([lm.count(w) for w in targets], dtype=np.int64))
    meta = {"format": FORMAT, "domain_langs": langs, "lens": {str(k): v for k, v in lens.items()},
            "abbrev": sorted(lex.abbrev), "rstar": {str(k): v for k, v in lm.rstar.items()},
            "unseen": lm.unseen, "logz": lm.logz, "n_targets": len(targets), "n_known": len(known)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return out


class CompiledLexicon:
    """Interfejs `Lexicon` używany przez korektor (known, target, abbrev, by_length, fold_group)."""

    def __init__(self, path: Path):
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta["format"] != FORMAT:
            raise ValueError(f"{path}: format {meta['format']}, oczekiwano {FORMAT}")
        self.meta = meta
        self._targets = _read_words(path / "targets.txt")
        self._known = _read_words(path / "known.txt")
        self._folds = _read_words(path / "folds.txt")
        self._perm = np.load(path / "fold_perm.npy")
        self._lens = {int(k): (s, e) for k, (s, e) in meta["lens"].items()}
        self._by_len = {n: self._targets[s:e] for n, (s, e) in self._lens.items()}
        self.abbrev = set(meta["abbrev"])

    def known(self, w: str) -> bool:
        i = bisect.bisect_left(self._known, w)
        return i < len(self._known) and self._known[i] == w

    def index(self, w: str) -> int:
        """Indeks w liście targets albo -1."""
        se = self._lens.get(len(w))
        if se is None:
            return -1
        s, e = se
        i = bisect.bisect_left(self._targets, w, s, e)
        return i if i < e and self._targets[i] == w else -1

    def target(self, w: str) -> bool:
        return self.index(w) >= 0

    def by_length(self) -> dict[int, list[str]]:
        return self._by_len

    def fold_group(self, key: str) -> list[str]:
        f = fold(key)
        lo = bisect.bisect_left(self._folds, f)
        hi = bisect.bisect_right(self._folds, f, lo)
        return [self._targets[j] for j in self._perm[lo:hi].tolist()]


class CompiledLM:
    """Interfejs `UnigramLM` (logp, count, unseen, logz) dla słów z listy targets."""

    def __init__(self, path: Path, lex: CompiledLexicon):
        m = lex.meta
        self.lex = lex
        self.counts_arr = np.load(path / "counts.npy")
        self.rstar = {int(k): v for k, v in m["rstar"].items()}
        self.unseen = m["unseen"]
        self.logz = m["logz"]

    def count(self, w: str) -> int:
        i = self.lex.index(w)
        return int(self.counts_arr[i]) if i >= 0 else 0

    def logp(self, w: str) -> float:
        c = self.count(w)
        return math.log(self.rstar.get(c, float(c)) if c else self.unseen) - self.logz


def load(domain_langs: Iterable[str], path: Path | None = None, build_if_missing: bool = True):
    """(CompiledLexicon, CompiledLM); brakujący indeks buduje (raz) w $PP_CACHE/compiled/<klucz>."""
    path = path or default_dir(domain_langs)
    if not (path / "meta.json").exists():
        if not build_if_missing:
            raise FileNotFoundError(path)
        t0 = time.time()
        build(path, domain_langs)
        print(f"[pp] zbudowano indeks {path} w {time.time() - t0:.0f}s", flush=True)
    lex = CompiledLexicon(path)
    return lex, CompiledLM(path, lex)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="zbuduj skompilowany indeks słownika (raz, przed pierwszym uruchomieniem)")
    ap.add_argument("--domain", default="en,la,pl")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    langs = [x for x in a.domain.split(",") if x]
    out = a.out or default_dir(langs)
    t0 = time.time()
    build(out, langs)
    print(f"{out} ({time.time() - t0:.0f}s)")
