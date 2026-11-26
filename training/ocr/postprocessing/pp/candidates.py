"""Generowanie kandydatów (ticket 03): prefiltr rapidfuzz Lev ≤ d_max w kubełkach długości,
potem rescoring modelem kanału (pp/error_model) + P(w) w pp/decide.

Prefiltr nie zależy od modelu błędu, więc w CV liczymy go raz dla wszystkich kluczy
(`Prefilter.batch`), a rescoring per fold. `cdist(..., workers=-1)` liczy równolegle.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from pp.lexicon import Lexicon, fold


class Prefilter:
    def __init__(self, lexicon: Lexicon, d_max: int = 3, workers: int = -1):
        self.d_max = d_max
        self.workers = workers
        self.by_len = lexicon.by_length()
        if hasattr(lexicon, "fold_group"):  # pp.compiled.CompiledLexicon: grupy z posortowanej listy, bez słownika
            self._fold_group = lexicon.fold_group
        else:
            by_fold = lexicon.by_fold()
            self._fold_group = lambda key: by_fold.get(fold(key), ())

    def _diacritic_variants(self, key: str, out: dict[str, int]) -> None:
        # warianty diakrytyczne zawsze (opcja 2): przy wielu diakrytykach mogą mieć Lev > d
        for w in self._fold_group(key):
            out.setdefault(w, Levenshtein.distance(key, w))

    def __call__(self, key: str, d_max: int | None = None) -> list[tuple[str, int]]:
        return self.batch([key], d_max)[key]

    def batch(self, keys: Iterable[str], d_max: int | None = None, chunk: int = 256) -> dict[str, list[tuple[str, int]]]:
        d = self.d_max if d_max is None else d_max
        by_n: dict[int, list[str]] = defaultdict(list)
        for k in set(keys):
            by_n[len(k)].append(k)
        res: dict[str, dict[str, int]] = {}
        for n, ks in by_n.items():
            for k in ks:
                res[k] = {}
            for L in range(max(1, n - d), n + d + 1):
                bucket = self.by_len.get(L)
                if not bucket:
                    continue
                for c0 in range(0, len(ks), chunk):
                    part = ks[c0:c0 + chunk]
                    mat = process.cdist(part, bucket, scorer=Levenshtein.distance, score_cutoff=d,
                                        dtype=np.uint8, workers=self.workers)
                    rows, cols = np.nonzero(mat <= d)
                    for r, c in zip(rows.tolist(), cols.tolist()):
                        res[part[r]][bucket[c]] = int(mat[r, c])
        for k, out in res.items():
            self._diacritic_variants(k, out)
        return {k: sorted(v.items(), key=lambda x: (x[1], x[0])) for k, v in res.items()}
