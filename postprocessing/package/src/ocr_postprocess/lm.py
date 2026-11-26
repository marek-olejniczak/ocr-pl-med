"""P(w): unigramy NKJP (podkorpus zrównoważony) + Simple Good-Turing (ticket 05).

Liczności c ≥ 1 wygładzone SGT (Gale & Sampson 1995), formy słownika niewidziane w NKJP
dostają stały licznik `unseen_count` (domyślnie 0.1). P(w) = c*(w) / Z, Z = suma po
formach-kandydatach. Tylko ranking i skala log P(w) mają znaczenie (waga lambda w P4).
"""

from __future__ import annotations

import gzip
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ocr_postprocess.resources import cache_dir, resource_path

NKJP = "nkjp-1grams.gz"
WORD_RE = re.compile(r"^[a-ząćęłńóśźż]+$")


@lru_cache(maxsize=None)
def nkjp_counts() -> dict[str, int]:
    cached = cache_dir() / "nkjp-1grams.clean.tsv"
    if not cached.exists():
        c: Counter = Counter()
        with gzip.open(resource_path(NKJP), "rt", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2 and WORD_RE.match(parts[1]):
                    c[parts[1]] += int(parts[0])
        tmp = cached.with_suffix(".part")
        with open(tmp, "w", encoding="utf-8") as out:
            for w, n in c.most_common():
                out.write(f"{w}\t{n}\n")
        tmp.replace(cached)
    d = {}
    with open(cached, encoding="utf-8") as f:
        for line in f:
            w, n = line.rstrip("\n").split("\t")
            d[w] = int(n)
    return d


def simple_good_turing(freq_of_freq: dict[int, int], confidence: float = 1.96) -> dict[int, float]:
    """r -> r* wg Gale & Sampson (1995), z regułą przełączenia Turing → LGT."""
    rs = sorted(freq_of_freq)
    n = [freq_of_freq[r] for r in rs]
    z = []
    for i, r in enumerate(rs):
        q = rs[i - 1] if i else 0
        t = rs[i + 1] if i + 1 < len(rs) else 2 * r - q
        z.append(n[i] / (0.5 * (t - q)))
    lx = [math.log(r) for r in rs]
    lz = [math.log(v) for v in z]
    mx, mz = sum(lx) / len(lx), sum(lz) / len(lz)
    b = sum((x - mx) * (y - mz) for x, y in zip(lx, lz)) / sum((x - mx) ** 2 for x in lx)
    a = mz - b * mx

    def S(r: float) -> float:
        return math.exp(a + b * math.log(r))

    rstar: dict[int, float] = {}
    use_lgt = False
    for i, r in enumerate(rs):
        y = (r + 1) * S(r + 1) / S(r)
        if not use_lgt:
            if i + 1 < len(rs) and rs[i + 1] == r + 1:
                x = (r + 1) * n[i + 1] / n[i]
                sd = confidence * math.sqrt((r + 1) ** 2 * n[i + 1] / n[i] ** 2 * (1 + n[i + 1] / n[i]))
                if abs(x - y) > sd:
                    rstar[r] = x
                    continue
            use_lgt = True
        rstar[r] = y
    return rstar


class UnigramLM:
    def __init__(self, vocab: Iterable[str], counts: dict[str, int] | None = None, unseen_count: float = 0.1):
        counts = nkjp_counts() if counts is None else counts
        rstar = simple_good_turing(Counter(counts.values()))
        self.counts = counts
        self.rstar = rstar
        self.unseen = unseen_count
        z = 0.0
        for w in vocab:
            z += self.adjusted(w)
        self.logz = math.log(z)

    def adjusted(self, w: str) -> float:
        c = self.counts.get(w, 0)
        return self.rstar.get(c, float(c)) if c else self.unseen

    def logp(self, w: str) -> float:
        return math.log(self.adjusted(w)) - self.logz

    def count(self, w: str) -> int:
        return self.counts.get(w, 0)
