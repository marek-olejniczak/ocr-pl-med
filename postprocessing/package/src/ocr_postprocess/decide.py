"""Reguła decyzji (ticket 04, tryb A: tylko OOV).

Dla tokenu OOV z kluczem k (małe litery) i kandydatów c z prefiltra:
    s(c)    = log P_ch(k | c) + lam * log P(c)
    s_keep  = log P_ch(k | k) + lam * log P_unseen          (k jest poprawnym, niewidzianym słowem)
Korekta na najlepszego b, jeśli:
    s(b) - s_keep >= delta,   s(b) - s(b2) >= mu,   lev(k, b) <= d_max,
    len(k) >= L_min,          count_NKJP(b) >= f_min (filtr kandydatów, przed argmax)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ocr_postprocess.error_model import ErrorModel
from ocr_postprocess.lm import UnigramLM


@dataclass(frozen=True)
class DecisionParams:
    lam: float = 0.5
    delta: float = 2.0
    mu: float = 1.0
    d_max: int = 2
    L_min: int = 4
    f_min: int = 0
    # Ocena kandydatów przez Suryę (ticket 20, krok C): s(c) += w * [logP_surya(linia z c) - logP_surya(oryginał)].
    # w = 0 wyłącza; k kandydatów (top-k wg ch + lam_gen*lp, lev <= 3) dostaje ocenę Suryi.
    w: float = 0.0
    k: int = 5
    lam_gen: float = 0.75
    # Wybór tylko wśród top-k (jak w kroku C, także bez Suryi): na v4 LOGO −1.59 p.p. zamiast −1.22 z pełną listą.
    topk_only: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Scored:
    """Kandydaci jednego klucza ocenieni jednym modelem kanału (niezależne od lam/progów)."""
    key: str
    cands: list[str]
    lev: list[int]
    ch: list[float]
    lp: list[float]
    cnt: list[int]
    keep_ch: float


def score_key(key: str, cands: list[tuple[str, int]], model: ErrorModel, lm: UnigramLM,
              forward: bool = False) -> Scored:
    names = [c for c, _ in cands if c != key]
    levs = [d for c, d in cands if c != key]
    return Scored(key, names, levs, [model.logp(key, c, forward) for c in names], [lm.logp(c) for c in names],
                  [lm.count(c) for c in names], model.logp(key, key, forward))


def best_two(sc: Scored, lam: float, d_max: int, f_min: int) -> tuple[int, float, float]:
    """(indeks najlepszego, s(b), s(b2)); -1 gdy brak kandydatów po filtrach."""
    b, sb, sb2 = -1, -math.inf, -math.inf
    for i in range(len(sc.cands)):
        if sc.lev[i] > d_max or sc.cnt[i] < f_min:
            continue
        s = sc.ch[i] + lam * sc.lp[i]
        if s > sb:
            b, sb, sb2 = i, s, sb
        elif s > sb2:
            sb2 = s
    return b, sb, sb2


def decide(sc: Scored, p: DecisionParams, lp_unseen: float) -> str | None:
    if len(sc.key) < p.L_min:
        return None
    b, sb, sb2 = best_two(sc, p.lam, p.d_max, p.f_min)
    if b < 0:
        return None
    s_keep = sc.keep_ch + p.lam * lp_unseen
    if sb - s_keep >= p.delta and sb - sb2 >= p.mu:
        return sc.cands[b]
    return None


def top_k(sc: Scored, k: int, lam_gen: float) -> list[int]:
    """Indeksy kandydatów do oceny Suryą (kolejność jak przy generowaniu w eksperymencie)."""
    idx = [j for j in range(len(sc.cands)) if sc.lev[j] <= 3]
    return sorted(idx, key=lambda j: -(sc.ch[j] + lam_gen * sc.lp[j]))[:k]


def decide_rescored(sc: Scored, idx: list[int], sd: list[float], p: DecisionParams, lp_unseen: float) -> str | None:
    """Jak `decide`, ale tylko wśród `idx` i z dodatkiem w*sd (sd[n] dla idx[n])."""
    if len(sc.key) < p.L_min:
        return None
    b, sb, sb2 = -1, -math.inf, -math.inf
    for j, d in zip(idx, sd):
        if sc.cnt[j] < p.f_min:
            continue
        s = sc.ch[j] + p.lam * sc.lp[j] + p.w * d
        if s > sb:
            b, sb, sb2 = j, s, sb
        elif s > sb2:
            sb2 = s
    if b >= 0 and sb - (sc.keep_ch + p.lam * lp_unseen) >= p.delta and sb - sb2 >= p.mu:
        return sc.cands[b]
    return None
