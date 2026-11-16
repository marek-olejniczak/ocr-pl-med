"""Model kanału Brill & Moore dla par słów (GT → predykcja), ticket 02, spec P3.

P(obs | w) = max (Viterbi) albo suma (forward) po ścieżkach złożonych z jednostek α→β:
  1:1 / usunięcie   a → b, a → ε           P(b|a) Witten-Bell/Dirichlet z priorem jednostajnym
  wstawienie        ε → b  (na slot)       P_ins(b), oraz P(brak wstawienia) na slot
  wieloznakowe      1:2, 2:1, 2:2          z inwentarza (N ≥ min_n, ≥ min_groups grup),
                                          wygładzone w stronę najlepszej dekompozycji na 1:1
Uczenie: hard-EM (iteracja 0 = zwykły Levenshtein), jak analyze_ngram.py (error_analysis).
Liczniki błędów mają limit udziału jednej grupy (`max_group_share`), żeby jeden zeszyt
(np. `cz→w`) nie dominował. Wszystko na małych literach.

`ErrorModel.uniform(...)` to probabilistyczny Levenshtein (jedna stopa błędu + stopa wstawień,
ML z danych): punkt odniesienia z kryterium ukończenia P3.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

NEG_INF = float("-inf")
POLISH = "aąbcćdeęfghijklłmnńoóprsśtuvwxyzźżq-"


@dataclass
class Params:
    iterations: int = 4
    s: float = 1.0               # siła Witten-Bell (k_a = s * N1+(a))
    min_n: int = 3               # min. liczba zdarzeń jednostki wieloznakowej
    min_groups: int = 3          # min. liczba grup z tą jednostką
    s_multi: float = 1.0         # siła priora (dekompozycji) dla jednostek wieloznakowych
    max_group_share: float = 0.5  # limit udziału jednej grupy w liczniku błędu
    multi: bool = True           # ablacja: bez jednostek wieloznakowych
    uniform: bool = False        # ablacja: `ErrorModel.uniform` zamiast wyuczonych pomyłek
    gamma: float = 1.0           # skalar na log P(c→c) (ticket 02 §2), strojony w P4


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def valid_multi(src: str, tgt: str) -> bool:
    if (len(src), len(tgt)) not in {(1, 2), (2, 1), (2, 2)}:
        return False
    if len(src) == 2 and len(tgt) == 2:
        return src[0] != tgt[0] and src[1] != tgt[1]
    if len(src) == 1:
        return src not in tgt
    return tgt not in src


def capped(per_group: Counter, share: float) -> float:
    """Suma liczników z limitem udziału największej grupy (iteracyjnie)."""
    vals = sorted(per_group.values(), reverse=True)
    if len(vals) < 2 or share >= 1:
        return float(sum(vals))
    vals = [float(v) for v in vals]
    for _ in range(len(vals)):
        total = sum(vals)
        top = vals[0]
        if top <= share * total + 1e-9:
            break
        vals[0] = share / (1 - share) * (total - top)
        vals.sort(reverse=True)
    return sum(vals)


class ErrorModel:
    def __init__(self):
        self.l1: dict[tuple[str, str], float] = {}     # log P(b|a), b może być ""
        self.l1_default: dict[str, float] = {}         # log P(niewidziane b | a)
        self.lins: dict[str, float] = {}
        self.lins_default = NEG_INF
        self.lnoins = 0.0
        self.lmulti: dict[tuple[str, str], float] = {}
        self.multi_by_shape: dict[tuple[int, int], dict[tuple[str, str], float]] = {}
        self.unk_src = NEG_INF                          # znak źródłowy spoza alfabetu
        self.unk_id = 0.0
        self.gamma = 1.0
        self.unit = False                               # koszty Levenshteina (iteracja 0)

    # ------------------------------------------------------------------ koszty
    def log1(self, a: str, b: str) -> float:
        if self.unit:
            return 0.0 if a == b else -1.0
        v = self.l1.get((a, b))
        if v is None:
            v = self.l1_default.get(a, self.unk_src)
            if a == b:
                v = self.unk_id
        return v * self.gamma if a == b else v

    def logins(self, b: str) -> float:
        if self.unit:
            return -1.0
        return self.lins.get(b, self.lins_default)

    # ------------------------------------------------------------------ DP
    def logp(self, obs: str, cand: str, forward: bool = False) -> float:
        """log P(obs | cand): cand = słowo źródłowe (GT), obs = predykcja OCR."""
        n, m = len(cand), len(obs)
        nois = 0.0 if self.unit else self.lnoins
        comb = _logsumexp if forward else max
        m12 = self.multi_by_shape.get((1, 2), {})
        m21 = self.multi_by_shape.get((2, 1), {})
        m22 = self.multi_by_shape.get((2, 2), {})
        prev2: list[float] | None = None
        prev: list[float] | None = None
        for i in range(n + 1):
            cur = [NEG_INF] * (m + 1)
            for j in range(m + 1):
                if i == 0 and j == 0:
                    cur[0] = 0.0
                    continue
                v = NEG_INF
                if i:
                    a = cand[i - 1]
                    if j:
                        v = comb(v, prev[j - 1] + self.log1(a, obs[j - 1]) + nois)
                    v = comb(v, prev[j] + self.log1(a, "") + nois)
                if j:
                    v = comb(v, cur[j - 1] + self.logins(obs[j - 1]))
                if i >= 1 and j >= 2 and m12:
                    c = m12.get((cand[i - 1], obs[j - 2:j]))
                    if c is not None:
                        v = comb(v, prev[j - 2] + c + nois)
                if i >= 2 and j >= 1 and m21:
                    c = m21.get((cand[i - 2:i], obs[j - 1]))
                    if c is not None:
                        v = comb(v, prev2[j - 1] + c + 2 * nois)
                if i >= 2 and j >= 2 and m22:
                    c = m22.get((cand[i - 2:i], obs[j - 2:j]))
                    if c is not None:
                        v = comb(v, prev2[j - 2] + c + 2 * nois)
                cur[j] = v
            prev2, prev = prev, cur
        return prev[m] + nois

    def align(self, gt: str, pred: str) -> list[tuple[str, str]]:
        """Najlepsza ścieżka (Viterbi) jako lista jednostek (src, tgt)."""
        n, m = len(gt), len(pred)
        nois = 0.0 if self.unit else self.lnoins
        dp = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
        bk: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0
        shapes = [(k, self.multi_by_shape.get(k, {})) for k in ((1, 2), (2, 1), (2, 2))]
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                best, arg = NEG_INF, None
                if i and j:
                    v = dp[i - 1][j - 1] + self.log1(gt[i - 1], pred[j - 1]) + nois
                    if v > best:
                        best, arg = v, (1, 1)
                if i:
                    v = dp[i - 1][j] + self.log1(gt[i - 1], "") + nois
                    if v > best:
                        best, arg = v, (1, 0)
                if j:
                    v = dp[i][j - 1] + self.logins(pred[j - 1])
                    if v > best:
                        best, arg = v, (0, 1)
                for (di, dj), table in shapes:
                    if table and i >= di and j >= dj:
                        c = table.get((gt[i - di:i], pred[j - dj:j]))
                        if c is not None:
                            v = dp[i - di][j - dj] + c + di * nois
                            if v > best:
                                best, arg = v, (di, dj)
                dp[i][j], bk[i][j] = best, arg
        ops = []
        i, j = n, m
        while i or j:
            di, dj = bk[i][j]
            ops.append((gt[i - di:i], pred[j - dj:j]))
            i, j = i - di, j - dj
        ops.reverse()
        return ops

    # ------------------------------------------------------------------ uczenie
    @classmethod
    def unit_cost(cls) -> "ErrorModel":
        m = cls()
        m.unit = True
        return m

    @classmethod
    def uniform(cls, pairs: Sequence[tuple[str, str, str]]) -> "ErrorModel":
        """Levenshtein probabilistyczny: P(a→a)=1-e, każdy inny wynik (sub/del) e/V; wstawienie r/V na slot."""
        lev = cls.unit_cost()
        chars = errs = ins = slots = 0
        alphabet = set(POLISH)
        for gt, pred, _ in pairs:
            alphabet.update(gt)
            alphabet.update(pred)
            chars += len(gt)
            slots += len(gt) + 1
            for s, t in lev.align(gt, pred):
                if not s:
                    ins += 1
                elif s != t:
                    errs += 1
        V = len(alphabet)
        e, r = max(errs, 0.5) / chars, max(ins, 0.5) / slots
        m = cls()
        m.unk_id = math.log(1 - e)
        m.unk_src = math.log(e / V)
        m.l1_default = {a: math.log(e / V) for a in alphabet}
        m.lins_default = math.log(r / V)
        m.lnoins = math.log(1 - r)
        return m

    @classmethod
    def fit(cls, pairs: Sequence[tuple[str, str, str]], params: Params | None = None) -> "ErrorModel":
        """pairs: (gt_word, pred_word, group), małe litery."""
        p = params or Params()
        if p.uniform:
            return cls.uniform(pairs)
        alphabet = set(POLISH)
        cnt1: Counter = Counter()
        cnt2: Counter = Counter()
        slots = 0
        for gt, pred, _ in pairs:
            alphabet.update(gt)
            alphabet.update(pred)
            cnt1.update(gt)
            cnt2.update(gt[k:k + 2] for k in range(len(gt) - 1))
            slots += len(gt) + 1
        model = cls.unit_cost()
        for _ in range(p.iterations + 1):
            ev: dict[tuple[str, str], Counter] = defaultdict(Counter)
            cand: dict[tuple[str, str], Counter] = defaultdict(Counter)
            for gt, pred, g in pairs:
                ops = model.align(gt, pred)
                for s, t in ops:
                    ev[(s, t)][g] += 1
                k = 0
                while k < len(ops) - 1:
                    (s1, t1), (s2, t2) = ops[k], ops[k + 1]
                    if s1 != t1 and s2 != t2 and len(s1) <= 1 and len(t1) <= 1 and len(s2) <= 1 and len(t2) <= 1:
                        if valid_multi(s1 + s2, t1 + t2):
                            cand[(s1 + s2, t1 + t2)][g] += 1
                            k += 2
                            continue
                    k += 1
            model = cls._estimate(ev, cand, cnt1, cnt2, slots, alphabet, p)
        return model

    @classmethod
    def _estimate(cls, ev, cand, cnt1, cnt2, slots, alphabet, p: Params) -> "ErrorModel":
        m = cls()
        m.gamma = p.gamma
        V = len(alphabet) + 1
        # --- 1:1 i usunięcia
        c1: dict[str, dict[str, float]] = defaultdict(dict)
        used_by_multi: Counter = Counter()
        ins: dict[str, float] = {}
        tot_err = tot_chars = 0.0
        for (s, t), per_g in ev.items():
            if len(s) == 1 and len(t) <= 1:
                c = float(sum(per_g.values())) if s == t else capped(per_g, p.max_group_share)
                c1[s][t] = c
                tot_chars += c
                tot_err += 0 if s == t else c
            elif not s:
                ins[t] = capped(per_g, p.max_group_share)
            else:
                for ch in s:
                    used_by_multi[ch] += sum(per_g.values())
                tot_chars += len(s)
                tot_err += len(s)
        rho = tot_err / max(tot_chars, 1.0)
        err_prior = rho / len(alphabet)  # wyjścia inne niż a: pozostałe znaki alfabetu i usunięcie
        for a in alphabet:
            row = c1.get(a, {})
            n1 = max(1, len(row))
            k = p.s * n1
            denom = sum(row.values()) + used_by_multi[a] + k
            for b, c in row.items():
                m.l1[(a, b)] = math.log((c + k * ((1 - rho) if b == a else err_prior)) / denom)
            m.l1_default[a] = math.log(k * err_prior / denom)
            if (a, a) not in m.l1:
                m.l1[(a, a)] = math.log(k * (1 - rho) / denom)
        m.unk_src = math.log(rho / V)
        m.unk_id = math.log(1 - rho)
        # --- wstawienia
        tot_ins = sum(ins.values())
        k_ins = p.s * max(1, len(ins))
        for b, c in ins.items():
            m.lins[b] = math.log((c + k_ins / V) / (slots + k_ins))
        m.lins_default = math.log((k_ins / V) / (slots + k_ins))
        m.lnoins = math.log(max(1e-6, 1 - (tot_ins + k_ins) / (slots + k_ins)))
        # --- jednostki wieloznakowe
        if p.multi:
            pool: dict[tuple[str, str], Counter] = defaultdict(Counter)
            for d in (cand, {k: v for k, v in ev.items() if len(k[0]) + len(k[1]) >= 3}):
                for k, per_g in d.items():
                    pool[k].update(per_g)
            for (s, t), per_g in pool.items():
                if not valid_multi(s, t):
                    continue
                if sum(per_g.values()) < p.min_n or len(per_g) < p.min_groups:
                    continue
                occ = cnt1[s] if len(s) == 1 else cnt2[s]
                if not occ:
                    continue
                dec = math.exp(m._decomp(s, t))
                prob = (capped(per_g, p.max_group_share) + p.s_multi * dec) / (occ + p.s_multi)
                lp = math.log(min(prob, 1.0)) - len(s) * m.lnoins  # w DP dodajemy no-ins na znak
                if lp > m._decomp(s, t) - len(s) * m.lnoins:
                    m.lmulti[(s, t)] = lp
        for (s, t), lp in m.lmulti.items():
            m.multi_by_shape.setdefault((len(s), len(t)), {})[(s, t)] = lp
        return m

    def inventory(self) -> list[tuple[str, str, float]]:
        return sorted(((s, t, math.exp(lp + len(s) * self.lnoins)) for (s, t), lp in self.lmulti.items()),
                      key=lambda x: -x[2])

    def _decomp(self, s: str, t: str) -> float:
        """log P najlepszej dekompozycji jednostki na 1:1 / indel (z no-ins na znak źródła)."""
        saved = self.multi_by_shape
        self.multi_by_shape = {}
        try:
            return self.logp(t, s) - self.lnoins  # logp dodaje końcowy slot no-ins
        finally:
            self.multi_by_shape = saved


def word_pairs(gt_lines: Iterable[str], pred_lines: Iterable[str], groups: Iterable[str]) -> list[tuple[str, str, str]]:
    """Pary słów (GT, pred, grupa) z wyrównania 1:1 tokenów; rdzenie małymi literami, tylko słowa."""
    from pp.tokenize import is_word, split_token, token_pairs

    out = []
    for gt, pr, g in zip(gt_lines, pred_lines, groups):
        gtok, ptok = gt.split(), pr.split()
        for gi, pi in token_pairs(gtok, ptok):
            a, b = split_token(gtok[gi]).core.lower(), split_token(ptok[pi]).core.lower()
            if is_word(a) and b and len(a) <= 30 and len(b) <= 30:
                out.append((a, b, g))
    return out
