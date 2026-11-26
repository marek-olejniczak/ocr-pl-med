"""API korektora: `Corrector.correct_lines(lines)` (partią, bo prefiltr jest szybki tylko w partii).

    from pp.correct import Corrector
    corr = Corrector.from_training(gt_lines, pred_lines, groups, params)
    corr.correct_lines(["linia z OCR", ...])
"""

from __future__ import annotations

import math
from typing import Sequence

from pp.candidates import Prefilter
from pp.decide import DecisionParams, decide, decide_rescored, score_key, top_k
from pp.error_model import ErrorModel, Params, word_pairs
from pp.lexicon import Lexicon, domain_terms
from pp.lm import UnigramLM
from pp.normalize import hygiene
from pp.protect import protected_class, train_abbreviations, upper_share
from pp.tokenize import is_word, restore_case, tokenize


class Corrector:
    def __init__(self, lexicon: Lexicon, lm: UnigramLM, model: ErrorModel, params: DecisionParams,
                 abbrev: set[str], prefilter: Prefilter | None = None, disabled_protect: Sequence[str] = ()):
        self.lex, self.lm, self.model, self.p = lexicon, lm, model, params
        self.abbrev = abbrev
        self.pref = prefilter or Prefilter(lexicon, d_max=max(params.d_max, 1))
        self.lp_unseen = math.log(lm.unseen) - lm.logz
        self.disabled = tuple(disabled_protect)
        self.p_rescored: DecisionParams | None = None  # progi dla correct_lines_rescored (z paczki)

    @classmethod
    def from_training(cls, gt_lines, pred_lines, groups, params: DecisionParams = DecisionParams(),
                      em_params: Params | None = None, lexicon: Lexicon | None = None, lm: UnigramLM | None = None,
                      domain_langs: Sequence[str] = ()):
        """`domain_langs`: języki listy terminów TA, np. ("la", "pl"); puste = lista wyłączona.

        Progi stroimy bez listy, więc jej zmiana nie wymaga ponownego strojenia.
        """
        lex = lexicon or Lexicon(extra_known=domain_terms(domain_langs))
        lm = lm or UnigramLM(lex.targets())
        model = ErrorModel.fit(word_pairs(gt_lines, pred_lines, groups), em_params)
        abbrev = lex.abbrev | train_abbreviations(gt_lines, lex.known)
        return cls(lex, lm, model, params, abbrev)

    def eligible(self, tok, ups: float, p: DecisionParams | None = None) -> bool:
        return (is_word(tok.core) and not self.lex.known(tok.key) and len(tok.key) >= (p or self.p).L_min
                and protected_class(tok, ups, self.abbrev, self.disabled) is None)

    def correct_lines(self, lines: Sequence[str]) -> list[str]:
        toks = [tokenize(hygiene(l)) for l in lines]
        ups = [upper_share(t) for t in toks]
        keys = {t.key for line, u in zip(toks, ups) for t in line if self.eligible(t, u)}
        cands = self.pref.batch(keys, 3 if self.p.topk_only else self.p.d_max) if keys else {}
        cache: dict[str, str | None] = {}
        out = []
        for line, u in zip(toks, ups):
            parts = []
            for t in line:
                new = None
                if t.key in cands:
                    if t.key not in cache:
                        sc = score_key(t.key, cands[t.key], self.model, self.lm)
                        if self.p.topk_only:
                            idx = top_k(sc, self.p.k, self.p.lam_gen)
                            cache[t.key] = decide_rescored(sc, idx, [0.0] * len(idx), self.p, self.lp_unseen)
                        else:
                            cache[t.key] = decide(sc, self.p, self.lp_unseen)
                    new = cache[t.key]
                parts.append(t.rebuild(restore_case(t.core, new)) if new else t.raw)
            out.append(" ".join(parts))
        return out

    def correct_lines_rescored(self, lines: Sequence[str], images: Sequence, scorer,
                               params: DecisionParams | None = None) -> list[str]:
        """Korekta z oceną kandydatów przez Suryę (`pp.surya_score.SuryaScorer`); `images[i]` = obraz linii `lines[i]` (PIL).

        Wymaga progów strojonych z w > 0 (deploy bundle `*_rescored`). Każdy token rozstrzygany niezależnie:
        warianty linii różnią się jednym tokenem względem wyjścia OCR (jak w eksperymencie).
        """
        p = params or self.p_rescored
        if p is None or p.w <= 0:
            raise ValueError("brak progów z oceną Suryi (params_rescored w paczce)")
        toks = [tokenize(hygiene(l)) for l in lines]
        ups = [upper_share(t) for t in toks]
        keys = {t.key for line, u in zip(toks, ups) for t in line if self.eligible(t, u, p)}
        cands = self.pref.batch(keys, 3) if keys else {}
        scored = {k: score_key(k, c, self.model, self.lm) for k, c in cands.items()}
        out = []
        for line, u, img in zip(toks, ups, images):
            base = [t.raw for t in line]
            jobs = []  # (pozycja, sc, idx)
            for pi, t in enumerate(line):
                sc = scored.get(t.key)
                if sc is None or not self.eligible(t, u, p):
                    continue
                idx = top_k(sc, p.k, p.lam_gen)
                if idx:
                    jobs.append((pi, sc, idx))
            parts = list(base)
            if jobs:
                variants = []
                for pi, sc, idx in jobs:
                    for j in idx:
                        v = list(base)
                        v[pi] = line[pi].rebuild(restore_case(line[pi].core, sc.cands[j]))
                        variants.append(" ".join(v))
                lps = scorer.score(img, [" ".join(base)] + variants)
                ref, pos = lps[0], 1
                for pi, sc, idx in jobs:
                    sd = [x - ref for x in lps[pos:pos + len(idx)]]
                    pos += len(idx)
                    new = decide_rescored(sc, idx, sd, p, self.lp_unseen)
                    if new:
                        parts[pi] = line[pi].rebuild(restore_case(line[pi].core, new))
            out.append(" ".join(parts))
        return out

    def correct_lines_detailed(self, lines: Sequence[str]) -> list[dict]:
        """Jak correct_lines, plus lista zmian per linia: indeks tokenu (po higienie), przed i po (do podświetlenia w UI)."""
        out = []
        for line, new in zip(lines, self.correct_lines(lines)):
            before, after = hygiene(line).split(), new.split()
            changes = [{"token": i, "from": b, "to": c} for i, (b, c) in enumerate(zip(before, after)) if b != c] \
                if len(before) == len(after) else []
            out.append({"text": new, "changes": changes})
        return out

    def correct(self, line: str) -> str:
        return self.correct_lines([line])[0]
