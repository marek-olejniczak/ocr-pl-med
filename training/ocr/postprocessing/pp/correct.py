"""API korektora: `Corrector.correct_lines(lines)` (partią, bo prefiltr jest szybki tylko w partii).

    from pp.correct import Corrector
    corr = Corrector.from_training(gt_lines, pred_lines, groups, params)
    corr.correct_lines(["linia z OCR", ...])
"""

from __future__ import annotations

import math
from typing import Sequence

from pp.candidates import Prefilter
from pp.decide import DecisionParams, decide, score_key
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

    def eligible(self, tok, ups: float) -> bool:
        return (is_word(tok.core) and not self.lex.known(tok.key) and len(tok.key) >= self.p.L_min
                and protected_class(tok, ups, self.abbrev, self.disabled) is None)

    def correct_lines(self, lines: Sequence[str]) -> list[str]:
        toks = [tokenize(hygiene(l)) for l in lines]
        ups = [upper_share(t) for t in toks]
        keys = {t.key for line, u in zip(toks, ups) for t in line if self.eligible(t, u)}
        cands = self.pref.batch(keys, self.p.d_max) if keys else {}
        cache: dict[str, str | None] = {}
        out = []
        for line, u in zip(toks, ups):
            parts = []
            for t in line:
                new = None
                if t.key in cands:
                    if t.key not in cache:
                        cache[t.key] = decide(score_key(t.key, cands[t.key], self.model, self.lm), self.p,
                                              self.lp_unseen)
                    new = cache[t.key]
                parts.append(t.rebuild(restore_case(t.core, new)) if new else t.raw)
            out.append(" ".join(parts))
        return out

    def correct(self, line: str) -> str:
        return self.correct_lines([line])[0]
