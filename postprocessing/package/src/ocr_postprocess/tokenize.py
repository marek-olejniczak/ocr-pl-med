"""Tokenizacja linii (ticket 04 §4): tokeny po białych znakach, brzegowa interpunkcja odcięta,
wzorzec wielkości liter przenoszony z tokenu obserwowanego na kandydata.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein


def _edge(c: str) -> bool:
    return unicodedata.category(c)[0] in "PS"


@dataclass(frozen=True)
class Token:
    raw: str
    pre: str
    core: str
    post: str

    @property
    def key(self) -> str:
        return self.core.lower()

    def rebuild(self, new_core: str) -> str:
        return self.pre + new_core + self.post


def split_token(raw: str) -> Token:
    i, j = 0, len(raw)
    while i < j and _edge(raw[i]):
        i += 1
    while j > i and _edge(raw[j - 1]):
        j -= 1
    return Token(raw, raw[:i], raw[i:j], raw[j:])


def tokenize(line: str) -> list[Token]:
    return [split_token(t) for t in line.split()]


def is_word(core: str) -> bool:
    """Same litery (dopuszczalny wewnętrzny dywiz), co najmniej jedna litera."""
    return bool(core) and any(c.isalpha() for c in core) and all(c.isalpha() or c == "-" for c in core)


def case_pattern(core: str) -> str:
    letters = [c for c in core if c.isalpha()]
    if not letters:
        return "none"
    if all(c.isupper() for c in letters):
        return "upper" if len(letters) > 1 else "title"
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "title"
    if all(c.islower() for c in letters):
        return "lower"
    return "mixed"


def restore_case(observed: str, candidate: str) -> str:
    p = case_pattern(observed)
    if p == "upper":
        return candidate.upper()
    if p == "title":
        return candidate[:1].upper() + candidate[1:]
    if p == "mixed":
        # pozycyjnie tam, gdzie długości pozwalają; reszta małymi
        return "".join(c.upper() if i < len(observed) and observed[i].isupper() else c
                       for i, c in enumerate(candidate))
    return candidate


def token_pairs(gt_tokens: list[str], pred_tokens: list[str]) -> list[tuple[int, int]]:
    """Pary indeksów (GT, pred) z wyrównania Levenshteina na tokenach (equal + replace, 1:1)."""
    out = []
    for op in Levenshtein.opcodes(gt_tokens, pred_tokens):
        if op.tag in ("equal", "replace"):
            for k in range(min(op.src_end - op.src_start, op.dest_end - op.dest_start)):
                out.append((op.src_start + k, op.dest_start + k))
    return out
