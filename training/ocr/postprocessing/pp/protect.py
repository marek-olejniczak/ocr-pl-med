"""Klasy tokenów chronionych P1–P8 (ticket 04 §3). Token chroniony nie jest korygowany.

Listy skrótów: PoliMorf 2026 (tagi brev:*), TA98, oraz skróty z GT **tylko grup treningowych
danego foldu** (`train_abbreviations`). Nigdy z GT grupy testowej (przeciek, map.md).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from pp.tokenize import Token, split_token

UPPER = "A-ZĄĆĘŁŃÓŚŹŻ"
RE_SYMBOLS = re.compile(r"^[\W_]+$")
RE_NUM_UNIT = re.compile(r"^\d+[.,]?\d*[a-zµ%]+$", re.IGNORECASE)
RE_ABBR_DOT = re.compile(r"^[^\W\d_]{1,5}\.$")
RE_ABBR_MULTI = re.compile(r"^([^\W\d_]{1,3}\.){2,}")
RE_SHORT_UPPER = re.compile(rf"^[{UPPER}]{{2,3}}\.?$")

CLASSES = ("P1_symbols", "P2_digits", "P3_num_unit", "P4_abbr_dot", "P5_abbr_list",
           "P6_short_upper", "P7_len_le_2", "P8_hyphen")


def upper_share(line_tokens: Sequence[Token]) -> float:
    letters = [c for t in line_tokens for c in t.core if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


def protected_class(tok: Token, line_upper_share: float, abbrev: set[str] | frozenset[str],
                    disabled: Iterable[str] = ()) -> str | None:
    """Pierwsza pasująca klasa albo None."""
    off = set(disabled)
    raw, core = tok.raw, tok.core
    tail = core + tok.post[:1]  # skrót z kropką: rdzeń + pierwsza brzegowa kropka
    checks = (
        ("P1_symbols", lambda: not core or bool(RE_SYMBOLS.match(raw))),
        ("P2_digits", lambda: any(c.isdigit() for c in raw)),
        ("P3_num_unit", lambda: bool(RE_NUM_UNIT.match(core))),
        ("P4_abbr_dot", lambda: (tok.post.startswith(".") and bool(RE_ABBR_DOT.match(tail))
                                 and (core.islower() or core.lower() in abbrev))
                                or bool(RE_ABBR_MULTI.match(core + tok.post))),
        ("P5_abbr_list", lambda: core.lower() in abbrev),
        ("P6_short_upper", lambda: bool(RE_SHORT_UPPER.match(core + tok.post[:1])) and line_upper_share < 0.7),
        ("P7_len_le_2", lambda: len(core) <= 2),
        ("P8_hyphen", lambda: "-" in core or raw.startswith("-") or raw.endswith("-")),
    )
    for name, f in checks:
        if name not in off and f():
            return name
    return None


def train_abbreviations(gt_lines: Iterable[str], known) -> set[str]:
    """Skróty z GT grup treningowych: krótki token z kropką, rdzeń spoza słownika."""
    out = set()
    for line in gt_lines:
        for raw in line.split():
            t = split_token(raw)
            k = t.core.lower()
            if t.post.startswith(".") and 1 <= len(k) <= 6 and k.isalpha() and not known(k):
                out.add(k)
    return out
