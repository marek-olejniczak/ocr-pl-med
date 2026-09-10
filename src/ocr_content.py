"""Content for standalone OCR line renders.

The real test material is student anatomy notebooks, not filled-in forms, and
the errors cluster on what forms never produce: single short words, headings
written in capitals, bullet points and arrows between terms. This module
builds that kind of text while keeping the form-style generators around as a
minority share.

Each behaviour is a separately switchable "family" so an ablation can turn one
off and keep the rest:

    short_words     single words / two-word terms of 3-12 characters
    caps            capitalised headings and capitalised first letters
    arrows_bullets  "- ", "1) ", "a) " prefixes and "->" / "=>" between terms
    anatomy_vocab   anatomy / disease / lab-test pools instead of form fill-ins,
                    which also drops the digit share from ~22% to a few percent
"""

import random
import re
from pathlib import Path
from typing import Callable, Optional

from field_content import _shrink_to_fit, generate_field_content
from text_sanitize import font_charset, normalize_for_handwriting, sanitize
from vocabulary import Vocabulary

CONTENT_FAMILIES = ("short_words", "caps", "arrows_bullets", "anatomy_vocab")

# Old form-style mix, used when anatomy_vocab is switched off.
FORM_CONTENT_KINDS = [("t", 0.70), ("n", 0.22), ("mix", 0.08)]

# Line kinds when every family is on. Weights are relative.
LINE_KINDS = {
    "word": 0.34,       # one short word, sometimes two
    "phrase": 0.30,     # anatomy / medical term, 2-6 words
    "heading": 0.08,    # capitals
    "bullet": 0.08,     # "- phrase", "1) phrase"
    "arrow": 0.06,      # "term -> term"
    "form_text": 0.08,  # the old "t" generator
    "number": 0.03,     # the old "n" generator
    "mix": 0.03,        # the old "mix" generator
}
# Family that owns each kind; kinds without an owner are always available.
KIND_FAMILY = {
    "word": "short_words",
    "heading": "caps",
    "bullet": "arrows_bullets",
    "arrow": "arrows_bullets",
}
# Probability that a phrase gets its first letter capitalised (caps family).
CAPITALISE_PROB = 0.12
# Probability that a "word" line carries two words instead of one.
TWO_WORD_PROB = 0.25
SHORT_WORD_LEN = (3, 12)

BULLET_PREFIXES = ["- ", "- ", "- ", "• ", "* ", "1) ", "2) ", "3) ", "a) ", "b) ", "c) ", "1. ", "2. "]
ARROWS = ["->", "->", "->", "=>"]

_POOL_FILES = {
    "anatomy": ("anatomy_seed.txt",),
    "anatomy_llm": ("anatomy_phrases.txt",),
    "diseases": ("icd11_disease_names.txt",),
    "tests": ("lab_tests.txt",),
    "procedures": ("common_procedures.txt",),
    "conditions": ("common_conditions.txt",),
}
# How often each pool is drawn for a phrase. Anatomy dominates because that
# is what the notebooks are about.
# The LLM pool reads like a student who half-remembers the lecture (real
# words in odd combinations), so it gets a smaller share than the curated
# seed list.
_POOL_WEIGHTS = {"anatomy": 0.44, "anatomy_llm": 0.18, "diseases": 0.14,
                 "tests": 0.08, "procedures": 0.08, "conditions": 0.08}

_WORD_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+(?:-[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+)?")


class LinePools:
    """Phrase and word pools for notebook-style lines."""

    def __init__(self, resource_dir: str) -> None:
        base = Path(resource_dir)
        self.phrases: dict[str, list[str]] = {}
        for name, candidates in _POOL_FILES.items():
            lines: list[str] = []
            for filename in candidates:
                path = base / filename
                if path.exists():
                    lines.extend(
                        l.strip() for l in path.read_text(encoding="utf-8").splitlines()
                        if l.strip() and not l.startswith("#")
                    )
            if name == "anatomy_llm":
                # The model capitalises every line; notes mostly do not. The
                # caps family adds capitals back deliberately.
                lines = [_decapitalise(l) for l in lines]
            self.phrases[name] = sorted(set(lines))
        if not self.phrases["anatomy"]:
            raise FileNotFoundError(
                f"no anatomy pool in {base} (anatomy_seed.txt or anatomy_phrases.txt)")

        # Word pool: every word of 3-12 letters from the anatomy pool, weighted
        # by how often it occurs there, plus a thinner slice of disease words.
        counts: dict[str, int] = {}
        for phrase in self.phrases["anatomy"]:
            for word in _WORD_RE.findall(phrase):
                if SHORT_WORD_LEN[0] <= len(word) <= SHORT_WORD_LEN[1]:
                    counts[word.lower()] = counts.get(word.lower(), 0) + 3
        for pool in ("diseases", "conditions", "procedures"):
            for phrase in self.phrases[pool]:
                for word in _WORD_RE.findall(phrase):
                    if SHORT_WORD_LEN[0] <= len(word) <= SHORT_WORD_LEN[1]:
                        counts[word.lower()] = counts.get(word.lower(), 0) + 1
        self.words = list(counts.keys())
        self.word_weights = [counts[w] for w in self.words]

    def phrase(self) -> str:
        pools = [p for p in _POOL_WEIGHTS if self.phrases.get(p)]
        weights = [_POOL_WEIGHTS[p] for p in pools]
        pool = random.choices(pools, weights=weights, k=1)[0]
        return random.choice(self.phrases[pool])

    def word(self) -> str:
        return random.choices(self.words, weights=self.word_weights, k=1)[0]


def pick_line_kind(enabled: set[str]) -> str:
    kinds = [k for k in LINE_KINDS if KIND_FAMILY.get(k) is None or KIND_FAMILY[k] in enabled]
    weights = [LINE_KINDS[k] for k in kinds]
    return random.choices(kinds, weights=weights, k=1)[0]


def _decapitalise(text: str) -> str:
    """Lower a leading capital unless it starts an abbreviation ("C1-C7")."""
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return text[0].lower() + text[1:]
    return text


def _capitalise(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _compose(kind: str, pools: LinePools, enabled: set[str],
             charset: Optional[frozenset[int]] = None) -> str:
    """Raw text for one line of the given kind (not yet fitted to width)."""
    if kind == "word":
        text = pools.word()
        if random.random() < TWO_WORD_PROB:
            text = f"{text} {pools.word()}"
    elif kind == "phrase":
        text = pools.phrase()
        if random.random() < 0.15:
            text = f"{text} {random.choice(['prawy', 'lewy', 'prawa', 'lewa', 'górny', 'dolny', 'przedni', 'tylny'])}"
    elif kind == "heading":
        text = pools.phrase() if random.random() < 0.6 else pools.word()
        text = text.upper()
    elif kind == "bullet":
        prefixes = [p for p in BULLET_PREFIXES
                    if charset is None or all(ord(c) in charset for c in p.strip())]
        text = random.choice(prefixes or ["- "]) + pools.phrase()
    elif kind == "arrow":
        parts = [pools.phrase() if random.random() < 0.5 else pools.word()
                 for _ in range(random.choice([2, 2, 2, 3]))]
        arrow = random.choice(ARROWS)
        text = f" {arrow} ".join(parts)
    else:
        raise ValueError(kind)

    if "caps" in enabled and kind in ("phrase", "bullet", "arrow", "word"):
        if random.random() < CAPITALISE_PROB:
            text = _capitalise(text)
    return text


def generate_line_content(
    pools: Optional[LinePools],
    vocab: Vocabulary,
    measure: Callable[[str], float],
    target_width: int,
    font_path: str,
    enabled: set[str],
) -> tuple[str, str]:
    """Return (text, kind) for one line, fitted to `target_width` pixels.

    With anatomy_vocab disabled (or no pools) this falls back to the old
    form-style generators, so an ablation compares like with like.
    """
    if pools is None or "anatomy_vocab" not in enabled:
        kinds, weights = zip(*FORM_CONTENT_KINDS)
        kind = random.choices(kinds, weights=weights, k=1)[0]
        return generate_field_content(kind, vocab, measure, target_width, font_path=font_path), kind

    kind = pick_line_kind(enabled)
    # The custom hands (Marek_*, Pokladowski) have no ">" glyph, so an arrow
    # line cannot be written in them; it becomes a plain phrase instead.
    if kind == "arrow" and font_path and ord(">") not in font_charset(font_path):
        kind = "phrase"
    if kind in ("form_text", "number", "mix"):
        field_kind = {"form_text": "t", "number": "n", "mix": "mix"}[kind]
        return generate_field_content(field_kind, vocab, measure, target_width, font_path=font_path), kind

    for _ in range(6):
        raw = _compose(kind, pools, enabled,
                       font_charset(font_path) if font_path else None)
        text = sanitize(raw, font_path) if font_path else normalize_for_handwriting(raw)
        text = " ".join(text.split())
        if not text:
            continue
        # Short words set their own width; longer kinds are trimmed to the
        # requested width like a form field would be.
        if kind != "word":
            text = _shrink_to_fit(text, measure, target_width)
            text = text.rstrip(" -•*>=,;:(")
            # A trimmed line should not end on an orphaned abbreviation
            # ("unerwienie: n.").
            words = text.split()
            if len(words) > 1 and len(words[-1]) <= 2 and words[-1].endswith((".", ":")):
                text = " ".join(words[:-1]).rstrip(" ,;:")
        # Trimming may have eaten the very thing the kind is about; try again
        # rather than label a plain phrase as an arrow line.
        if kind == "arrow" and not any(a in text for a in ARROWS):
            continue
        # Fonts without a "•" glyph lose the marker in sanitize; retry.
        if kind == "bullet" and (len(text.split()) < 2 or text[0] not in "-•*0123456789abc"):
            continue
        if text and not text.isspace():
            return text, kind
    return "", kind
