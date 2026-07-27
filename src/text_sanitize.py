"""Keep generated content renderable by every handwriting font we ship.

Custom fonts built from real handwriting cover letters, digits and Polish
diacritics but not the typographic oddities our medical vocabulary carries
(ICD-10 daggers, en dashes, micro signs). Rendering those produces an empty
.notdef box while the ground-truth transcription still claims a character is
there — poison for the OCR model that trains on this data.

Two stages, in this order:
    normalize_for_handwriting — replace typography with what a hand writes
    strip_unsupported         — drop whatever the chosen font still lacks
"""

import re
from functools import lru_cache

from fontTools.ttLib import TTFont

# Typographic character -> what a person filling the form would actually draw
CHAR_REPLACEMENTS: dict[str, str] = {
    "‑": "-",   # non-breaking hyphen
    "–": "-",   # en dash
    "—": "-",   # em dash
    "µ": "u",   # micro sign: 100µg -> 100ug
    "μ": "u",   # greek small mu
    "†": "",    # dagger — ICD-10 aetiology marker, never handwritten
    "‡": "",    # double dagger
    "[": "(",
    "]": ")",
    "„": '"',
    "”": '"',
    "“": '"',
    "’": "'",
    " ": " ",   # non-breaking space
}

_MULTISPACE = re.compile(r"\s{2,}")


def normalize_for_handwriting(text: str) -> str:
    """Replace typographic characters with their handwritten equivalents.

    Dropping a character can leave a double space behind, so whitespace is
    collapsed afterwards.
    """
    out = "".join(CHAR_REPLACEMENTS.get(ch, ch) for ch in text)
    return _MULTISPACE.sub(" ", out).strip()


@lru_cache(maxsize=64)
def font_charset(font_path: str) -> frozenset[int]:
    """Return the unicode code points a font can render.

    Empty frozenset means the file could not be parsed; callers treat that
    as "unknown" and leave the text alone rather than deleting everything.
    """
    try:
        font = TTFont(font_path, fontNumber=0, lazy=True)
    except Exception:
        return frozenset()
    try:
        codes: set[int] = set()
        for table in font["cmap"].tables:
            codes |= set(table.cmap.keys())
        return frozenset(codes)
    except Exception:
        return frozenset()
    finally:
        font.close()


def strip_unsupported(text: str, font_path: str) -> str:
    """Drop characters the font has no glyph for. Spaces are always kept."""
    codes = font_charset(font_path)
    if not codes:
        return text
    out = "".join(ch for ch in text if ch == " " or ord(ch) in codes)
    return _MULTISPACE.sub(" ", out).strip()


def sanitize(text: str, font_path: str) -> str:
    """Normalize typography, then drop anything the font still cannot draw."""
    return strip_unsupported(normalize_for_handwriting(text), font_path)
