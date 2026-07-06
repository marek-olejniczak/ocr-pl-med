"""Field-content generation for form filling.

Replaces the old pseudo-Polish syllable filler: fields are filled with real
Polish medical vocabulary (the same images later train the OCR model, so the
written content is a training label, not throwaway filler).

Content per field kind:
    t / f -> ~70% medical terms (diagnoses, drugs, dosages, abbreviations,
             departments, ICD descriptions), ~30% personal data (names,
             addresses, cities) — real forms contain both
    n     -> realistic digit content: PESEL, dates, phones, NIP, digit runs
    mix   -> ICD-10 codes, dose strings (2x500mg), document numbers

Width fitting: the caller provides a `measure(text) -> px` callable (usually
font.getlength adjusted for glyph stretch and letter tracking). Content is
grown unit by unit toward a random target of 30-100% of the field width and
hard-capped at 97% so augmentation jitter can't push ink past the field edge.
"""

import random
from typing import Callable

from vocabulary import Vocabulary

# Probability that a text field draws medical content (vs personal data)
MEDICAL_TEXT_PROB = 0.7

_TEXT_MEDICAL = [
    "diagnosis", "icd_description", "drug", "dosage",
    "abbreviation", "department", "hospital_name", "approval",
]
_TEXT_PERSONAL = ["patient_name", "doctor_name", "address", "city"]

# Fill target: fraction of field width the written line should span
FILL_FRAC_MIN = 0.30
FILL_FRAC_MAX = 1.00
HARD_CAP_FRAC = 0.97


def _sample_text_unit(vocab: Vocabulary) -> str:
    pool = _TEXT_MEDICAL if random.random() < MEDICAL_TEXT_PROB else _TEXT_PERSONAL
    text, _ = vocab.get_random_text(random.choice(pool))
    return text


def _random_nip() -> str:
    digits = "".join(random.choice("0123456789") for _ in range(10))
    if random.random() < 0.5:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:8]}-{digits[8:]}"
    return digits


def _random_digit_run() -> str:
    return "".join(random.choice("0123456789") for _ in range(random.randint(1, 6)))


def _sample_number_unit(vocab: Vocabulary) -> str:
    r = random.random()
    if r < 0.20:
        return vocab.get_random_text("pesel")[0]
    if r < 0.45:
        return vocab.get_random_text("date")[0]
    if r < 0.65:
        return vocab.get_random_text("phone")[0]
    if r < 0.75:
        return _random_nip()
    return _random_digit_run()


def _random_doc_number() -> str:
    letters = "".join(random.choice("ABCDEFGHJKLMNPRSTUWXYZ") for _ in range(3))
    return letters + "".join(random.choice("0123456789") for _ in range(7))


def _random_dose() -> str:
    n = random.choice([1, 2, 3])
    amount = random.choice([50, 75, 100, 250, 500, 1000])
    return f"{n}x{amount}mg"


def _sample_mix_unit(vocab: Vocabulary) -> str:
    r = random.random()
    if r < 0.40:
        return vocab.get_random_text("icd_code")[0]
    if r < 0.70:
        return _random_dose()
    return _random_doc_number()


_UNIT_SAMPLERS = {
    "n": _sample_number_unit,
    "mix": _sample_mix_unit,
    # "t" and "f" both fall through to text units
}


def _shrink_to_fit(text: str, measure: Callable[[str], float], cap: float) -> str:
    """Trim text to fit under `cap` px: whole words first, then characters."""
    words = text.split(" ")
    while len(words) > 1 and measure(" ".join(words)) > cap:
        words.pop()
    text = " ".join(words)
    while text and measure(text) > cap:
        text = text[:-1].rstrip()
    return text


def generate_field_content(
    kind: str,
    vocab: Vocabulary,
    measure: Callable[[str], float],
    bbox_w: int,
) -> str:
    """Build field content spanning ~30-100% of the field width.

    Args:
        kind: Field label — "t"/"f" (text), "n" (digits), "mix".
        vocab: Loaded Vocabulary.
        measure: Callable returning the rendered pixel width of a string
            (must already account for glyph stretch and letter tracking).
        bbox_w: Field width in pixels.

    Returns:
        Content string whose measured width is <= 0.97 * bbox_w
        (empty string if even one character doesn't fit).
    """
    sampler = _UNIT_SAMPLERS.get(kind, _sample_text_unit)
    target = bbox_w * random.uniform(FILL_FRAC_MIN, FILL_FRAC_MAX)
    hard_cap = bbox_w * HARD_CAP_FRAC

    text = _shrink_to_fit(sampler(vocab), measure, hard_cap)
    if not text:
        return ""

    while True:
        candidate = text + " " + sampler(vocab)
        w = measure(candidate)
        if w > hard_cap:
            break
        text = candidate
        if w >= target:
            break
    return text
