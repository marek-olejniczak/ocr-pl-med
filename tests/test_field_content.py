import random

import pytest

from field_content import generate_field_content, _shrink_to_fit


class FakeVocab:
    """Deterministic stand-in for Vocabulary — no resource loading."""

    SAMPLES = {
        "diagnosis": "nadciśnienie tętnicze",
        "icd_description": "Ostre zakażenie górnych dróg oddechowych",
        "drug": "Amoksycylina",
        "dosage": "2x dziennie",
        "abbreviation": "Rp.",
        "department": "Oddział Kardiologii",
        "hospital_name": "Szpital Miejski w Radomiu",
        "approval": "Pilne badanie",
        "patient_name": "Anna Kowalska",
        "doctor_name": "lek. Jan Nowak",
        "address": "ul. Polna 12, 26-600 Radom",
        "city": "Radom",
        "pesel": "44051401359",
        "date": "12.03.2021",
        "phone": "601 202 303",
        "icd_code": "J45.0",
    }

    def get_random_text(self, category):
        return (self.SAMPLES[category], category)


def char_measure(text: str) -> float:
    return len(text) * 10.0  # 10 px per char


def test_shrink_word_wise_then_char_wise():
    assert _shrink_to_fit("abc def ghi", char_measure, cap=80.0) == "abc def"
    # single word wider than cap -> trimmed char by char
    assert _shrink_to_fit("abcdefghij", char_measure, cap=50.0) == "abcde"
    assert _shrink_to_fit("abc", char_measure, cap=5.0) == ""


@pytest.mark.parametrize("kind", ["t", "f", "n", "mix"])
def test_content_fits_hard_cap(kind):
    random.seed(7)
    vocab = FakeVocab()
    for _ in range(200):
        bbox_w = random.randint(60, 1200)
        text = generate_field_content(kind, vocab, char_measure, bbox_w)
        assert char_measure(text) <= bbox_w * 0.97 + 1e-6


def test_text_kind_uses_vocab_words():
    random.seed(3)
    vocab = FakeVocab()
    seen = " ".join(
        generate_field_content("t", vocab, char_measure, 900) for _ in range(50)
    )
    # real vocabulary content, not pseudo-Polish syllables
    known_fragments = ["nadciśnienie", "Amoksycylina", "Kowalska", "Oddział",
                       "Radom", "zakażenie", "Rp.", "Nowak", "Pilne", "Szpital"]
    assert any(frag in seen for frag in known_fragments)


def test_number_kind_is_digit_based():
    random.seed(11)
    vocab = FakeVocab()
    for _ in range(50):
        text = generate_field_content("n", vocab, char_measure, 600)
        if text:
            digits = sum(ch.isdigit() for ch in text)
            assert digits >= len(text) * 0.5, text


def test_mix_kind_has_letters_and_digits():
    random.seed(5)
    vocab = FakeVocab()
    combined = " ".join(
        generate_field_content("mix", vocab, char_measure, 700) for _ in range(50)
    )
    assert any(ch.isdigit() for ch in combined)
    assert any(ch.isalpha() for ch in combined)
