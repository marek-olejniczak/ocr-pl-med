import pytest

from evaluation.run_holdout import split_lines, sub_logo
from pp.lexicon import domain_terms


def test_split_lines_by_group():
    assert split_lines(["G1_noga", "G3_serce", "G4_unaczynienie"]) == ([1], [0, 2])


def test_sub_logo_stays_inside_lines():
    groups = ["A", "B", "T", "A", "B", "T"]
    folds = list(sub_logo(groups, [0, 1, 3, 4]))
    assert [(g, tr, te) for g, tr, te in folds] == [("A", [1, 4], [0, 3]), ("B", [0, 3], [1, 4])]


def test_domain_terms_filters_by_language():
    assert {"muscle", "femur"} <= domain_terms(("en",))
    latin = domain_terms(("la", "pl"))
    assert "femur" in latin and "muscle" not in latin
    assert domain_terms(()) == set()


def test_domain_terms_rejects_unknown_language():
    with pytest.raises(ValueError):
        domain_terms(("de",))
