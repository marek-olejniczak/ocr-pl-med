from collections import Counter

import pytest

from evaluation.compare import align_to_gt, token_table
from evaluation.folds import assign_group, line_groups, logo, nested_logo
from evaluation.metrics import load_predictions


def test_every_line_in_exactly_one_group():
    rows = load_predictions()
    c = Counter(line_groups(rows))
    assert sum(c.values()) == 2160 and len(c) == 8
    assert c["G1_noga"] == 404 and c["G8_zrzuty_0512"] == 394
    assert len(set(line_groups(rows, "coarse"))) == 6


def test_unmatched_line_raises():
    with pytest.raises(ValueError):
        assign_group("nieznane_line1.jpg")


def test_logo_disjoint_and_nested():
    groups = ["a", "a", "b", "c", "c", "c"]
    folds = list(logo(groups))
    assert [g for g, _, _ in folds] == ["a", "b", "c"]
    for g, tr, te in folds:
        assert not set(tr) & set(te) and sorted(tr + te) == list(range(6))
        assert {groups[i] for i in te} == {g}
    for g, tr, te, inner in nested_logo(groups):
        assert len(inner) == 2
        for h, itr, ite in inner:
            assert set(itr) | set(ite) == set(tr) and g not in {groups[i] for i in itr + ite}


def test_align_to_gt():
    assert align_to_gt(["a", "b", "c"], ["a", "x", "c"]) == ["a", "x", "c"]
    assert align_to_gt(["a", "b", "c"], ["a", "c"]) == ["a", None, "c"]


def test_token_table_categories():
    gts = ["kość udowa boczna"]
    base = ["kosc udowa boczna"]
    corr = ["kość udowo bocz"]
    t = token_table(gts, base, corr)
    assert (t["fixed"], t["broken"], t["changed_wrong"], t["unchanged_correct"]) == (1, 2, 0, 0)
    assert t["net"] == -1 and t["harm"] == 1.0
    t = token_table(["ala ma kota"], ["ala mo kot"], ["ala ma kat"])
    assert (t["fixed"], t["changed_wrong"], t["unchanged_correct"]) == (1, 1, 1)
    assert t["precision"] == 0.5 and t["recall"] == 0.5 and t["harm"] == 0.0
