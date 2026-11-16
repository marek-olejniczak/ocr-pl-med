from pp.decide import DecisionParams, Scored, best_two, decide
from pp.protect import protected_class, train_abbreviations
from pp.tokenize import case_pattern, restore_case, split_token, token_pairs


def sc():
    # klucz "kosc": kandydaci kość (tani kanał, częsty), kosz (droższy), koc (lev 1, rzadki)
    return Scored("kosc", ["kość", "kosz", "koc"], [1, 1, 1], [-1.0, -6.0, -7.0], [-9.0, -10.0, -12.0],
                  [5000, 3000, 0], -9.0)


def test_best_two_and_filters():
    b, sb, sb2 = best_two(sc(), 1.0, 2, 0)
    assert b == 0 and sb == -10.0 and sb2 == -16.0
    assert best_two(sc(), 1.0, 0, 0)[0] == -1          # d_max=0 wyklucza wszystkich
    assert best_two(sc(), 1.0, 2, 4000)[0] == 0        # f_min zostawia tylko kość


def test_decide_thresholds():
    lp_unseen = -20.0
    # s_keep = -9 + (-20) = -29; s(b) = -10 -> zysk 19
    assert decide(sc(), DecisionParams(lam=1.0, delta=5, mu=1, d_max=2, L_min=4), lp_unseen) == "kość"
    assert decide(sc(), DecisionParams(lam=1.0, delta=25, mu=1, d_max=2, L_min=4), lp_unseen) is None
    assert decide(sc(), DecisionParams(lam=1.0, delta=5, mu=10, d_max=2, L_min=4), lp_unseen) is None
    assert decide(sc(), DecisionParams(lam=1.0, delta=5, mu=1, d_max=2, L_min=5), lp_unseen) is None


def test_tokenize_and_case():
    t = split_token("(Kosc,")
    assert (t.pre, t.core, t.post, t.key) == ("(", "Kosc", ",", "kosc")
    assert t.rebuild(restore_case(t.core, "kość")) == "(Kość,"
    assert restore_case("KOSC", "kość") == "KOŚĆ"
    assert case_pattern("kosc") == "lower" and case_pattern("KoSc") == "mixed"
    assert token_pairs(["a", "b", "c"], ["a", "c"]) == [(0, 0), (2, 1)]


def test_protect_classes():
    abbrev = {"zewn"}
    cls = lambda raw, ups=0.0: protected_class(split_token(raw), ups, abbrev)
    assert cls("->") == "P1_symbols"
    assert cls("L5-S1") == "P2_digits"
    assert cls("np.") == "P4_abbr_dot"
    assert cls("zewn") == "P5_abbr_list"
    assert cls("USG") == "P6_short_upper" and cls("USG", 0.9) is None
    assert cls("ul") == "P7_len_le_2"
    assert cls("krzyżowo-biodrowy") == "P8_hyphen"
    assert cls("kosc") is None
    assert protected_class(split_token("np."), 0.0, abbrev, disabled=["P4_abbr_dot"]) == "P7_len_le_2"


def test_train_abbreviations_only_oov_with_dot():
    got = train_abbreviations(["t. udowa przyśr. kość."], known=lambda w: w in {"kość", "udowa"})
    assert got == {"t", "przyśr"}


def test_deloop():
    from pp.normalize import deloop, hygiene
    assert deloop("A Maring Maring Maring Maring") == "A Maring"
    assert deloop("- - - - - -") == "-"
    assert deloop("Mungy<br>M Mungy<br>M Mungy") == "Mungy M Mungy"
    assert deloop("bardzo bardzo dobrze") == "bardzo bardzo dobrze"
    assert deloop("1. 1. 1. 1. x") == "1. x"
    assert deloop("ab cd ab cd ab cd ef") == "ab cd ef"
    assert hygiene("kość  udowa") == "kość udowa"
