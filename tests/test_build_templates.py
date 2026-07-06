from build_templates import map_old_label


def test_generic_labels():
    assert map_old_label("printed") == "p"
    assert map_old_label("text") == "t"
    assert map_old_label("number") == "n"


def test_legacy_field_labels():
    # numeric-ish legacy labels -> n
    for lbl in ["pesel", "pesel_grid", "full_date", "phone_num", "icd_10",
                "age", "day_and_month", "last_2_digits_year"]:
        assert map_old_label(lbl) == "n", lbl
    # text-ish legacy labels -> t
    for lbl in ["city", "name_and_surname", "address", "diagnosis",
                "approval", "hospital", "full_signature"]:
        assert map_old_label(lbl) == "t", lbl


def test_junk_labels_rejected():
    assert map_old_label("JÁDRA") is None
    assert map_old_label("label") is None
    assert map_old_label("") is None


def test_case_insensitive():
    assert map_old_label("  Printed ") == "p"
