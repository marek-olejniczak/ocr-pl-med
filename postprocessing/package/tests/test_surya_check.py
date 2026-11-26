from ocr_postprocess.surya_check import compare_scores


def test_compare_scores():
    assert compare_scores([-1.0, -9.0, -3.0], [-1.1, -9.2, -3.0]) == []
    assert len(compare_scores([-1.0, -9.0, -3.0], [-1.0, -4.0, -3.0])) == 1          # partia != pojedynczo
    assert len(compare_scores([-9.0, -1.0, -3.0], [-9.0, -1.0, -3.0])) == 1          # zła kolejność
    assert compare_scores([float("nan"), -9.0, -3.0], [-1.0, -9.0, -3.0])            # NaN to też błąd
