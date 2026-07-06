import random
from collections import Counter

import numpy as np
from PIL import Image

from fill_form import SCAN_PROFILES, pick_scan_profile, apply_scan_augmentation


def test_profile_names_and_weights():
    names = [n for n, _ in SCAN_PROFILES]
    assert names == ["clean_color", "grayscale", "photocopy"]
    assert abs(sum(w for _, w in SCAN_PROFILES) - 1.0) < 1e-9


def test_pick_distribution_roughly_matches_weights():
    random.seed(42)
    counts = Counter(pick_scan_profile() for _ in range(4000))
    assert 0.38 < counts["clean_color"] / 4000 < 0.52
    assert 0.28 < counts["grayscale"] / 4000 < 0.42
    assert 0.14 < counts["photocopy"] / 4000 < 0.26


def _form():
    img = Image.new("RGB", (400, 300), (250, 248, 245))
    img.paste((28, 42, 120), (50, 100, 250, 130))  # a blue "text" block
    return img


def test_apply_returns_image_and_meta_same_size():
    random.seed(1)
    out, meta = apply_scan_augmentation(_form())
    assert out.size == (400, 300)          # geometry untouched
    assert meta["profile"] in ("clean_color", "grayscale", "photocopy")
    assert "rotation_angle_deg" not in meta


def test_grayscale_and_photocopy_kill_color():
    random.seed(0)
    for _ in range(30):
        out, meta = apply_scan_augmentation(_form())
        arr = np.array(out).astype(int)
        max_chroma = np.abs(arr[:, :, 0] - arr[:, :, 2]).max()
        if meta["profile"] in ("grayscale", "photocopy"):
            assert max_chroma <= 25, meta["profile"]  # JPEG may leak a bit of chroma
        else:
            assert max_chroma > 25  # blue ink survives in clean_color
