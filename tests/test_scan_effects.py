import numpy as np
from PIL import Image

from transforms import to_grayscale, photocopy_contrast, salt_pepper_noise, toner_streak


def _gradient_img(w=200, h=100):
    arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return Image.merge("RGB", [Image.fromarray(arr)] * 3)


def test_to_grayscale_keeps_size_and_mode():
    img = Image.new("RGB", (120, 80), (28, 42, 120))  # blue ink color
    out = to_grayscale(img)
    assert out.size == (120, 80)
    assert out.mode == "RGB"
    r, g, b = out.getpixel((10, 10))
    assert r == g == b  # actually gray


def test_photocopy_contrast_crushes_extremes():
    out = photocopy_contrast(_gradient_img(), low=100, high=190)
    arr = np.array(out.convert("L"))
    assert arr[:, :70].max() == 0      # below low -> pure black
    assert arr[:, 160:].min() == 255   # above high -> pure white
    mid = arr[:, 100:140]
    assert 0 < mid.mean() < 255        # midtones ramp, not binarized


def test_salt_pepper_changes_some_pixels():
    img = Image.new("RGB", (100, 100), (128, 128, 128))
    out = salt_pepper_noise(img, amount=0.01)
    arr = np.array(out)
    n_black = (arr == 0).all(axis=2).sum()
    n_white = (arr == 255).all(axis=2).sum()
    assert n_black + n_white > 0
    assert n_black + n_white < 100 * 100 * 0.05  # sparse, not destroyed


def test_toner_streak_touches_narrow_band_only():
    img = Image.new("RGB", (300, 100), (200, 200, 200))
    out = toner_streak(img)
    diff_cols = (np.array(out) != np.array(img)).any(axis=(0, 2))
    assert 0 < diff_cols.sum() <= 300 * 0.05  # a narrow vertical band changed
