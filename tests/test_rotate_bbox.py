"""Pin the contract of transforms.rotate_bbox.

Since page rotation was dropped, artifacts.draw_stamp is the only caller of
this function, and its argument order silently decides whether stamp ground
truth lands on the text or a line away from it. The expected values below are
derived by hand from the quarter-turn mapping, not copied from the function's
own output, so a swapped-argument regression cannot quietly redefine "correct".
"""

import math

import numpy as np
from PIL import Image

from transforms import rotate_bbox


def test_identity_when_angle_is_zero():
    """No rotation, same canvas: the box must come back untouched."""
    assert rotate_bbox((10, 5, 30, 15), 0.0, (100, 40), (100, 40)) == (10, 5, 30, 15)


def test_quarter_turn_matches_hand_derived_mapping():
    """A 90 deg turn of a (100, 40) canvas maps (x, y) -> (y, 100 - x).

    PIL rotates counterclockwise, so the source top-left corner ends up at the
    destination bottom-left. Working that mapping over the four corners of
    (10.5, 5.5, 30.5, 15.5):

        (10.5,  5.5) -> ( 5.5, 89.5)
        (30.5,  5.5) -> ( 5.5, 69.5)
        (30.5, 15.5) -> (15.5, 69.5)
        (10.5, 15.5) -> (15.5, 89.5)

    Enclosing box = (5.5, 69.5, 15.5, 89.5), floored/ceiled to (5, 69, 16, 90).
    Half-integer inputs are used on purpose: cos(-90 deg) evaluates to 6.1e-17
    rather than 0, so an exactly-integer result can land a hair below the
    integer and floor down by one.
    """
    assert rotate_bbox((10.5, 5.5, 30.5, 15.5), 90.0, (100, 40), (40, 100)) == (5, 69, 16, 90)


def test_quarter_turn_integer_bbox_within_rounding():
    """Same quarter turn on integer corners: (10,5,30,15) -> (5,70,15,90)."""
    result = rotate_bbox((10, 5, 30, 15), 90.0, (100, 40), (40, 100))
    for got, want in zip(result, (5, 70, 15, 90)):
        assert abs(got - want) <= 1, f"{result} too far from hand-derived (5,70,15,90)"


def test_src_and_dst_sizes_are_not_interchangeable():
    """Swapping src_size and dst_size must change the answer.

    This is the regression the stamp code is exposed to: with the sizes
    swapped the box shifts by the expansion the rotation added, which at a
    stamp's line pitch is roughly a whole line of text.
    """
    correct = rotate_bbox((10, 5, 30, 15), 90.0, (100, 40), (40, 100))
    swapped = rotate_bbox((10, 5, 30, 15), 90.0, (40, 100), (100, 40))
    assert correct != swapped
    assert abs(correct[1] - swapped[1]) > 20


def test_agrees_with_pil_pixel_rotation():
    """The box must track where PIL actually moves the pixels.

    A filled rectangle is rotated with the same call draw_stamp uses; the ink's
    real bounding box in the result must match what rotate_bbox predicts.
    """
    source = Image.new("L", (200, 80), 0)
    source.paste(255, (40, 20, 150, 60))

    for angle in (20.0, -13.0, 7.5):
        rotated = source.rotate(angle, resample=Image.NEAREST, expand=True)
        ys, xs = np.nonzero(np.array(rotated) > 128)
        actual = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
        predicted = rotate_bbox((40, 20, 150, 60), angle, source.size, rotated.size)
        for got, want in zip(predicted, actual):
            assert abs(int(got) - int(want)) <= 2, (
                f"angle {angle}: rotate_bbox {predicted} vs real ink {actual}"
            )
