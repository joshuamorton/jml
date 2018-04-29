from unittest import mock
import numpy as np
import pytest
import random

from experiments.utils import circles


@pytest.mark.parametrize(
    "w,h,r",
    [
        (10, 10, 4),
        (10, 10, 3),
        (10, 10, 2),
        (10, 10, 1),
        (5, 5, 2),
        (5, 5, 1),
        (3, 5, 2),
    ],
)
def test_boxed_cord(w, h, r):
    """Hypothesis-y testing that the coordinates are reasonable."""
    x, y = circles.boxed_coord(w, h, r)
    assert x >= r
    assert y >= r
    assert x <= w - r + 1
    assert y <= h - r + 1


@mock.patch.object(circles, "boxed_coord")
def test_random_boxed_circle(mock_coord):
    mock_coord.return_value = (5, 5)
    circ, (x, y) = circles.random_boxed_circle(10, 10, 2)
    assert (x, y) == (0.5, 0.5)
    assert np.sum(circ) == 13  # a 5x5 circle has 13 on pixels


@pytest.mark.parametrize(
    "x,y,r,s,cnt",
    [
        (5, 5, 2, (10, 10), 13),
        (3, 3, 2, (10, 10), 13),
        (5, 5, 1, (10, 10), 5),
        (5, 5, 0, (10, 10), 1),
        (0, 5, 2, (10, 10), 9),
        (5, 0, 2, (10, 10), 9),
        (10, 10, 2, (10, 10), 1),
    ],
)
def test_circle(x, y, r, s, cnt):
    c = circles.circle(x, y, r, s)
    assert np.sum(c) == cnt
