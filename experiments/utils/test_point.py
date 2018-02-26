import numpy as np
import pytest

from experiments.utils import point

@pytest.mark.parametrize('w,h,x,y', [
    (10, 10, 5, 5),
    (3, 3, 1, 1),
    (1, 1, 0, 0),
])
def test_point(w, h, x, y):
    img, p = point.point(w, h, x, y)
    assert p == (x, y)
    assert img[p] == 1
    assert np.sum(img) == 1

def test_random_point_a_bunch():
    for _ in range(100):
        img, p = point.random_point(10, 10)
        assert img[p] == 1
        assert np.sum(img) == 1
