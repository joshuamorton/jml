import random
from typing import Tuple

import numpy as np


def random_point(width: int, height: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = np.zeros((width, height))
    p = random.randint(0, width - 1), random.randint(0, height - 1)
    img[p[0], p[1]] = 1
    return img, p


def point(
    width: int, height: int, x: int, y: int
) -> Tuple[np.ndarray, Tuple[int, int]]:
    img = np.zeros((width, height))
    img[x, y] = 1
    return img, (x, y)
