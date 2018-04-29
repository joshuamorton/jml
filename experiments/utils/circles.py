"""Utility functions for creating circles.
"""

import random

import cv2
import numpy as np
from typing import Tuple


def boxed_coord(width: int, height: int, buffer: int) -> Tuple[int, int]:
    """Selects a pair of coordinates a given distance away from image edges.
    """
    x = random.randint(buffer, height - buffer - 1)
    y = random.randint(buffer, width - buffer - 1)
    return x, y


def random_boxed_circle(
    width: int, height: int, radius: int
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Creates an image containing a circle at a random location.

    Args:
        width: Width of output image.
        height: Height of output image.
        radius: Radius of the resulting circle.
    Return:
        A tuple containing a greyscale image of a filled circle and the
        coordinates of the center of that circle.
    """
    # 3 channel image
    empty = np.zeros((width, height, 3))

    # thickness = -1 forces a solid circle
    x, y = boxed_coord(width, height, radius)
    img = cv2.circle(empty, (x, y), radius, color=(1, 1, 1), thickness=-1)
    # convert image to greyscale
    img = img[:, :, 1]
    return img, (x / width, y / height)


def circle(
    x: int, y: int, radius: int, shape: Tuple[int, int] = (10, 10)
) -> np.ndarray:
    width, height = shape
    empty = np.zeros((width, height, 3))
    img = cv2.circle(empty, (x, y), radius, color=(1, 1, 1), thickness=-1)
    img = img[:, :, 1]
    return img.reshape([width, height, 1])
