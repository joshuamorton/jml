"""Utility functions for creating circles.
"""

import random

import cv2
import numpy as np


def boxed_coord(width, height, radius):
    """Selects a set of coordinates a given distance away from image edges.
    """
    x = random.randint(radius, height - radius - 1)
    y = random.randint(radius, width - radius - 1)
    return x, y

def random_boxed_circle(width, height, radius):
    """Creates an image containing a circle at a random location.

    Args:
        width: width of output image
        height: height of output image
        radius: radius of the resulting circle
    Return:
        Tuple[image, Tuple[x, y]]
        where image is the image containing the circle and x,y are the
        coordinates of the center of the circle
    """
    # 3 channel image
    empty = np.zeros((width, height, 3))

    # thickness = -1 forces a solid circle
    x, y = boxed_coord(width, height, radius)
    img = cv2.circle(empty, (x, y), radius, color=(1,1,1), thickness=-1)
    # convert image to greyscale
    img = img[:,:,1]
    return img, (x, y)
