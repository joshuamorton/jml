

import random

import numpy as np

def random_point(width, height):
    img = np.zeros((width, height))
    point = random.randint(0, width - 1), random.randint(0, height - 1)
    img[point[0],point[1]] = 1
    return img, point
