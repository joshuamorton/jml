import random

import numpy as np

def random_point(width, height):
    img = np.zeros((width, height))
    point = random.randint(0, width - 1), random.randint(0, height - 1)
    img[point[0],point[1]] = 1
    return img, point

def point(width, height, x, y):
    img = np.zeros((width, height))
    img[x, y] = 1
    return img, (x, y)
