from typing import Iterable
import numpy as np


def flatten_indicies(vals: Iterable[Iterable[int]], shape: Iterable[int] = (10, 10)):
    """
    Flattens a set of coordinates to a coordinate in a 1-hot vector.

    Given a series of indicies in an array, and the shape of the array, returns
    a single value representing the index of the coordinate in the one hot
    vector made by flattening the provided shape. `vals` is accepted as a
    vector for easier use with numpy primitives. For example:

    `flatten_indicies([(0,0), (1,1)], shape=(10,10))` returns `[0, 11]`
    """
    cumprod = np.cumprod([1] + list(shape))[:-1][::-1]
    return [sum(val * cumprod) for val in vals]
