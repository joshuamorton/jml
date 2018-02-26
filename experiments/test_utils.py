from experiments import utils

def test_flatten_indicies_default():
    assert [0, 11] == utils.flatten_indicies([(0,0), (1,1)])

def test_flatten_indicies():
    assert [0, 6] == utils.flatten_indicies([(0, 0), (1, 1)], shape=(5,5))

def test_flatten_indicies_multi_dim():
    assert [0, 6, 26, 31] == utils.flatten_indicies([
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 1),
    ], shape=(5,5,5))
