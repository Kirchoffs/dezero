from dezero import Variable
import numpy as np


def test_variable_initialization_non_ndarray():
    try:
        Variable(2.718)
    except TypeError as e:
        print(e)


def test_get_item():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    indices = np.array([0, 0, 1])
    assert np.array_equal(x[indices].data, np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]]))
