import numpy as np
import dezero.functions as F
from dezero import Variable


def test_sin():
    x = Variable(np.array([np.pi, np.pi / 2, np.pi / 6]))
    y = F.sin(x)
    assert np.allclose(y.data, [0.0, 1.0, 0.5]), "Sin function failed"


def test_addition():
    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = Variable(np.array([[11, 22, 33], [44, 55, 66]]))
    z = x + y
    expected = np.array([[12, 24, 36], [48, 60, 72]])
    assert np.allclose(z.data, expected), "Addition operation failed"
