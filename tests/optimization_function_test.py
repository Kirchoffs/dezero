import math
import numpy as np
from dezero import Variable


def test_sphere():
    def sphere(x, y):
        return x ** 2 + y ** 2


    x = Variable(np.array(1.0))
    y = Variable(np.array(3.0))
    z = sphere(x, y)
    z.backward()

    assert x.grad == 2.0
    assert y.grad == 6.0


def test_matyas():
    def matyas(x, y):
        return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = matyas(x, y)
    z.backward()

    assert math.isclose(x.grad, 0.04, rel_tol = 1e-9)
    assert math.isclose(y.grad, 0.04, rel_tol = 1e-9)


def test_goldstein_price():
    def goldstein_price(x, y):
        return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
            (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = goldstein_price(x, y)
    z.backward()

    assert math.isclose(x.grad, -5376.0, rel_tol = 1e-9)
    assert math.isclose(y.grad, 8064.0, rel_tol = 1e-9)
