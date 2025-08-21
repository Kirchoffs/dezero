import numpy as np
from dezero import Variable


def test_second_order_gradient():
    def f(x):
        return x ** 4 - 2 * x ** 2
    
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph = True)
    assert x.grad.data == 24.0

    gx = x.grad
    x.clean_grad()
    gx.backward()
    assert x.grad.data == 44.0


def test_back_propagation_after_back_propagation():
    x = Variable(np.array(2.0))
    y = x ** 2
    y.backward(create_graph = True)
    gx = x.grad
    x.clean_grad()

    z = gx ** 3 + y
    z.backward()
    assert x.grad.data == 100.0
