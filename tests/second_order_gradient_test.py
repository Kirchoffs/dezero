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
