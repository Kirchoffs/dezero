import numpy as np

from dezero import add, square
from dezero import numerical_derivative
from dezero import Variable


def test_square_derivative():
    x = Variable(np.random.rand(1))
    y = square(x)
    y.backward()

    numerical_grad = numerical_derivative(square, x)
    print("Numerical gradient:", numerical_grad)
    print("Backpropagation gradient:", x.grad)
    assert np.allclose(x.grad, numerical_grad), "Square derivative test failed"


def test_add_derivative_same_variable():
    x = Variable(np.random.rand(1))
    y = add(x, x)
    y.backward(retain_grad = True)

    assert y.grad.data == 1, "Add derivative same variable test failed for y variable"
    assert x.grad.data == 2, "Add derivative same variable test failed for x variable"

    x.clean_grad()
    y = add(add(x, x), x)
    y.backward(retain_grad = True)

    assert y.grad.data == 1, "Add derivative same variable test failed for y variable"
    assert x.grad.data == 3, "Add derivative same variable test failed for x variable"


def test_retain_grad_false():
    a = Variable(np.array([1.0]))
    b = Variable(np.array([1.0]))
    c = add(a, b)
    d = add(c, a)
    d.backward()

    assert d.grad is None, "Retain grad false test failed for d variable"
    assert c.grad is None, "Retain grad false test failed for c variable"

    assert a.grad.data == 2, "Retain grad false test failed for a variable"
    assert b.grad.data == 1, "Retain grad false test failed for b variable"


def test_newton_optimization():
    def f(x):
        return x ** 4 - 2 * x ** 2

    x = Variable(np.array(2.0))
    iters = 10
    for i in range(iters):
        print(i, x)

        y = f(x)
        x.clean_grad()

        y.backward(create_graph = True)
        gx = x.grad
        x.clean_grad()

        gx.backward()
        g2x = x.grad

        x.data -= gx.data / g2x.data
