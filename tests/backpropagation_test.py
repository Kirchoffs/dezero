import numpy as np

from dezero import add, exp, square
from dezero import numerical_derivative
from dezero import Variable
from dezero import Exp, Square


def test_backpropagation_1():
    print("\nTest Backpropagation 1")

    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array([1.0]))
    a = A(x)
    b = B(a)
    c = C(b)

    c.grad = np.array(1.0)
    c.backward()
    print(x.grad)


def test_backpropagation_2():
    print("\nTest Backpropagation 2")

    x = Variable(np.array([1.0]))
    a = square(x)
    b = exp(a)
    c = square(b)

    c.grad = np.array(1.0)
    c.backward()
    print(x.grad)


def test_backpropagation_3():
    print("\nTest Backpropagation 3")

    x = Variable(np.array([1.0]))
    y = square(exp(square(x)))

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


def test_backpropagation_4():
    print("\nTest Backpropagation 4")

    x = Variable(np.array([1.0]))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)


def test_backpropagation_5():
    print("\nTest Backpropagation 5")

    x = Variable(np.array([1.0]))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)

    composed_function = lambda x: square(exp(square(x)))
    numerical_grad = numerical_derivative(composed_function, x)
    assert np.allclose(x.grad, numerical_grad), "Backpropagation gradient does not match numerical gradient"


def test_backpropagation_6():
    print("\nTest Backpropagation 6")

    x = Variable(np.array([2.0]))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)
