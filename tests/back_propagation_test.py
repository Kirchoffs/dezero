if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np

from dezero import numerical_derivative
from dezero import Variable
from dezero import Exp, Square
from dezero import add, exp, square


def test_back_propagation_1():
    print("\nTest Back Propagation 1")

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


def test_back_propagation_2():
    print("\nTest Back Propagation 2")

    x = Variable(np.array([1.0]))
    a = square(x)
    b = exp(a)
    c = square(b)

    c.grad = np.array(1.0)
    c.backward()
    print(x.grad)


def test_back_propagation_3():
    print("\nTest Back Propagation 3")

    x = Variable(np.array([1.0]))
    y = square(exp(square(x)))

    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


def test_back_propagation_4():
    print("\nTest Back Propagation 4")

    x = Variable(np.array([1.0]))
    y = square(exp(square(x)))

    y.backward()
    print(x.grad)


def test_back_propagation_5():
    print("\nTest Back Propagation 5")

    x = Variable(np.array([1.0]))
    #y = square(exp(square(x)))

    #y.backward()
    #print(x.grad)

    composed_function = lambda x: square(x)
    numerical_grad = numerical_derivative(composed_function, x)
    # assert np.allclose(x.grad, numerical_grad), "Backpropagation gradient does not match numerical gradient"


def test_back_propagation_6():
    print("\nTest Back Propagation 6")

    x = Variable(np.array([2.0]))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data)
    print(x.grad)
