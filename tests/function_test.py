import numpy as np

import dezero.functions as F
from dezero import Variable
from dezero import Add, Square


def test_add():
    print("\nTest Add")

    x = Variable(np.array([1.5]))
    y = Variable(np.array([2.5]))
    z1 = Add()(x, y)
    z2 = x + y

    assert np.allclose(z1.data, np.array([4.0])), "Add function failed"
    assert np.allclose(z2.data, np.array([4.0])), "Add function failed"


def test_square():
    print("\nTest Square")

    x = Variable(np.array([2.0]))
    y = Square()(x)

    assert np.allclose(y.data, np.array([4.0])), "Square function failed"


def test_mul():
    print("\nTest Mul")

    x = Variable(np.array([3.0]))
    y = Variable(np.array([2.0]))
    z = Variable(np.array([1.0]))

    res = x * y + z
    res.backward()

    assert np.allclose(res.data, np.array([7.0])), "Mul function failed"
    assert np.allclose(x.grad, np.array([2.0])), "x.grad is incorrect"
    assert np.allclose(y.grad, np.array([3.0])), "y.grad is incorrect"
    assert np.allclose(z.grad, np.array([1.0])), "z.grad is incorrect"


def test_radd():
    print("\nTest RAdd")

    x = 2
    y = Variable(np.array([3]))

    z = x + y

    assert np.allclose(z.data, np.array([5])), "RAdd function failed"


def test_rmul():
    print("\nTest RMul")

    x = 2
    y = Variable(np.array([3]))

    z = x * y

    assert np.allclose(z.data, np.array([6])), "RMul function failed"


def test_radd_with_ndarray():
    print("\nTest RAdd with NDArray")

    x = np.array([2])
    y = Variable(np.array([3]))

    z = x + y

    assert np.allclose(z.data, np.array([5])), "RAdd function failed"


def test_rmul_with_ndarray():
    print("\nTest RMul with NDArray")

    x = np.array([2])
    y = Variable(np.array([3]))

    z = x * y

    assert np.allclose(z.data, np.array([6])), "RMul function failed"


def test_neg():
    print("\nTest Neg")

    x = Variable(np.array([2.0]))
    y = -x

    assert np.allclose(y.data, np.array([-2.0])), "Neg function failed"


def test_sub_rsub():
    print("\nTest Sub and RSub")

    x = Variable(np.array([2.0]))
    y = Variable(np.array([3.0]))

    z = x - y
    assert np.allclose(z.data, np.array([-1.0])), "Sub function failed"

    z = x - 3.0
    assert np.allclose(z.data, np.array([-1.0])), "Sub function failed"

    z = 3.0 - x
    assert np.allclose(z.data, np.array([1.0])), "RSub function failed"


def test_div_rdiv():
    print("\nTest Div and RDiv")

    x = Variable(np.array([2.0]))
    y = Variable(np.array([4.0]))

    z = x / y
    assert np.allclose(z.data, np.array([0.5])), "Div function failed"

    z = x / 4.0
    assert np.allclose(z.data, np.array([0.5])), "Div function failed"

    z = 4.0 / x
    assert np.allclose(z.data, np.array([2.0])), "RDiv function failed"


def test_pow():
    print("\nTest Pow")

    x = Variable(np.array([2.0]))
    y = x ** 3

    assert np.allclose(y.data, np.array([8.0])), "Pow function failed"


def test_sin():
    print("\nTest Sin")

    x = Variable(np.array([np.pi / 2]))
    y = F.sin(x)

    assert np.allclose(y.data, np.array([1.0])), "Sin function failed"


def test_sin_maclaurin():
    print("\nTest Sin Maclaurin")

    x = Variable(np.array([np.pi / 4]))
    y_maclaurin = F.sin_maclaurin(x, threshold = 1e-6)
    y_maclaurin.backward()

    assert np.allclose(y_maclaurin.data, np.array([np.sin(np.pi / 4)]), atol = 1e-5), "Sin Maclaurin function failed"
    assert np.allclose(x.grad, np.array([np.cos(np.pi / 4)]), atol = 1e-5), "Sin Maclaurin backward failed"


def test_reshape():
    print("\nTest Reshape")

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.reshape(x, (6,))
    y.backward(retain_grad = True)
    
    assert np.allclose(x.grad, np.array([[1, 1, 1], [1, 1, 1]])), "Reshape backward failed"


def test_transpose():
    print("\nTest Transpose")

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.transpose(x)
    y.backward(retain_grad = True)

    assert np.allclose(x.grad, np.array([[1, 1, 1], [1, 1, 1]])), "Transpose backward failed"


def test_sum():
    print("\nTest Sum")

    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()

    assert np.allclose(y.data, np.array(21)), "Sum function 1 failed"
    assert np.allclose(x.grad, np.array([1, 1, 1, 1, 1, 1])), "Sum backward 1 failed"

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()

    assert np.allclose(y.data, np.array(21)), "Sum function 2 failed"
    assert np.allclose(x.grad, np.array([[1, 1, 1], [1, 1, 1]])), "Sum backward 2 failed"


def test_broadcast():
    print("\nTest Broadcast")

    x = Variable(np.array([1, 2, 3]))
    y = Variable(np.array([6]))

    z = x + y
    assert np.allclose(z.data, np.array([7, 8, 9])), "Broadcast add failed"

    z.backward()
    assert np.allclose(x.grad, np.array([1, 1, 1])), "Broadcast backward x failed"
    assert np.allclose(y.grad, np.array([3])), "Broadcast backward y failed"


def test_matmul():
    print("\nTest MatMul")

    x = Variable(np.array([[1, 2], [3, 4]]))
    W = Variable(np.array([[5, 6], [7, 8]]))

    y = F.matmul(x, W)
    y.backward()

    assert np.allclose(y.data, np.array([[19, 22], [43, 50]])), "MatMul function failed"
    assert np.allclose(x.grad, np.array([[11, 15], [11, 15]])), "MatMul backward x failed"
    assert np.allclose(W.grad, np.array([[4, 4], [6, 6]])), "MatMul backward W failed"
