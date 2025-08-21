import numpy as np
import math
from .core import Variable, Function
from .core import as_array, as_variable
from .utils import reshape_sum_backward_for_broadcast, sum_to_shape


class Add(Function):
    def forward(self, x_left, x_right):
        self.x_left_shape = x_left.shape
        self.x_right_shape = x_right.shape
        return x_left + x_right

    def backward(self, gy):
        gx_left, gx_right = gy, gy
        if self.x_left_shape != self.x_right_shape:
            gx_left = sum_to(gy, self.x_left_shape)
            gx_right = sum_to(gy, self.x_right_shape)
        return gx_left, gx_right


class Sub(Function):
    def forward(self, x_left, x_right):
        return x_left - x_right

    def backward(self, gy):
        return gy, -gy
    

class Mul(Function):
    def forward(self, x_left, x_right):
        return x_left * x_right
    
    def backward(self, gy):
        x_left, x_right = self.inputs
        gx_left = gy * x_right
        gx_right = gy * x_left
        return gx_left, gx_right


class Div(Function):
    def forward(self, x_left, x_right):
        return x_left / x_right
    
    def backward(self, gy):
        x_left, x_right = self.inputs
        gx_left = gy / x_right
        gx_right = -gy * x_left / (x_right ** 2)
        return gx_left, gx_right


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x, = self.inputs
        gx = np.exp(x) * gy
        return gx


class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        return x ** self.c

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * (x ** (c - 1)) * gy
        return gx


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        gx = 2 * x * gy
        return gx
    

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x, = self.inputs
        gx = -gy * sin(x)
        return gx
    

class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y, = self.outputs
        gx = gy * (1 - y() ** 2)
        return gx


class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.original_shape = x.shape
        return x.reshape(self.shape)

    def backward(self, gy):
        return reshape(gy, self.original_shape)


class Transpose(Function):
    def forward(self, x):
        return np.transpose(x)

    def backward(self, gy):
        return transpose(gy)


class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis = self.axis, keepdims = self.keepdims)
    
    def backward(self, gy):
        gy = reshape_sum_backward_for_broadcast(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)
    

class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.broadcast_to(x, self.shape)

    def backward(self, gy):
        return sum_to(gy, self.x_shape)
    

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return sum_to_shape(x, self.shape)

    def backward(self, gy):
        return broadcast_to(gy, self.x_shape)
    

class MatMul(Function):
    def forward(self, x, W):
        return x.dot(W)

    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    

class MeanSquareError(Function):
    def forward(self, y_pred, y_true):
        diff = y_pred - y_true
        return np.sum(diff ** 2) / len(diff)

    def backward(self, gy):
        y_pred, y_true = self.inputs
        diff = y_pred - y_true
        gx = (2 / len(diff)) * diff * gy
        return gx, -gx
    

class Linear(Function):
    def forward(self, x, W, b = None):
        y = x.dot(W)
        if b is not None:
            y += b
        return y

    def backward(self, gy):
        x, W, b = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        gb = None if b.data is None else sum_to(gy, b.shape)
        return gx, gW, gb
    

class Sigmoid(Function):
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, gy):
        y, = self.outputs
        gx = gy * y() * (1 - y())
        return gx


class ReLU(Function):
    def forward(self, x):
        return np.maximum(0, x)

    def backward(self, gy):
        x, = self.inputs
        mask = x.data > 0
        gx = gy * mask
        return gx


def add(x, y):
    y = as_array(y)
    return Add()(x, y)


def sub(x, y):
    y = as_array(y)
    return Sub()(x, y)


def rsub(x, y):
    y = as_array(y)
    return Sub()(y, x)


def mul(x, y):
    y = as_array(y)
    return Mul()(x, y)


def div(x, y):
    y = as_array(y)
    return Div()(x, y)


def rdiv(x, y):
    y = as_array(y)
    return Div()(y, x)


def exp(x):
    return Exp()(x)


def pow(x, c):
    return Pow(c)(x)


def square(x):
    return Square()(x)


def neg(x):
    return Neg()(x)


def sin(x):
    return Sin()(x)


def cos(x):
    return Cos()(x)


def tanh(x):
    return Tanh()(x)


def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


def transpose(x):
    return Transpose()(x)


def sum(x, axis = None, keepdims = False):
    return Sum(axis, keepdims)(x)


def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)


def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)


def matmul(x, W):
    return MatMul()(x, W)


def mean_square_error(y_pred, y_true):
    return MeanSquareError()(y_pred, y_true)


def linear(x, W, b):
    return Linear()(x, W, b)


def linear_simple(x, W, b = None):
    t = matmul(x, W)
    if b is None:
        return t
    
    y = t + b
    t.data = None
    return y


def sigmoid(x):
    return Sigmoid()(x)


def sigmoid_simple(x):
    return 1 / (1 + exp(-as_variable(x)))


def relu(x):
    return ReLU()(x)


def sin_maclaurin(x, threshold = 1e-5):
    y = 0
    for i in range(999):
        coeff = (-1) ** i / math.factorial(2 * i + 1)
        term = coeff * x ** (2 * i + 1)
        y = y + term
        if np.all(np.abs(term.data) < threshold):
            break
    return y


def numerical_derivative(f, x, eps = 1e-6):
    x_minus = Variable(x.data - eps)
    x_plus = Variable(x.data + eps)
    return (f(x_plus).data - f(x_minus).data) / (2 * eps)


def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__neg__ = neg
