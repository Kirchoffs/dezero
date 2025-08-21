import numpy as np
import contextlib
import weakref
import math
import heapq


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    __array_priority__ = 256

    def __init__(self, data, name = None):
        if data is None or not isinstance(data, np.ndarray):
            raise TypeError("{} is not supported".format(type(data)))
        
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clean_grad(self):
        self.grad = None

    # def backward(self):
    #     f = self.creator
    #     if f is not None:
    #         x = f.input
    #         x.grad = f.backward(self.grad)
    #         x.backward()

    def backward(self, retain_grad = False, create_graph = False):
        if not Config.enable_backprop:
            raise RuntimeError("Backpropagation is disabled. Set 'Config.enable_backprop' to True to enable it.")

        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs_queue = []
        funcs_seen = set()

        def add_func(f):
            if f is not None and f not in funcs_seen:
                heapq.heappush(funcs_queue, f)
                funcs_seen.add(f)
            
        add_func(self.creator)
        while funcs_queue:
            f = heapq.heappop(funcs_queue)
            gys = tuple(output().grad for output in f.outputs)

            with using_config("enable_backprop", create_graph):
                gxs = f.backward(*gys)
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)

                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                    
                    add_func(x.creator)
                
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None

    def __array__(self, dtype = None):
        return self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        content = str(self.data).replace('\n', '\n' + ' ' * len("Variable("))
        return f"Variable(" + content + ")"


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


def as_variable(obj):
    if not isinstance(obj, Variable):
        obj = Variable(obj)
    return obj


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# The instance variables 'self.inputs' / 'self.outputs' are tuple of Variables.
# The parameters of forward and backward methods are just variables.
# The arguments of forward and backward methods can be tuples.
class Function:
    def __call__(self, *inputs):   
        inputs = tuple(as_variable(x) for x in inputs)

        xs = tuple(x.data for x in inputs)
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = tuple(Variable(as_array(y)) for y in ys)

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])

            for output in outputs:
                output.set_creator(self)

        self.inputs = inputs
        self.outputs = tuple(weakref.ref(output) for output in outputs)

        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented by subclasses")

    def backward(self, gy):
        raise NotImplementedError("Backward method must be implemented by subclasses")
    
    def __lt__(self, other):
        return self.generation > other.generation


class Add(Function):
    def forward(self, x_left, x_right):
        return x_left + x_right

    def backward(self, gy):
        return gy, gy


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


def sin_maclaurin(x, threshold = 1e-5):
    y = 0
    for i in range(999):
        coeff = (-1) ** i / math.factorial(2 * i + 1)
        term = coeff * x ** (2 * i + 1)
        y = y + term
        if np.all(np.abs(term.data) < threshold):
            break
    return y


def numerical_derivative(f, x, eps=1e-6):
    x_minus = Variable(x.data - eps)
    x_plus = Variable(x.data + eps)
    return (f(x_plus).data - f(x_minus).data) / (2 * eps)
