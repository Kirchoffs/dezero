import numpy as np
import contextlib
import weakref
import heapq
from . import cuda


try:
    import cupy as cp
    array_types = (np.ndarray, cp.ndarray)
except ImportError:
    array_types = (np.ndarray)


class Config:
    enable_backprop = True
    train = True


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


def test_mode():
    return using_config("train", False)


class Variable:
    __array_priority__ = 256

    def __init__(self, data, name = None):
        if data is not None and not isinstance(data, array_types):
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
    
    @property
    def T(self):
        return self.transpose()

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
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
            xp = cuda.get_array_module(self.data)
            self.grad = Variable(xp.ones_like(self.data))

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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]

        from .functions import reshape
        return reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        
        from .functions import transpose
        return transpose(self, axes)

    def max(self, axis = None, keepdims = False):
        from .functions import max
        return max(self, axis, keepdims)

    def unchain(self):
        self.creator = None

    def unchain_backward(self):
        if self.creator is not None:
            funcs = [self.creator]
            while funcs:
                f = funcs.pop()
                for x in f.inputs:
                    if x.creator is not None:
                        funcs.append(x.creator)
                        x.unchain()

    def __getitem__(self, slices):
        from .functions import get_item
        return get_item(self, slices)

    def __array__(self, dtype = None):
        return self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return "Variable(None)"
        content = str(self.data).replace('\n', '\n' + ' ' * len("Variable("))
        return f"Variable(" + content + ")"

    def to_cpu(self):
        if self.data is not None:
            self.data = cuda.as_numpy(self.data)

    def to_gpu(self):
        if self.data is not None:
            self.data = cuda.as_cupy(self.data)


def as_variable(obj):
    if not isinstance(obj, Variable):
        obj = Variable(obj)
    return obj


def as_array(x, array_module = np):
    if array_module.isscalar(x):
        return array_module.array(x)
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
        # forward operations are based on ndarray
        raise NotImplementedError("Forward method must be implemented by subclasses")

    def backward(self, gy):
        # backward operations are based on Variable
        raise NotImplementedError("Backward method must be implemented by subclasses")
    
    def __lt__(self, other):
        return self.generation > other.generation


class Parameter(Variable):
    pass
