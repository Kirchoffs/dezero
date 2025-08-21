import numpy as np
import weakref
from .core import Parameter
from .functions import linear


class Layer:
    def __init__(self):
        self._params = set()

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        self.inputs = tuple(weakref.ref(x) for x in inputs)
        self.outputs = tuple(weakref.ref(y) for y in outputs)

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError("Forward method must be implemented in subclasses")

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def clear_grad(self):
        for param in self.params():
            param.clear_grad()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)


class Linear(Layer):
    # def __init__(self, in_size, out_size, has_bias = True, dtype = np.float32):
    #     super().__init__()
        
    #     I, O = in_size, out_size
    #     W_data = np.random.randn(I, O).astype(dtype) * np.sqrt(1 / I)
    #     self.W = Parameter(W_data, name = "W")
    #     if has_bias:
    #         self.b = Parameter(np.zeros(O, dtype = dtype), name = "b")
    #     else:
    #         self.b = None

    def __init__(self, in_size = None, out_size = None, has_bias = True, dtype = np.float32):
        super().__init__()
        if out_size is None:
            raise ValueError("out_size must be specified")
        
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = "W")
        if self.in_size is not None:
            self._init_W()

        if has_bias:
            self.b = Parameter(np.zeros(out_size, dtype = dtype), name = "b")
        else:
            self.b = None

    def _init_W(self):
        I, O = self.in_size, self.out_size
        
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        return linear(x, self.W, self.b)
