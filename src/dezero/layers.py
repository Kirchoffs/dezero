import numpy as np
import weakref
import os
from .core import Parameter
from .functions import linear
from .cuda import get_array_module


class Layer:
    def __init__(self):
        self._params = set()
        self._is_gpu_applied = False

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

    def save_weights(self, path):
        was_gpu = self._is_gpu_applied
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}

        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise

        if was_gpu:
            self.to_gpu()

    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]
        
        if self._is_gpu_applied:
            self.to_gpu()

    def _flatten_params(self, param_dict, parent_key = ''):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(param_dict, key)
            else:
                param_dict[key] = obj

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def to_cpu(self):
        self._is_gpu_applied = False
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        self._is_gpu_applied = True
        for param in self.params():
            param.to_gpu()


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

    def _init_W(self, xp = np):
        I, O = self.in_size, self.out_size
        
        W_data = xp.random.randn(I, O).astype(self.dtype) * xp.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        return linear(x, self.W, self.b)
