import numpy as np
import weakref
import os
from .core import Parameter
from .functions import linear, dropout, tanh, sigmoid
from .functions_conv import conv2d_simple, conv2d, deconv2d_simple, deconv2d
from .cuda import get_array_module
from .utils import pair


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


class Dropout(Layer):
    def __init__(self, dropout_ratio = 0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        return dropout(x, self.dropout_ratio)


class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride = 1, padding = 0, has_bias = True, in_channels = None, dtype = np.float32):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        if in_channels is not None:
            self._init_W()

        if has_bias:
            self.b = Parameter(np.zeros(out_channels, dtype = dtype), name = 'b')
        else:
            self.b = None

    def _init_W(self, xp = np):
        c, oc = self.in_channels, self.out_channels
        kh, kw = pair(self.kernel_size)
        scale = np.sqrt(1 / (c * kh * kw))
        W_data = xp.random.randn(oc, c, kh, kw).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        return conv2d(x, self.W, self.b, self.stride, self.padding)


class Conv2dSimple(Layer):
    def __init__(self, out_channels, kernel_size, stride = 1, padding = 0, has_bias = True, in_channels = None, dtype = np.float32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if has_bias:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self, xp = np):
        c, oc = self.in_channels, self.out_channels
        kh, kw = pair(self.kernel_size)
        scale = np.sqrt(1 / (c * kh * kw))
        W_data = xp.random.randn(oc, c, kh, kw).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        return conv2d_simple(x, self.W, self.b, self.stride, self.padding)


class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride = 1, padding = 0, has_bias = True, in_channels = None, out_size = None, dtype = np.float32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        if in_channels is not None:
            self._init_W()

        if has_bias:
            self.b = Parameter(np.zeros(out_channels, dtype = dtype), name = 'b')
        else:
            self.b = None

    def _init_W(self, xp = np):
        c, oc = self.in_channels, self.out_channels
        kh, kw = pair(self.kernel_size)
        scale = np.sqrt(1 / (c * kh * kw))
        W_data = xp.random.randn(c, oc, kh, kw).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        return deconv2d(x, self.W, self.b, self.stride, self.padding, self.out_size)


class Deconv2dSimple(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, padding=0, has_bias=True, in_channels=None, out_size=None, dtype=np.float32):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        if in_channels is not None:
            self._init_W()

        if has_bias:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')
        else:
            self.b = None

    def _init_W(self, xp = np):
        c, oc = self.in_channels, self.out_channels
        kh, kw = pair(self.kernel_size)
        scale = np.sqrt(1 / (c * kh * kw))
        W_data = xp.random.randn(c, oc, kh, kw).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            xp = get_array_module(x)
            self._init_W(xp)

        return deconv2d_simple(x, self.W, self.b, self.stride, self.padding, self.out_size)


class RNN(Layer):
    def __init__(self, hidden_size, in_size = None):
        super().__init__()

        self.x2h = Linear(in_size = in_size, out_size = hidden_size)
        self.h2h = Linear(in_size = in_size, out_size = hidden_size, has_bias = True)
        self.h = None

    def reset_state(self):
        self.h = None
    
    def forward(self, x):
        if self.h is None:
            h_new = tanh(self.x2h(x))
        else:
            h_new = tanh(self.x2h(x) + self.h2h(self.h))
        
        self.h = h_new
        return h_new


class LSTM(Layer):
    def __init__(self, hidden_size, in_size = None):
        super().__init__()

        self.x2f = Linear(in_size = in_size, out_size = hidden_size)
        self.x2i = Linear(in_size = in_size, out_size = hidden_size)
        self.x2o = Linear(in_size = in_size, out_size = hidden_size)
        self.x2u = Linear(in_size = in_size, out_size = hidden_size)

        self.h2f = Linear(in_size = hidden_size, out_size = hidden_size, has_bias = False)
        self.h2i = Linear(in_size = hidden_size, out_size = hidden_size, has_bias = False)
        self.h2o = Linear(in_size = hidden_size, out_size = hidden_size, has_bias = False)
        self.h2u = Linear(in_size = hidden_size, out_size = hidden_size, has_bias = False)

        self.reset_state()

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            f = sigmoid(self.x2f(x))
            i = sigmoid(self.x2i(x))
            o = sigmoid(self.x2o(x))
            u = tanh(self.x2u(x))
        else:
            f = sigmoid(self.x2f(x) + self.h2f(self.h))
            i = sigmoid(self.x2i(x) + self.h2i(self.h))
            o = sigmoid(self.x2o(x) + self.h2o(self.h))
            u = tanh(self.x2u(x) + self.h2u(self.h))

        if self.c is None:
            c_new = i * u
        else:
            c_new = f * self.c + i * u

        h_new = o * tanh(c_new)

        self.h, self.c = h_new, c_new
        return h_new
