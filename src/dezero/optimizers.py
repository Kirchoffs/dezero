import numpy as np
import math

from .cuda import get_array_module


class Optimizer:
    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        self.target = target
        return self
    
    def update(self):
        params = [param for param in self.target.params() if param.grad is not None]

        for hook in self.hooks:
            hook(params)
        
        for param in params:
            self.update_one(param)
        
    def update_one(self, param):
        raise NotImplementedError("Update method must be implemented in subclasses")

    def add_hook(self, hook):
        self.hooks.append(hook)


class SGD(Optimizer):
    def __init__(self, lr = 0.01):
        super().__init__()

        self.lr = lr

    def update_one(self, param):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr = 0.01, momentum_coeff = 0.9):
        super().__init__()

        self.lr = lr
        self.beta = momentum_coeff
        self.momentum_dict = dict()

    def update_one(self, param):
        parem_id = id(param)
        if parem_id not in self.momentum_dict:
            self.momentum_dict[parem_id] = np.zeros_like(param.data)
        
        momentum = self.momentum_dict[parem_id]
        momentum *= self.beta
        momentum += self.lr * param.grad.data
        param.data -= momentum


class AdaGrad(Optimizer):
    def __init__(self, lr = 1e-3, eps = 1e-8):
        super().__init__()

        self.lr = lr
        self.eps = eps
        self.grad_squared_dict = {}

    def update_one(self, param):
        xp = get_array_module(param.data)

        param_id = id(param)
        if param_id not in self.grad_squared_dict:
            self.grad_squared_dict[param_id] = xp.zeros_like(param.data)

        lr = self.lr
        eps = self.eps
        grad = param.grad.data
        grad_squared = self.grad_squared_dict[param_id]

        grad_squared += grad * grad
        param.data -= lr * grad / (xp.sqrt(grad_squared) + eps)


class Adam(Optimizer):
    def __init__(self, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        super().__init__()

        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    def update_one(self, param):
        xp = get_array_module(param.data)

        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param.data)
            self.vs[key] = xp.zeros_like(param.data)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = param.grad.data

        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)

        # m_hat = m_t / (1 - beta1^t)
        # v_hat = v_t / (1 - beta2^t)
        m_hat = m / (1.0 - math.pow(beta1, self.t))
        v_hat = v / (1.0 - math.pow(beta2, self.t))

        param.data -= (self.alpha / (xp.sqrt(v_hat) + eps)) * m_hat
