import numpy as np


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
