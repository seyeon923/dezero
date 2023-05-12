import numpy as np

from . import Layer, Variable


class Optimizer:
    def __init__(self):
        self.__target: Layer = None
        self.__hooks = []

    def setup(self, target):
        self.__target = target
        return self

    def update(self):
        params = [p for p in self.__target.params() if p.grad is not None]

        # preprocessing
        for f in self.__hooks:
            f(params)

        # update params
        for param in params:
            self.update_one(param)

    def update_one(self, param: Variable):
        raise NotImplementedError()

    def add_hook(self, f):
        self.__hooks.append(f)


class SGD(Optimizer):
    def __init__(self, lr=0.01):
        super().__init__()

        self.lr = lr

    def update_one(self, param: Variable):
        param.data -= self.lr * param.grad.data


class MomentumSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0.9):
        super().__init__()

        self.lr = lr
        self.momentum = momentum
        self.__vs = {}

    def update_one(self, param):
        v_key = id(param)
        if v_key not in self.__vs:
            self.__vs[v_key] = np.zeros_like(param.data)

        v = self.__vs[v_key]

        v *= self.momentum
        v -= self.lr * param.grad.data
        param.data += v


class RMSprop(Optimizer):
    def __init__(self, lr=0.01, beta=0.99):
        super().__init__()

        self.lr = lr
        self.beta = beta

        self.__s = {}

    def update_one(self, param):
        s_key = id(param)
        if s_key not in self.__s:
            self.__s[s_key] = np.zeros_like(param.data)

        s = self.__s[s_key]

        grad = param.grad.data

        s = (self.beta * s) + (1 - self.beta) * grad * grad

        param.data -= self.lr * grad / np.sqrt(s)


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__()

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.__v = {}
        self.__s = {}

        self.__t = {}

    def update_one(self, param):
        param_key = id(param)
        if param_key not in self.__v:
            self.__v[param_key] = np.zeros_like(param.data)
            self.__s[param_key] = np.zeros_like(param.data)

            self.__t[param_key] = 1

        v = self.__v[param_key]
        s = self.__s[param_key]

        t = self.__t[param_key]

        grad = param.grad.data

        v = (self.beta1 * v) + (1 - self.beta1) * grad
        s = (self.beta2 * s) + (1 - self.beta2) * grad * grad

        v /= 1 - self.beta1**t
        s /= 1 - self.beta2**t

        param.data -= self.lr * v / (np.sqrt(s) + self.epsilon)
