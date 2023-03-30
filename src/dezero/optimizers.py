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
