__all__ = ['Variable', 'Function', 'functions']

import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]

        while funcs:
            func = funcs.pop()
            x, y = func.input, func.output
            x.grad = func.backward(y.grad)

            if x.creator:
                funcs.append(x.creator)

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError(
                f'{Variable.__name__}.data must be numpy.ndarray or None')

        self.__data = data

    @property
    def grad(self):
        return self.__grad

    @grad.setter
    def grad(self, grad):
        if grad is not None and not isinstance(grad, np.ndarray):
            raise TypeError(
                f'{Variable.__name__}.grad must be numpy.ndarray or None')

        self.__grad = grad

    @property
    def creator(self):
        return self.__creator

    @creator.setter
    def creator(self, creator):
        if creator is not None and not isinstance(creator, Function):
            raise TypeError(
                f'{Variable.__name__}.creator must be {Function.__name__} or None')

        self.__creator = creator


class Function:
    def __init__(self):
        self.__input = None
        self.__output = None

    def __call__(self, input: Variable):
        if type(input) is not Variable:
            raise TypeError(f'input type must be {Variable.__name__} type')

        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)

        self.__input = input
        self.__output = output
        return output

    def get_intput(self):
        return self.__input

    def get_output(self):
        return self.__output

    input = property(get_intput)
    output = property(get_output)

    def _forward(self, x):
        raise NotImplementedError()

    def forward(self, x):
        """Do not overwrite this method, instead overwrite `_forward` method"""
        return self.__as_array(self._forward(x))

    def _backward(self, gy):
        raise NotImplementedError()

    def backward(self, gy):
        """Do not overwrite this method, instead overwrite `_backword` method"""
        return self.__as_array(self._backward(gy))

    def __as_array(self, x):
        if np.isscalar(x):
            return np.array(x)
        return x


if __name__ == '__main__':
    x = Variable(np.array(1.))
    x = Variable(None)
