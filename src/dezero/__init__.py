__all__ = ['Variable', 'Function', 'functions']

from typing import Iterable
import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

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
        if data is not None:
            data = self.__as_array(data)
            if not isinstance(data, np.ndarray):
                raise TypeError(
                    f'{Variable.__name__}.data must be numpy.ndarray or None')

        self.__data = data

    @property
    def grad(self):
        return self.__grad

    @grad.setter
    def grad(self, grad):
        if grad is not None:
            grad = self.__as_array(grad)
            if not isinstance(grad, np.ndarray):
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

    def __as_array(self, x):
        if np.isscalar(x):
            return np.array(x)
        return x


class Function:
    def __init__(self):
        self.__inputs = None
        self.__outputs = None

    def __call__(self, inputs: Iterable[Variable]):
        if not self.__is_iterable_of_type(inputs, Variable):
            raise TypeError(
                f'inputs type must be iterable of {Variable.__name__}')

        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(y) for y in ys]

        for output in outputs:
            output.creator = self

        self.__inputs = inputs
        self.__outputs = outputs
        return outputs

    def get_intputs(self):
        return self.__inputs

    def get_outputs(self):
        return self.__outputs

    inputs = property(get_intputs)
    outputs = property(get_outputs)

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()

    def __is_iterable_of_type(self, iterable, type):
        try:
            for item in iterable:
                if type(item) is not type:
                    return False
            return True
        except:
            return False


if __name__ == '__main__':
    x = Variable(np.array(1.))
    x = Variable(None)
