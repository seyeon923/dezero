__all__ = ['Variable', 'Function', 'functions']

import numpy as np
import weakref
import contextlib
from collections import deque


@contextlib.contextmanager
def using_config(name: str, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


def disable_backprob():
    return using_config('enable_backprob', False)


def enable_backprob():
    return using_config('enable_backprob', True)


class Config:
    enable_backprob = True


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = deque()
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)

        add_func(self.creator)

        while funcs:
            func = funcs.popleft()
            gys = [output().grad for output in func.outputs]
            gxs = func.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(func.inputs, gxs):
                if x.grad is None:
                    x.grad = gx.copy()
                else:
                    x.grad += gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in func.outputs:
                    y().grad = None

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

    def __call__(self, *inputs: Variable):
        if not self.__is_iterable_of_variable(inputs):
            raise TypeError(
                f'{Function.__name__} is only can be called with arguments of {Variable.__name__}')
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(y) for y in ys]

        if Config.enable_backprob:
            for output in outputs:
                output.creator = self

            self.__inputs = inputs
            self.__outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

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

    def __is_iterable_of_variable(self, iterable):
        try:
            for item in iterable:
                if type(item) is not Variable:
                    return False
            return True
        except:
            return False


if __name__ == '__main__':
    x = Variable(np.array(1.))
    x = Variable(None)

    Config.enable_backprob = False

    with enable_backprob():
        assert Config.enable_backprob == True
    assert Config.enable_backprob == False

    Config.enable_backprob = True
    with disable_backprob():
        assert Config.enable_backprob == False
    assert Config.enable_backprob == True
