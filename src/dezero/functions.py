__all__ = ['Add', 'add', 'Sub', 'sub', 'Mul', 'mul', 'Div', 'div', 'Neg', 'neg', 'Pow', 'pow',
           'Square', 'square', 'Exp', 'exp', 'Sin', 'sin']
from . import __is_simple_core
from .core_simple import Variable

if __is_simple_core:
    from .core_simple import (Function, Add, Mul, Sub, Div, Neg, Pow, add, sub,
                              mul, div, pow, neg)
else:
    raise NotImplementedError('core module not implemented')

import numpy as np

Add = Add
Sub = Sub
Mul = Mul
Div = Div
Neg = Neg
Pow = Pow


add = add
sub = sub
mul = mul
div = div
neg = neg
pow = pow


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy * np.cos(self.inputs[0].data)


def sin(x):
    return Sin()(x)


if __name__ == '__main__':
    x = Variable(np.pi / 4)
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
