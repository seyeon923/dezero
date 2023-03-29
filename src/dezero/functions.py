__all__ = ['Add', 'add', 'Sub', 'sub', 'Mul', 'mul', 'Div', 'div', 'Neg', 'neg', 'Pow', 'pow',
           'Square', 'square', 'Exp', 'exp', 'Sin', 'sin', 'Cos', 'cos']
from . import __is_simple_core
from .core_simple import Variable

if __is_simple_core:
    from .core_simple import (Function, Add, Mul, Sub, Div, Neg, Pow, add, sub,
                              mul, div, pow, neg)
else:
    from .core import (Function, Add, Mul, Sub, Div, Neg, Pow, add, sub,
                       mul, div, pow, neg)

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
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sin(Function):
    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy * cos(self.inputs[0])


def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        return -gy * sin(self.inputs[0])


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)


def tanh(x: Variable):
    return Tanh()(x)


if __name__ == '__main__':
    x = Variable(np.pi / 4)
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
