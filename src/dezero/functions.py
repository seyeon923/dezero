from .core_simple import (Function, Add, Mul, Sub, Div, Neg, Pow, add, sub,
                          mul, div, pow, neg)

import numpy as np

Add = Add
Sub = Sub
Mul = Mul
Div = Div
Neg = Neg
Pow = Pow


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


add = add
sub = sub
mul = mul
div = div
neg = neg
pow = pow


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)
