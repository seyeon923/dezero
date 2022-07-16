from . import Function, Variable

import numpy as np


class Square(Function):
    def _forward(self, x):
        return x ** 2

    def _backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def _forward(self, x):
        return np.exp(x)

    def _backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


if __name__ == '__main__':
    assert square(Variable(np.array(10))).data == 100
    assert np.isclose(exp(Variable(np.array(1))).data, np.exp(1))

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    assert np.isclose(y.data, np.exp(0.25) * np.exp(0.25))

    A = Square()
    B = Exp()
    C = Square()
    a = A(x)
    b = B(a)
    y = C(b)

    assert y.creator == C
    assert C.input == b
    assert b.creator == B
    assert B.input == a
    assert a.creator == A
    assert A.input == x

    y = square(exp(square(x)))
    y.backward()
    print(x.grad)
