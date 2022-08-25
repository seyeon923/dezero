from . import Function, Variable

import numpy as np


class Add(Function):
    def forward(self, x0, x1):
        return (x0 + x1,)

    def backward(self, gy):
        return gy, gy


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


def add(x0, x1):
    return Add()(x0, x1)


def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)


if __name__ == '__main__':
    assert np.isclose(add(Variable(2.), Variable(3.)).data, 5.)

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
    assert C.inputs == (b,)
    assert b.creator == B
    assert B.inputs == (a,)
    assert a.creator == A
    assert A.inputs == (x,)

    y = square(exp(square(x)))
    y.backward()
    assert np.isclose(x.grad, 3.2974425)

    x = Variable(2.)
    y = Variable(3.)
    z = add(square(x), square(y))
    z.backward()
    assert np.isclose(z.data, 13., atol=1e-12)
    assert np.isclose(x.grad, 4., atol=1e-12)
    assert np.isclose(y.grad, 6., atol=1e-12)

    x = Variable(3.)
    y = add(x, x)
    y.backward()
    assert y.grad == 1.
    assert np.isclose(x.grad, 2., 1e-12)

    x.cleargrad()
    y = add(add(x, x), x)
    y.backward()
    assert y.grad == 1.
    assert np.isclose(x.grad, 3., atol=1e-12)
