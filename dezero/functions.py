from __init__ import Function, Variable

import numpy as np


class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)


if __name__ == '__main__':
    square = Square()
    exp = Exp()

    assert square(Variable(np.array(10))).data == 100
    assert np.isclose(exp(Variable(np.array(1))).data, np.exp(1))

    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    assert np.isclose(y.data, np.exp(0.25) * np.exp(0.25))
