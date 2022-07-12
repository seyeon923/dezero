from __init__ import Function, Variable

import numpy as np


class Square(Function):
    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    square = Square()
    assert square(Variable(np.array(10))).data == 100
