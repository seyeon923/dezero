__all__ = ['Add', 'add', 'Sub', 'sub', 'Mul', 'mul', 'Div', 'div', 'Neg', 'neg', 'Pow', 'pow',
           'Square', 'square', 'Exp', 'exp', 'Sin', 'sin', 'Cos', 'cos', 'Tanh', 'tanh',
           'Reshape', 'reshape', 'Transpose', 'transpose', 'SumTo', 'sum_to', 'BroadcastTo', 'broadcast_to',
           'Matmul', 'matmul', 'MatmulAdd', 'matmul_add', 'Sum', 'sum', 'MSE', 'mse', 'Sigmoid', 'sigmoid',
           'GetItem', 'get_item', 'Softmax', 'softmax']
import numpy as np

from . import __is_simple_core

if __is_simple_core:
    from .core_simple import (Variable, as_variable, Function, Add, Mul, Sub, Div, Neg, Pow, add, sub,
                              mul, div, pow, neg)
else:
    from .core import (Variable, as_variable, Function, Add, Mul, Sub, Div, Neg, Pow, add, sub,
                       mul, div, pow, neg)


from . import utils

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
    def __init__(self):
        super().__init__(name='Square')

    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


class Exp(Function):
    def __init__(self):
        super().__init__(name='Exp')

    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0]
        gx = exp(x) * gy
        return gx


def exp(x):
    return Exp()(x)


class Sin(Function):
    def __init__(self):
        super().__init__(name='Sin')

    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        return gy * cos(self.inputs[0])


def sin(x):
    return Sin()(x)


class Cos(Function):
    def __init__(self):
        super().__init__(name='Cos')

    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        return -gy * sin(self.inputs[0])


def cos(x):
    return Cos()(x)


class Tanh(Function):
    def __init__(self):
        super().__init__(name='Tanh')

    def forward(self, x):
        return np.tanh(x)

    def backward(self, gy):
        y = self.outputs[0]()
        return gy * (1 - y * y)


def tanh(x):
    return Tanh()(x)


class Reshape(Function):
    def __init__(self, shape):
        super().__init__(name=f'Reshape({shape})')
        self.__shape = shape

    def forward(self, x: np.ndarray):
        self.__x_shape = x.shape
        return x.reshape(self.__shape)

    def backward(self, gy):
        return reshape(gy, self.__x_shape)


def reshape(x: Variable | np.ndarray, shape):
    if x.shape == shape:
        return as_variable(x)
    return Reshape(shape)(x)


class Transpose(Function):
    def __init__(self, axes=None):
        name = 'Transpose' if axes is None else f'Transpose({axes})'
        super().__init__(name=name)

        self.__axes = axes
        self.__inv_axes = self.__get_inv_axes()

    def forward(self, x):
        return np.transpose(x, self.__axes)

    def backward(self, gy):
        return transpose(gy, self.__inv_axes)

    def __get_inv_axes(self):
        if self.__axes == None:
            return None
        inv_axes = []

        for i in range(len(self.__axes)):
            inv_axes.append(self.__axes.index(i))

        return inv_axes


def transpose(x, axes=None):
    return Transpose(axes=axes)(x)


class SumTo(Function):
    def __init__(self, shape):
        super().__init__(name=f'SumTo({shape})')
        self.__shape = shape
        self.__x_shape = None

    def forward(self, x):
        self.__x_shape = x.shape
        return utils.sum_to(x, self.__shape)

    def backward(self, gy):
        return broadcast_to(gy, self.__x_shape)


def sum_to(x, shape):
    if x.shape == tuple(shape):
        return as_variable(x)
    return SumTo(shape)(x)


class BroadcastTo(Function):
    def __init__(self, shape):
        super().__init__(f'BroadcastTo({shape})')
        self.__shape = shape
        self.__x_shape = None

    def forward(self, x):
        self.__x_shape = x.shape
        return np.broadcast_to(x, self.__shape)

    def backward(self, gy):
        return sum_to(gy, self.__x_shape)


def broadcast_to(x, shape):
    if x.shape == tuple(shape):
        return as_variable(x)
    return BroadcastTo(shape)(x)


class Matmul(Function):
    def __init__(self):
        super().__init__('MatMul')

    def forward(self, x0, x1):
        return np.matmul(x0, x1)

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = matmul(gy, x1.T)
        gx1 = matmul(x0.T, gy)

        return gx0, gx1


def matmul(x0, x1):
    return Matmul()(x0, x1)


class MatmulAdd(Function):
    def __init__(self):
        super().__init__(name='MatMulAdd')

    def forward(self, x, w, b):
        return np.matmul(x, w) + b

    def backward(self, gy):
        x, w, b = self.inputs

        gx = matmul(gy, w.T)
        gw = matmul(x.T, gy)
        gb = sum_to(gy, b.shape)
        return gx, gw, gb


def matmul_add(x, w, b=None):
    if b is None:
        return matmul(x, y)
    return MatmulAdd()(x, w, b)


class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        super().__init__(name=f'Sum(axis={axis}, keepdims={keepdims})')
        self.__axis = axis
        self.__keepdims = keepdims
        self.__x_shape = None

    def forward(self, x):
        self.__x_shape = x.shape
        return np.sum(x, axis=self.__axis, keepdims=self.__keepdims)

    def backward(self, gy):
        if not self.__keepdims:
            shape = list(self.__x_shape)

            axis = self.__axis

            if axis is None:
                axis = list(range(len(shape)))
            if isinstance(self.__axis, int):
                axis = (axis, )

            for ax in axis:
                shape[ax] = 1

            gy = reshape(gy, shape)

        return broadcast_to(gy, self.__x_shape)


def sum(x: Variable, axis=None, keepdims=False):
    return Sum(axis=axis, keepdims=keepdims)(x)


class MSE(Function):
    def __init__(self):
        super().__init__(name='MSE')

    def forward(self, x0, x1):
        diff = x0 - x1

        return np.sum(diff*diff) / diff.size

    def backward(self, gy):
        x0, x1 = self.inputs

        diff = x0 - x1

        gx0 = diff * 2 / diff.size * gy
        gx1 = -gx0

        return gx0, gx1


def mse(x0: Variable, x1: Variable):
    return MSE()(x0, x1)


class Sigmoid(Function):
    def __init__(self):
        super().__init__(name='Sigmoid')

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, gy):
        y = self.outputs[0]()
        return y*(1-y) * gy


def sigmoid(x: Variable):
    return Sigmoid()(x)


class GetItem(Function):
    def __init__(self, slices):
        super().__init__(name=f'GetItem({slices})')

        self.__slices = slices

    def forward(self, x):
        return x[self.__slices]

    def backward(self, gy):
        return GetItemGrad(self.__slices, self.inputs[0].shape)(gy)


def get_item(x, slices):
    return GetItem(slices)(x)


class GetItemGrad(Function):
    def __init__(self, gx_slices, gx_shape, gy_slices=slice(None, None, None)):
        super().__init__(
            name=f'GetItemGrad(gx_slices={gx_slices}, gx_shape={gx_shape}, gy_slices={gy_slices})')

        self.__gx_slices = gx_slices
        self.__gx_shape = gx_shape
        self.__gy_slices = gy_slices

    def forward(self, gy):
        gx = np.zeros(self.__gx_shape, dtype=gy.dtype)

        gx[self.__gx_slices] = gy[self.__gy_slices]
        return gx

    def backward(self, ggx):
        return get_item(ggx, self.__slices)


class Softmax(Function):
    def __init__(self, axis=-1):
        super().__init__(name=f'Softmax({axis})')

        self.__axis = axis

    def forward(self, x):
        x = x - np.max(x, axis=self.__axis, keepdims=True)
        e = np.exp(x)
        return e / np.sum(e, axis=self.__axis, keepdims=True)

    def backward(self, gy):
        y = self.outputs[0]()

        gx = y * gy
        sumdx = sum(gx, axis=self.__axis, keepdims=True)
        gx -= y * sumdx

        return gx


def softmax(x, axis=-1):
    return Softmax(axis=axis)(x)


class Clip(Function):
    def __init__(self, min_val, max_val):
        super().__init__(name=f'Clip({min_val}, {max_val})')

        self.__min_val = min_val
        self.__max_val = max_val

        self.__mask = None

    def forward(self, x):
        self.__mask = (x >= self.__min_val) & (x <= self.__max_val)
        lt_min_mask = x < self.__min_val
        gt_max_mask = x > self.__max_val

        y = np.zeros_like(x)
        y[lt_min_mask] = self.__min_val
        y[self.__mask] = x[self.__mask]
        y[gt_max_mask] = self.__max_val

        return y

    def backward(self, gy):
        return GetItemGrad(self.__mask, self.inputs[0].shape, self.__mask)(gy)


def clip(x, min_val, max_val):
    return Clip(min_val, max_val)(x)


class Log(Function):
    def __init__(self):
        super().__init__(name='Log')

    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        x, = self.inputs

        return gy / x


def log(x):
    return Log()(x)


def softmax_cross_entropy_simple(x, t):
    x, t = as_variable(x), as_variable(t)
    n = np.sum(x.shape[:-1])

    p = softmax(x)
    p = clip(p, 1e-15, 1.0)
    log_p = reshape(log(p), (n, -1))
    assert log_p.shape == (n, x.shape[-1])
    slices = (range(n), t.data)
    tlog_p = get_item(log_p, slices)
    return -1 * sum(tlog_p) / n


if __name__ == '__main__':
    x = Variable(np.pi / 4)
    y = sin(x)
    y.backward()

    print(y.data)
    print(x.grad)
