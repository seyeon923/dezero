import numpy as np
import weakref
import contextlib
import heapq


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


def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    def __init__(self, name=None):
        self.__inputs = None
        self.__outputs = None
        self.__generation = None
        self.__name = name

    def __call__(self, *inputs):
        inputs = [as_variable(input) for input in inputs]

        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [as_variable(y) for y in ys]

        if Config.enable_backprob:
            self.__generation = max([x.generation for x in inputs])
            for output in outputs:
                output.creator = self

            self.__inputs = inputs
            self.__outputs = [weakref.ref(output) for output in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def get_intputs(self):
        return self.__inputs

    def get_outputs(self):
        return self.__outputs

    def get_generation(self):
        return self.__generation

    def get_name(self):
        return self.__name

    inputs = property(get_intputs)
    outputs = property(get_outputs)
    generation = property(get_generation)
    name = property(get_name)

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        return (x0 + x1,)

    def backward(self, gy):
        return gy, gy


class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1

    def backward(self, gy):
        return gy, -gy


class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy * x1, gy * x0


class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0 = self.inputs[0].data
        x1 = self.inputs[1].data
        return gy / x1, -gy * x0 / (x1 * x1)


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy


class Pow(Function):
    def __init__(self, c):
        self.__c = c

    def forward(self, x):
        return x ** self.__c

    def backward(self, gy):
        x = self.inputs[0].data
        c = self.__c
        return c * x ** (c-1) * gy


def pow(x, c):
    return Pow(c)(x)


def neg(x):
    return Neg()(x)


def add(x0, x1):
    return Add()(x0, x1)


def sub(x0, x1):
    return Sub()(x0, x1)


def mul(x0, x1):
    return Mul()(x0, x1)


def div(x0, x1):
    return Div()(x0, x1)


class Variable:
    __array_priority__ = 200

    def __init__(self, data, name=None):
        self.data = data
        self.grad = None
        self.creator = None
        self.__generation = 0
        if name is not None and not isinstance(name, str):
            raise TypeError('Variable.name must be str')
        self.__name = name

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []  # (-generation, input_idx, func)
        seen_set = set()

        input_idx = 0

        def add_func(f):
            nonlocal input_idx
            if f not in seen_set:
                seen_set.add(f)
                heapq.heappush(funcs, (-f.generation, input_idx, f))
                input_idx += 1

        add_func(self.creator)

        while funcs:
            _, _, func = heapq.heappop(funcs)
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

    def get_name(self):
        return self.__name

    name = property(get_name)

    @property
    def generation(self):
        return self.__generation

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, data):
        if data is not None:
            data = np.array(data)

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
        if creator is not None:
            self.__generation = creator.generation + 1
            if not isinstance(creator, Function):
                raise TypeError(
                    f'{Variable.__name__}.creator must be {Function.__name__} or None')

        self.__creator = creator

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __str__(self):
        name = ''
        if self.name:
            name = f"'{self.name}'"

        if self.ndim > 1:
            numpy_str = f'\n{repr(self.data)}'
        else:
            numpy_str = repr(self.data)
        return f'<dezero.Variable {name} shape={self.shape} dtype={self.dtype}, numpy={numpy_str}>'

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return f'Variable({p})'

    def __as_array(self, x):
        if np.isscalar(x):
            return np.array(x)
        return x


Variable.__add__ = add
Variable.__radd__ = add
Variable.__sub__ = sub
Variable.__rsub__ = lambda self, other: sub(other, self)
Variable.__mul__ = mul
Variable.__rmul__ = mul
Variable.__truediv__ = div
Variable.__rtruediv__ = lambda self, other: div(other, self)
Variable.__neg__ = neg
Variable.__pow__ = pow
