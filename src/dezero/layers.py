import weakref
from typing import Generator

import numpy as np

from . import Parameter, Variable
from . import functions


class Layer:
    def __init__(self):
        self._params: set[str] = set()

    def __setattr__(self, name: str, value) -> None:
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs, )

        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, *inputs):
        raise NotImplementedError()

    def params(self) -> Generator[Parameter, None, None]:
        for name in self._params:
            obj = getattr(self, name)

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()


class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.__in_size = in_size
        self.__out_size = out_size
        self.__dtype = dtype

        self.__weights = Parameter(None, name='W')

        if in_size is not None:
            self.__init_weights()

        if nobias:
            self.__bias = None
        else:
            self.__bias = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def __init_weights(self):
        in_size, out_size, dtype = self.__in_size, self.__out_size, self.__dtype
        self.__weights.data = np.random.randn(
            in_size, out_size).astype(dtype) * np.sqrt(1/in_size)

    def forward(self, x):
        if self.__weights.data is None:
            self.__in_size = x.shape[-1]
            self.__init_weights()

        return functions.matmul_add(x, self.__weights, self.__bias)


if __name__ == '__main__':
    layer = Layer()

    layer.p1 = Parameter(np.array(1))
    layer.p2 = Parameter(np.array(2))
    layer.p3 = Variable(np.array(3))
    layer.p4 = 'test'

    print(layer._params)
    print('-----------')

    for name in layer._params:
        print(name, layer.__dict__[name])
