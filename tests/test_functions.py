import unittest
import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dezero.functions import *  # nopep8
from dezero import *  # nopep8

RNG = np.random.default_rng(0)


def numerical_diff(f: Function, *orig_inputs: Variable | np.ndarray, eps=1e-6):
    orig_inputs = [as_variable(x) for x in orig_inputs]
    gxs = [np.zeros_like(x.data) for x in orig_inputs]

    with disable_backprob():
        for gx_idx, gx in enumerate(gxs):
            inputs = [x.data.copy() for x in orig_inputs]

            for x_elem_idx in range(gx.size):
                x0 = orig_inputs[gx_idx].data.copy()
                x1 = orig_inputs[gx_idx].data.copy()

                val = x0.reshape(-1)[x_elem_idx]

                x0.reshape(-1)[x_elem_idx] = val - eps
                x1.reshape(-1)[x_elem_idx] = val + eps

                inputs[gx_idx] = x0
                ys0: tuple[Variable] = f(*inputs)
                inputs[gx_idx] = x1
                ys1: tuple[Variable] = f(*inputs)

                if not isinstance(ys0, tuple) and not isinstance(ys0, list):
                    ys0 = (ys0, )
                if not isinstance(ys1, tuple) and not isinstance(ys1, list):
                    ys1 = (ys1, )

                grad = 0

                for y_idx in range(len(ys0)):
                    y0 = as_variable(ys0[y_idx])
                    y1 = as_variable(ys1[y_idx])

                    grad += np.sum((y1.data - y0.data) / eps / 2)

                gx.reshape(-1)[x_elem_idx] += grad

    return [as_variable(gx) for gx in gxs]


class FunctionTest(unittest.TestCase):
    MAX_DIM_SIZE = 10
    MAX_NDIM = 5

    def setUp(self) -> None:
        self._test_inputs: list[tuple] = []
        self._target_f = None
        self._exact_forward_f = None
        self._exact_backward_f = None

    def assertHasGradient(self, var: Variable):
        self.assertIsNotNone(var.grad)
        self.assertTrue(var.data.shape == var.grad.shape)

    def assertEqualVariable(self, actual, expected, atol=1e-8):
        actual = as_variable(actual)
        expected = as_variable(expected)
        self.assertTrue(actual.shape == expected.shape,
                        f'acutal shape = {actual.shape}, expected shape = {expected.shape}')

        if np.issubdtype(expected.data.dtype, np.floating):
            self.assertTrue(np.allclose(actual.data, expected.data, atol=atol))
        else:
            self.assertTrue(np.all(actual.data == expected.data))

    def assertForwardWithInput(self, target_f, exact_f, *inputs):
        for x in inputs:
            self.assertTrue(isinstance(x, np.ndarray),
                            f'x: {x} is type of {type(x)}')

        y_actual = target_f(*inputs)
        y_expected = exact_f(*inputs)

        self.assertEqualVariable(y_actual, y_expected)

    def assertBackwardWithInput(self, target_f, *inputs, exact_f=None):
        for x in inputs:
            self.assertTrue(isinstance(x, np.ndarray),
                            f'x: {x} is type of {type(x)}')

        if exact_f is None:
            eps = 1e-4
            gxs_expected = numerical_diff(target_f, *inputs, eps=1e-4)
        else:
            eps = 1e-8
            gxs_expected = exact_f(*inputs)

        if not isinstance(gxs_expected, tuple) and not isinstance(gxs_expected, list):
            gxs_expected = (gxs_expected, )

        self.assertEqual(len(inputs), len(gxs_expected))

        inputs = [as_variable(x) for x in inputs]
        with enable_backprob():
            ys = target_f(*inputs)

            if not isinstance(ys, tuple) and not isinstance(ys, list):
                ys = (ys,)

            for y in ys:
                y.backward()

        for i in range(len(inputs)):
            gx_actual = inputs[i].grad
            gx_expected = gxs_expected[i]

            with self.subTest(f'{i}th input\'s gradient check'):
                self.assertEqualVariable(gx_actual, gx_expected, atol=eps*10)

    def get_rand_test_input(self, shape=None, ndim=None, min_ndim=0, max_ndim=MAX_NDIM, max_dim_size=MAX_DIM_SIZE,
                            min_val=0, max_val=1, dtype=np.float32):
        if shape is None:
            if ndim is None:
                ndim = int(RNG.random() * (max_ndim - min_ndim + 1) + min_ndim)

            shape = []
            for _ in range(ndim):
                dim_size = int(RNG.random() * max_dim_size + 1)

                shape.append(dim_size)

        return np.array(RNG.random(shape, dtype=dtype) * (max_val - min_val) + min_val)

    def test_forward(self):
        if type(self) == FunctionTest:
            self.skipTest(
                f'Skip test for class {FunctionTest.__name__} which is provided for automated test case generation')

        self.assertTrue(len(self._test_inputs) > 0,
                        'empty _test_inputs, pleas set _test_inputs to automated test case generation')

        for test_input in self._test_inputs:
            self.assertForwardWithInput(
                self._target_f, self._exact_forward_f, *test_input)

    def test_backward(self):
        if type(self) == FunctionTest:
            self.skipTest(
                f'Skip test for class {FunctionTest.__name__} which is provided for automated test case generation')

        self.assertTrue(len(self._test_inputs) > 0,
                        'empty _test_inputs, pleas set _test_inputs to automated test case generation')

        for test_input in self._test_inputs:
            self.assertBackwardWithInput(
                self._target_f, *test_input, exact_f=self._exact_backward_f)


class SquareTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Square()
        self._exact_forward_f = lambda x: x * x
        self._exact_backward_f = lambda x: 2*x

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class ExpTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Exp()
        self._exact_forward_f = np.exp
        self._exact_backward_f = np.exp

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class SinTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Sin()
        self._exact_forward_f = np.sin
        self._exact_backward_f = np.cos

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class CosTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Cos()
        self._exact_forward_f = np.cos
        self._exact_backward_f = lambda x: -np.sin(x)

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class TanhTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Tanh()
        self._exact_forward_f = np.tanh

        def exact_bacward_f(x):
            tanh_val = np.tanh(x)
            return 1 - tanh_val * tanh_val
        self._exact_backward_f = exact_bacward_f

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class SigmoidTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Sigmoid()
        self._exact_forward_f = lambda x: 1 / (1 + np.exp(-x))

        def exact_bacward_f(x):
            sig_val = self._exact_forward_f(x)
            return sig_val * (1 - sig_val)
        self._exact_backward_f = exact_bacward_f

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class NegTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Neg()
        self._exact_forward_f = lambda x: -x
        self._exact_backward_f = lambda x: -np.ones_like(x)

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(ndim=0),))
            self._test_inputs.append((self.get_rand_test_input(ndim=1),))
            self._test_inputs.append((self.get_rand_test_input(ndim=2),))
            self._test_inputs.append((self.get_rand_test_input(ndim=3),))
            self._test_inputs.append((self.get_rand_test_input(ndim=4),))


class PowTest(FunctionTest):
    class PowToCTest(FunctionTest):
        def __init__(self, c):
            super().__init__()

            self.__c = c

        def setUp(self) -> None:
            super().setUp()

            self._target_f = Pow(self.__c)
            self._exact_forward_f = lambda x: np.power(x, self.__c)
            self._exact_backward_f = lambda x: self.__c * \
                np.power(x, self.__c - 1)

            for _ in range(10):
                self._test_inputs.append((self.get_rand_test_input(ndim=0), ))
                self._test_inputs.append((self.get_rand_test_input(ndim=1), ))
                self._test_inputs.append((self.get_rand_test_input(ndim=2), ))
                self._test_inputs.append((self.get_rand_test_input(ndim=3), ))
                self._test_inputs.append((self.get_rand_test_input(ndim=4), ))

    def setUp(self) -> None:
        super().setUp()

        self.__tests = []

        for _ in range(10):
            c = RNG.random() * 10 - 5
            test = self.PowToCTest(c)
            test.setUp()
            self.__tests.append(test)

    def test_forward(self):
        for test in self.__tests:
            test.test_forward()

    def test_backward(self):
        for test in self.__tests:
            test.test_backward()


class AddTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Add()
        self._exact_forward_f = lambda x0, x1: x0 + x1
        self._exact_backward_f = lambda x0, x1: (
            np.ones_like(x0), np.ones_like(x1))

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(
                ndim=0), self.get_rand_test_input(ndim=0)))

            x0 = self.get_rand_test_input(ndim=1)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=2)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=3)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=4)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))


class SubTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Sub()
        self._exact_forward_f = lambda x0, x1: x0 - x1
        self._exact_backward_f = lambda x0, x1: (
            np.ones_like(x0), -np.ones_like(x1))

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(
                ndim=0), self.get_rand_test_input(ndim=0)))

            x0 = self.get_rand_test_input(ndim=1)
            x1 = self.get_rand_test_input(shape=x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=2)
            x1 = self.get_rand_test_input(shape=x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=3)
            x1 = self.get_rand_test_input(shape=x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=4)
            x1 = self.get_rand_test_input(shape=x0.shape)
            self._test_inputs.append((x0, x1))


class MulTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Mul()
        self._exact_forward_f = lambda x0, x1: x0 * x1
        self._exact_backward_f = lambda x0, x1: (x1, x0)

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(
                ndim=0), self.get_rand_test_input(ndim=0)))

            x0 = self.get_rand_test_input(ndim=1)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=2)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=3)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=4)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))


class DivTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Div()
        self._exact_forward_f = lambda x0, x1: x0 / x1
        self._exact_backward_f = lambda x0, x1: (1 / x1, -x0 / (x1*x1))

        for _ in range(100):
            self._test_inputs.append((self.get_rand_test_input(
                ndim=0), self.get_rand_test_input(ndim=0)))

            x0 = self.get_rand_test_input(ndim=1)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=2)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=3)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))

            x0 = self.get_rand_test_input(ndim=4)
            x1 = RNG.random(x0.shape)
            self._test_inputs.append((x0, x1))


class ReshapeTest(FunctionTest):
    class _ReshapeTest(FunctionTest):
        def __init__(self, from_shape, to_shape):
            super().__init__()

            self._target_f = Reshape(to_shape)
            self._exact_forward_f = lambda x: np.reshape(x, to_shape)
            self._exact_backward_f = lambda x: np.ones_like(x)

            self._test_inputs = []
            for _ in range(10):
                self._test_inputs.append(
                    (self.get_rand_test_input(shape=from_shape), ))

    def setUp(self) -> None:
        super().setUp()

        self.__tests = [
            self._ReshapeTest((5, 5), (25,)),
            self._ReshapeTest((25,), (5, 5)),
            self._ReshapeTest((3, 4, 5), (5, 4, 3)),
            self._ReshapeTest((28, 1), (4, 7)),
            self._ReshapeTest((2, 4, 5), (8, 5)),
            self._ReshapeTest((2, 9), (3, 6)),
            self._ReshapeTest((4, 52, 2), (8, 26, 2)),
        ]

    def test_forward(self):
        for test in self.__tests:
            test.test_forward()

    def test_backward(self):
        for test in self.__tests:
            test.test_backward()


class TransposeTest(FunctionTest):
    class _TransposeTest(FunctionTest):
        def __init__(self, ndim, axes=None):
            super().__init__()

            self._target_f = Transpose(axes=axes)
            self._exact_forward_f = lambda x: np.transpose(x, axes=axes)
            self._exact_backward_f = lambda x: np.ones_like(x)

            self._test_inputs = []
            for _ in range(10):
                self._test_inputs.append(
                    (self.get_rand_test_input(ndim=ndim), ))

    def setUp(self) -> None:
        super().setUp()

        self.__tests = [
            self._TransposeTest(2,),
            self._TransposeTest(3),
            self._TransposeTest(4),
            self._TransposeTest(3, [1, 0, 2]),
            self._TransposeTest(4, [2, 0, 1, 3]),
            self._TransposeTest(4, [0, 2, 1, 3]),
            self._TransposeTest(4, [3, 1, 2, 0]),
        ]

    def test_forward(self):
        for test in self.__tests:
            test.test_forward()

    def test_backward(self):
        for test in self.__tests:
            test.test_backward()


if __name__ == '__main__':
    unittest.main()
