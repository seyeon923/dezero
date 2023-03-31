import unittest
import sys
import os

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dezero.functions import *  # nopep8
from dezero import *  # nopep8

RNG = np.random.default_rng(0)


def numerical_diff(f: Function, *inputs: Variable | np.ndarray, eps=1e-6):
    inputs = [as_variable(x) for x in inputs]

    gxs = [np.zeros_like(x.data) for x in inputs]

    with disable_backprob():
        for i, gx in enumerate(gxs):
            numel = gx.data.size
            for i in range(numel):
                x0 = inputs[i].data.copy()
                x1 = inputs[i].data.copy()

                val = x0.reshape(-1)[i]

                x0.reshape(-1)[i] = val - eps
                x1.reshape(-1)[i] = val + eps

                y0: Variable = f(x0)
                y1: Variable = f(x1)

                val0 = y0.data.reshape(-1)[i]
                val1 = y1.data.reshape(-1)[i]

                grad = (val1 - val0) / eps / 2

                gx.reshape(-1)[i] = grad

    return [as_variable(gx) for gx in gxs]


class FunctionTest(unittest.TestCase):
    MAX_DIM_SIZE = 10
    MAX_NDIM = 5

    def assertHasGradient(self, var: Variable):
        self.assertIsNotNone(var.grad)
        self.assertTrue(var.data.shape == var.grad.shape)

    def assertEqualVariable(self, actual, expected, atol=1e-8):
        actual = as_variable(actual)
        expected = as_variable(expected)
        self.assertTrue(actual.shape == expected.shape)

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

        with self.subTest(inputs=inputs, target_f=target_f):
            self.assertEqualVariable(y_actual, y_expected)

    def assertBackwardWithInput(self, target_f, *inputs, exact_f=None):
        for x in inputs:
            self.assertTrue(isinstance(x, np.ndarray),
                            f'x: {x} is type of {type(x)}')

        if exact_f is None:
            gxs_expected = numerical_diff(target_f, *inputs)
        else:
            gxs_expected = [exact_f(x) for x in inputs]

        inputs = [as_variable(x) for x in inputs]
        with enable_backprob():
            y: Variable = target_f(*inputs)
            y.backward()

        with self.subTest(inputs=inputs, target_f=target_f):
            self.assertEqual(len(inputs), len(gxs_expected))

            for i in range(len(inputs)):
                gx_actual = inputs[i].grad
                gx_expected = gxs_expected[i]

                with self.subTest(f'{i}th input'):
                    self.assertEqualVariable(gx_actual, gx_expected)

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


class UnaryFuncTest(FunctionTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = None
        self._exact_forward_f = None
        self._exact_backward_f = None
        self._test_dims = []
        self.num_tests = 100

    def _test_forward_nd(self, ndim):
        with self.subTest(f'{ndim}d input test'):
            for _ in range(self.num_tests):
                x = self.get_rand_test_input(ndim=ndim)
                self.assertForwardWithInput(
                    self._target_f, self._exact_forward_f, x)

    def _test_backward_nd(self, ndim):
        with self.subTest(f'{ndim}d input test'):
            for _ in range(self.num_tests):
                x = self.get_rand_test_input(ndim=ndim)
                self.assertBackwardWithInput(
                    self._target_f, x, exact_f=self._exact_backward_f)

    def test_forward(self):
        if type(self) == UnaryFuncTest:
            self.skipTest(
                f'Skip test for class {UnaryFuncTest.__name__} which is provided for automated test case generation')

        self.assertTrue(len(self._test_dims) > 0,
                        'empty _test_dims, pleas set _test_dims to automated test case generation')
        for dim in self._test_dims:
            self._test_forward_nd(dim)

    def test_backward(self):
        if type(self) == UnaryFuncTest:
            self.skipTest(
                f'Skip test for class {UnaryFuncTest.__name__} which is provided for automated test case generation')

        self.assertTrue(len(self._test_dims) > 0,
                        'empty _test_dims, pleas set _test_dims to automated test case generation')

        for dim in self._test_dims:
            self._test_backward_nd(dim)


class SquareTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Square()
        self._exact_forward_f = lambda x: x * x
        self._exact_backward_f = lambda x: 2*x
        self._test_dims = range(5)


class ExpTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Exp()
        self._exact_forward_f = np.exp
        self._exact_backward_f = np.exp
        self._test_dims = range(5)


class SinTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Sin()
        self._exact_forward_f = np.sin
        self._exact_backward_f = np.cos
        self._test_dims = range(5)


class CosTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Cos()
        self._exact_forward_f = np.cos
        self._exact_backward_f = lambda x: -np.sin(x)
        self._test_dims = range(5)


class TanhTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Tanh()
        self._exact_forward_f = np.tanh

        def exact_bacward_f(x):
            tanh_val = np.tanh(x)
            return 1 - tanh_val * tanh_val
        self._exact_backward_f = exact_bacward_f
        self._test_dims = range(5)


class SigmoidTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Sigmoid()
        self._exact_forward_f = lambda x: 1 / (1 + np.exp(-x))

        def exact_bacward_f(x):
            sig_val = self._exact_forward_f(x)
            return sig_val * (1 - sig_val)
        self._exact_backward_f = exact_bacward_f
        self._test_dims = range(5)


class NegTest(UnaryFuncTest):
    def setUp(self) -> None:
        super().setUp()

        self._target_f = Neg()
        self._exact_forward_f = lambda x: -x
        self._exact_backward_f = lambda x: -np.ones_like(x)
        self._test_dims = range(5)


class PowTest(FunctionTest):
    class PowToCTest(UnaryFuncTest):
        def __init__(self, c):
            super().__init__()

            self.__c = c

        def setUp(self) -> None:
            super().setUp()

            self._target_f = Pow(self.__c)
            self._exact_forward_f = lambda x: np.power(x, self.__c)
            self._exact_backward_f = lambda x: self.__c * \
                np.power(x, self.__c - 1)
            self._test_dims = range(5)
            self.num_tests = 10

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


if __name__ == '__main__':
    unittest.main()
