import unittest

from src.dezero.functions import *


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / 2 / eps


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.))
        y = square(x)
        expected = np.array(4.)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        expected = np.array(6.)
        self.assertAlmostEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        self.assertTrue(np.allclose(x.grad, num_grad))


class ExpTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.))
        y = exp(x)
        expected = np.exp(np.array(2.))
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.))
        y = exp(x)
        y.backward()
        expected = np.exp(np.array(3.))
        self.assertAlmostEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = exp(x)
        y.backward()
        num_grad = numerical_diff(exp, x)
        self.assertTrue(np.allclose(x.grad, num_grad))


if __name__ == '__main__':
    unittest.main()
