import numpy as np
import gc
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from dezero.functions import *  # nopep8
from dezero import *  # nopep8

x = Variable(np.array(1.))
x = Variable(None)

Config.enable_backprob = False

with enable_backprob():
    assert Config.enable_backprob == True
assert Config.enable_backprob == False

Config.enable_backprob = True
with disable_backprob():
    assert Config.enable_backprob == False
assert Config.enable_backprob == True

x = Variable(1.)
assert x.shape == ()
assert x.ndim == 0
assert x.size == 1
assert np.issubdtype(x.dtype, np.floating)

x = Variable([1., 2.])
assert x.shape == (2,)
assert x.ndim == 1
assert x.size == 2
assert np.issubdtype(x.dtype, np.floating)
assert len(x) == 2

x = Variable([[1, 2], [3, 4], [5, 6]])
assert x.shape == (3, 2)
assert x.ndim == 2
assert x.size == 6
assert np.issubdtype(x.dtype, np.integer)
assert len(x) == 3

x = Variable(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9],
                      dtype=np.float64).reshape((1, 3, 3)), name='x')
assert x.shape == (1, 3, 3)
assert x.ndim == 3
assert x.size == 9
assert x.dtype == np.float64
assert len(x) == 1

print(x)
print(repr(x))

x = Variable(3.)
print(x)
print(repr(x))

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
assert C.inputs[0] == b
assert b.creator == B
assert B.inputs[0] == a
assert a.creator == A
assert A.inputs[0] == x

y = square(exp(square(x)))
y.backward()
assert np.isclose(x.grad.data, 3.2974425)

x = Variable(2.)
y = Variable(3.)
z = add(square(x), square(y))
z.backward()
assert np.isclose(z.data, 13., atol=1e-12)
assert np.isclose(x.grad.data, 4., atol=1e-12)
assert np.isclose(y.grad.data, 6., atol=1e-12)

x = Variable(3.)
y = add(x, x)
y.backward(retain_grad=True)
assert y.grad.data == 1.
assert np.isclose(x.grad.data, 2., 1e-12)

x.cleargrad()
y = add(add(x, x), x)
y.backward(retain_grad=True)
assert y.grad.data == 1.
assert np.isclose(x.grad.data, 3., atol=1e-12)

x = Variable(2.)
a = square(x)
y = add(square(a), square(a))
y.backward(retain_grad=True)
assert y.grad.data == 1
assert np.isclose(y.data, 32., atol=1e-12)
assert np.isclose(x.grad.data, 64., atol=1e-12)

gc.collect()
for i in range(10):
    print(f'collected {gc.collect()} objects')
    x = Variable(np.random.randn(10000))
    y = square(square(square(x)))

x0 = Variable(1.)
x1 = Variable(1.)
t = add(x0, x1)
y = add(x0, t)
y.backward(retain_grad=True)
assert y.grad.data == 1.
assert np.isclose(t.grad.data, 1., atol=1e-12)
assert np.isclose(x0.grad.data, 2., atol=1e-12)
assert np.isclose(x1.grad.data, 1., atol=1e-12)

x0 = Variable(1.)
x1 = Variable(1.)
t = add(x0, x1)
y = add(x0, t)
y.backward(retain_grad=False)
assert y.grad is None
assert t.grad is None
assert np.isclose(x0.grad.data, 2., atol=1e-12)
assert np.isclose(x1.grad.data, 1., atol=1e-12)

Config.enable_backprob = True
x = Variable(np.ones((100, 100, 100)))
y = square(square(square(x)))
y.backward()

Config.enable_backprob = False
x = Variable(np.ones((100, 100, 100)))
f = Square()
y = square(f(square(x)))
assert y.creator is None
assert f.inputs is None
assert f.outputs is None

Config.enable_backprob = True

a = Variable(3.)
b = Variable(2.)
c = Variable(1.)
y = add(mul(a, b), c)
y.backward()

assert np.isclose(y.data, 7., atol=1e-12)
assert np.isclose(a.grad.data, 2., atol=1e-12)
assert np.isclose(b.grad.data, 3., atol=1e-12)
assert np.isclose(c.grad.data, 1., atol=1e-12)

a = Variable(3.)
b = Variable(2.)
c = Variable(1.)
y = a*b + c
y.backward()

assert np.isclose(y.data, 7., atol=1e-12)
assert np.isclose(a.grad.data, 2., atol=1e-12)
assert np.isclose(b.grad.data, 3., atol=1e-12)
assert np.isclose(c.grad.data, 1., atol=1e-12)

a = Variable(1.)
b = Variable(3.)
y = a/b
y.backward()

assert np.isclose(b.grad.data, -1/9, atol=1e-12)

a = Variable(2.)
b = Variable(3.)
c = Variable(4.)
y = a - b*c
y.backward()

assert a.grad.data == 1.
assert np.isclose(b.grad.data, -4., atol=1e-12)
assert np.isclose(c.grad.data, -3., atol=1e-12)

x = Variable(2.)
y = x + np.array(3.)
assert np.isclose(y.data, 5., atol=1e-12)

x = Variable(2.)
y = x + 3.
assert np.isclose(y.data, 5., atol=1e-12)

y = 3. + x
assert np.isclose(y.data, 5., atol=1e-12)

y = 3. - x
assert np.isclose(y.data, 1., atol=1e-12)

y = 3. * x
assert np.isclose(y.data, 6., atol=1e-12)

y = 3. / x
assert np.isclose(y.data, 3/2, atol=1e-12)

x = Variable([1.])
y = np.array([3.]) / x
assert np.isclose(y.data, 3., atol=1e-12)

x = Variable(2.)
y = -x
assert y.data == -2.

x = Variable(2.)
y = x**3
assert np.isclose(y.data, 8., atol=1e-12)

a = Variable(1.)
b = square(a)
c = square(b)
d = square(b)
e = c + d
f = c + e
g = square(f)
assert np.isclose(g.data, 9., atol=1e-12)
g.backward()
assert np.isclose(a.grad.data, 72., atol=1e-12)

x = Variable(2.)
a = square(x)
b = square(a)
c = square(a)
d = add(b, c)
e = square(a)
y = add(e, d)
y.backward()
assert np.isclose(y.data, 48., atol=1e-12)
assert np.isclose(x.grad.data, 96., atol=1e-12)

x = Variable(3)
y = x + x
y.backward(retain_grad=True)

assert np.isclose(y.grad.data, 1, atol=1e-12)
assert np.isclose(x.grad.data, 2, atol=1e-12)


def f(x):
    y = x**4 - 2 * x ** 2
    return y


x = Variable(2)
y = f(x)
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()
gx.backward()
gx2 = x.grad

assert np.isclose(gx.data, 24, atol=1e-12)
assert np.isclose(gx2.data, 44, atol=1e-12)

x = Variable(2.)
iters = 10

for i in range(iters):
    print(i, x)

    y: Variable = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data

assert np.isclose(x.data, 1, atol=1e-12)

x = Variable(2)
y = x**2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
assert np.isclose(x.grad.data, 100, atol=1e-12)

x = Variable([[1, 2, 3], [4, 5, 6]])
y = functions.reshape(x, (6,))
y.backward(retain_grad=True)

assert y.data.shape == y.grad.shape
assert x.data.shape == x.grad.shape

x = Variable(np.random.randn(1, 2, 3))
y1 = x.reshape((2, 3))
y2 = x.reshape(2, 3)

assert y1.shape == y2.shape

x = Variable([[1, 2, 3], [4, 5, 6]])
y = functions.transpose(x)
y.backward(retain_grad=True)

assert np.all(y.data == np.array([[1, 4], [2, 5], [3, 6]]))
assert np.all(y.grad.data == np.array([[1, 1], [1, 1], [1, 1]]))
assert np.all(x.grad.data == np.array([[1, 1, 1], [1, 1, 1]]))

x = Variable(np.ones((1, 2, 3)))
y = functions.transpose(x, (1, 0, 2))
y.backward(retain_grad=True)

assert y.shape == (2, 1, 3)
assert y.data.shape == y.grad.shape
assert x.data.shape == x.grad.shape

x = Variable(np.ones((2, 3, 4, 5)))
y = functions.transpose(x)
y.backward(retain_grad=True)

assert y.shape == (5, 4, 3, 2)
assert y.data.shape == y.grad.shape
assert x.data.shape == x.grad.shape
