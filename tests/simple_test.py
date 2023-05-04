import gc
import sys
import os

import numpy as np

from import_dezero import *

x = dz.Variable(np.array(1.))
x = dz.Variable(None)

dz.Config.enable_backprob = False

with dz.enable_backprob():
    assert dz.Config.enable_backprob == True
assert dz.Config.enable_backprob == False

dz.Config.enable_backprob = True
with dz.disable_backprob():
    assert dz.Config.enable_backprob == False
assert dz.Config.enable_backprob == True

x = dz.Variable(1.)
assert x.shape == ()
assert x.ndim == 0
assert x.size == 1
assert np.issubdtype(x.dtype, np.floating)

x = dz.Variable([1., 2.])
assert x.shape == (2,)
assert x.ndim == 1
assert x.size == 2
assert np.issubdtype(x.dtype, np.floating)
assert len(x) == 2

x = dz.Variable([[1, 2], [3, 4], [5, 6]])
assert x.shape == (3, 2)
assert x.ndim == 2
assert x.size == 6
assert np.issubdtype(x.dtype, np.integer)
assert len(x) == 3

x = dz.Variable(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9],
                         dtype=np.float64).reshape((1, 3, 3)), name='x')
assert x.shape == (1, 3, 3)
assert x.ndim == 3
assert x.size == 9
assert x.dtype == np.float64
assert len(x) == 1

print(x)
print(repr(x))

x = dz.Variable(3.)
print(x)
print(repr(x))

assert np.isclose(functions.add(dz.Variable(2.), dz.Variable(3.)).data, 5.)

assert functions.square(dz.Variable(np.array(10))).data == 100
assert np.isclose(functions.exp(dz.Variable(np.array(1))).data, np.exp(1))

x = dz.Variable(np.array(0.5))
y = functions.square(functions.exp(functions.square(x)))
assert np.isclose(y.data, np.exp(0.25) * np.exp(0.25))

A = functions.Square()
B = functions.Exp()
C = functions.Square()
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert C.inputs[0] == b
assert b.creator == B
assert B.inputs[0] == a
assert a.creator == A
assert A.inputs[0] == x

y = functions.square(functions.exp(functions.square(x)))
y.backward()
assert np.isclose(x.grad.data, 3.2974425)

x = dz.Variable(2.)
y = dz.Variable(3.)
z = functions.add(functions.square(x), functions.square(y))
z.backward()
assert np.isclose(z.data, 13., atol=1e-12)
assert np.isclose(x.grad.data, 4., atol=1e-12)
assert np.isclose(y.grad.data, 6., atol=1e-12)

x = dz.Variable(3.)
y = functions.add(x, x)
y.backward(retain_grad=True)
assert y.grad.data == 1.
assert np.isclose(x.grad.data, 2., 1e-12)

x.cleargrad()
y = functions.add(functions.add(x, x), x)
y.backward(retain_grad=True)
assert y.grad.data == 1.
assert np.isclose(x.grad.data, 3., atol=1e-12)

x = dz.Variable(2.)
a = functions.square(x)
y = functions.add(functions.square(a), functions.square(a))
y.backward(retain_grad=True)
assert y.grad.data == 1
assert np.isclose(y.data, 32., atol=1e-12)
assert np.isclose(x.grad.data, 64., atol=1e-12)

gc.collect()
for i in range(10):
    print(f'collected {gc.collect()} objects')
    x = dz.Variable(np.random.randn(10000))
    y = functions.square(functions.square(functions.square(x)))

x0 = dz.Variable(1.)
x1 = dz.Variable(1.)
t = functions.add(x0, x1)
y = functions.add(x0, t)
y.backward(retain_grad=True)
assert y.grad.data == 1.
assert np.isclose(t.grad.data, 1., atol=1e-12)
assert np.isclose(x0.grad.data, 2., atol=1e-12)
assert np.isclose(x1.grad.data, 1., atol=1e-12)

x0 = dz.Variable(1.)
x1 = dz.Variable(1.)
t = functions.add(x0, x1)
y = functions.add(x0, t)
y.backward(retain_grad=False)
assert y.grad is None
assert t.grad is None
assert np.isclose(x0.grad.data, 2., atol=1e-12)
assert np.isclose(x1.grad.data, 1., atol=1e-12)

dz.Config.enable_backprob = True
x = dz.Variable(np.ones((100, 100, 100)))
y = functions.square(functions.square(functions.square(x)))
y.backward()

dz.Config.enable_backprob = False
x = dz.Variable(np.ones((100, 100, 100)))
f = functions.Square()
y = functions.square(f(functions.square(x)))
assert y.creator is None
assert f.inputs is None
assert f.outputs is None

dz.Config.enable_backprob = True

a = dz.Variable(3.)
b = dz.Variable(2.)
c = dz.Variable(1.)
y = functions.add(functions.mul(a, b), c)
y.backward()

assert np.isclose(y.data, 7., atol=1e-12)
assert np.isclose(a.grad.data, 2., atol=1e-12)
assert np.isclose(b.grad.data, 3., atol=1e-12)
assert np.isclose(c.grad.data, 1., atol=1e-12)

a = dz.Variable(3.)
b = dz.Variable(2.)
c = dz.Variable(1.)
y = a*b + c
y.backward()

assert np.isclose(y.data, 7., atol=1e-12)
assert np.isclose(a.grad.data, 2., atol=1e-12)
assert np.isclose(b.grad.data, 3., atol=1e-12)
assert np.isclose(c.grad.data, 1., atol=1e-12)

a = dz.Variable(1.)
b = dz.Variable(3.)
y = a/b
y.backward()

assert np.isclose(b.grad.data, -1/9, atol=1e-12)

a = dz.Variable(2.)
b = dz.Variable(3.)
c = dz.Variable(4.)
y = a - b*c
y.backward()

assert a.grad.data == 1.
assert np.isclose(b.grad.data, -4., atol=1e-12)
assert np.isclose(c.grad.data, -3., atol=1e-12)

x = dz.Variable(2.)
y = x + np.array(3.)
assert np.isclose(y.data, 5., atol=1e-12)

x = dz.Variable(2.)
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

x = dz.Variable([1.])
y = np.array([3.]) / x
assert np.isclose(y.data, 3., atol=1e-12)

x = dz.Variable(2.)
y = -x
assert y.data == -2.

x = dz.Variable(2.)
y = x**3
assert np.isclose(y.data, 8., atol=1e-12)

a = dz.Variable(1.)
b = functions.square(a)
c = functions.square(b)
d = functions.square(b)
e = c + d
f = c + e
g = functions.square(f)
assert np.isclose(g.data, 9., atol=1e-12)
g.backward()
assert np.isclose(a.grad.data, 72., atol=1e-12)

x = dz.Variable(2.)
a = functions.square(x)
b = functions.square(a)
c = functions.square(a)
d = functions.add(b, c)
e = functions.square(a)
y = functions.add(e, d)
y.backward()
assert np.isclose(y.data, 48., atol=1e-12)
assert np.isclose(x.grad.data, 96., atol=1e-12)

x = dz.Variable(3)
y = x + x
y.backward(retain_grad=True)

assert np.isclose(y.grad.data, 1, atol=1e-12)
assert np.isclose(x.grad.data, 2, atol=1e-12)


def f(x):
    y = x**4 - 2 * x ** 2
    return y


x = dz.Variable(2)
y = f(x)
y.backward(create_graph=True)

gx = x.grad
x.cleargrad()
gx.backward()
gx2 = x.grad

assert np.isclose(gx.data, 24, atol=1e-12)
assert np.isclose(gx2.data, 44, atol=1e-12)

x = dz.Variable(2.)
iters = 10

for i in range(iters):
    print(i, x)

    y: dz.Variable = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data

assert np.isclose(x.data, 1, atol=1e-12)

x = dz.Variable(2)
y = x**2
y.backward(create_graph=True)
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
assert np.isclose(x.grad.data, 100, atol=1e-12)

x = dz.Variable([[1, 2, 3], [4, 5, 6]])
y = functions.reshape(x, (6,))
y.backward(retain_grad=True)

assert y.data.shape == y.grad.shape
assert x.data.shape == x.grad.shape

x = dz.Variable(np.random.randn(1, 2, 3))
y1 = x.reshape((2, 3))
y2 = x.reshape(2, 3)

assert y1.shape == y2.shape

x = dz.Variable([[1, 2, 3], [4, 5, 6]])
y = functions.transpose(x)
y.backward(retain_grad=True)

assert np.all(y.data == np.array([[1, 4], [2, 5], [3, 6]]))
assert np.all(y.grad.data == np.array([[1, 1], [1, 1], [1, 1]]))
assert np.all(x.grad.data == np.array([[1, 1, 1], [1, 1, 1]]))

x = dz.Variable(np.ones((1, 2, 3)))
y = functions.transpose(x, (1, 0, 2))
y.backward(retain_grad=True)

assert y.shape == (2, 1, 3)
assert y.data.shape == y.grad.shape
assert x.data.shape == x.grad.shape

x = dz.Variable(np.ones((2, 3, 4, 5)))
y = functions.transpose(x)
y.backward(retain_grad=True)

assert y.shape == (5, 4, 3, 2)
assert y.data.shape == y.grad.shape
assert x.data.shape == x.grad.shape

x0 = dz.Variable([1, 2, 3])
x1 = dz.Variable(10)
y = x0 + x1
y.backward(retain_grad=True)


assert np.all(y.data == np.array([11, 12, 13]))
assert np.all(y.grad.data == np.array([1, 1, 1]))
assert np.all(x0.grad.data == np.array([1, 1, 1], dtype=x0.data.dtype))
assert np.all(x1.grad.data == 3)

x = dz.Variable(np.random.randn(2, 3))
w = dz.Variable(np.random.randn(3, 4))
y = functions.matmul(x, w)
y.backward()

assert x.grad.shape == x.data.shape
assert w.grad.shape == w.data.shape

x = dz.Variable([[1, 2, 3], [4, 5, 6]])
y = functions.sum(x, axis=1, keepdims=False)
assert y.shape == (2,)
y.backward()
assert x.shape == x.grad.shape
assert np.all(x.grad.data == np.array([[1, 1, 1], [1, 1, 1]]))

model = dz.models.MLP((10, 2))
x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([0, 1, 0, 1])
y = model(x)
loss = dz.functions.softmax_cross_entropy_simple(y, t)
print(loss)
