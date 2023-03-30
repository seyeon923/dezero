import numpy as np
import matplotlib.pylab as plt

from import_dezero import *

x = np.linspace(0, 1, 100).reshape((100, 1))
y = np.sin(2*np.pi * x) + np.random.rand(100, 1)
x, y = Variable(x), Variable(y)

i, h, o = 1, 10, 1
w1 = Variable(0.01 * np.random.randn(i, h))
b1 = Variable(np.zeros(h))
w2 = Variable(0.01 * np.random.randn(h, o))
b2 = Variable(np.zeros(o))


def predict(x):
    y = functions.matmul_add(x, w1, b1)
    y = functions.sigmoid(y)
    return functions.matmul_add(y, w2, b2)


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss = functions.mse(y, y_pred)

    w1.cleargrad()
    b1.cleargrad()
    w2.cleargrad()
    b2.cleargrad()
    loss.backward()

    w1.data -= lr * w1.grad.data
    b1.data -= lr * b1.grad.data
    w2.data -= lr * w2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 1000 == 0:
        print(f'loss={loss.data}')

y_pred = predict(x)

plt.scatter(x.data, y.data)
plt.plot(x.data, y_pred.data, 'r')
plt.show()
