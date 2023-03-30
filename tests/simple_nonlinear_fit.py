import numpy as np
import matplotlib.pylab as plt

from import_dezero import *

x = np.linspace(0, 1, 100).reshape((100, 1))
y = np.sin(2*np.pi * x) + np.random.rand(100, 1)
x, y = dz.Variable(x), dz.Variable(y)

i, h, o = 1, 10, 1
l1 = layers.Linear(h)
l2 = layers.Linear(o)


def predict(x):
    y = l1(x)
    y = functions.sigmoid(y)
    return l2(y)


lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = predict(x)
    loss: dz.Variable = functions.mse(y, y_pred)

    l1.cleargrads()
    l2.cleargrads()
    loss.backward()

    for l in [l1, l2]:
        for p in l.params():
            p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(f'loss={loss.data}')

y_pred = predict(x)

plt.scatter(x.data, y.data)
plt.plot(x.data, y_pred.data, 'r')
plt.show()
