import numpy as np
import matplotlib.pyplot as plt

from import_dezero import *

np.random.seed(0)
x = np.random.randn(100, 1)
y = 2*x + 5 + np.random.randn(100, 1)
x, y = Variable(x), Variable(y)

w = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    return functions.matmul_add(x, w, b)


lr = 0.1
iters = 100

for i in range(iters):
    y_pred = predict(x)
    loss = functions.mse(y, y_pred)

    w.cleargrad()
    b.cleargrad()
    loss.backward()

    w.data -= lr * w.grad.data
    b.data -= lr * b.grad.data

    print(f'w={w.data}, b={b.data}, loss={loss.data}')

y_pred = predict(x)
plt.scatter(x.data, y.data)
plt.plot(x.data, y_pred.data, 'r')
plt.show()
