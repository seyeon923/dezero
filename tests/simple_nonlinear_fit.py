import numpy as np
import matplotlib.pylab as plt

from import_dezero import *

x = np.linspace(0, 1, 100).reshape((100, 1))
y = np.sin(2*np.pi * x) + np.random.rand(100, 1)
x, y = dz.Variable(x), dz.Variable(y)


class TwoLayerModel(dz.Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()

        self.l1 = layers.Linear(hidden_size)
        self.l2 = layers.Linear(out_size)

    def forward(self, x):
        y = self.l1(x)
        y = functions.sigmoid(y)
        return self.l2(y)


model = TwoLayerModel(10, 1)
model.plot(x)

lr = 0.2
iters = 10000

for i in range(iters):
    y_pred = model(x)
    loss: dz.Variable = functions.mse(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(f'loss={loss.data}')

y_pred = model(x)

plt.scatter(x.data, y.data)
plt.plot(x.data, y_pred.data, 'r')
plt.show()
