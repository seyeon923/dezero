import numpy as np
import matplotlib.pylab as plt

from import_dezero import *

x = np.linspace(0, 1, 100).reshape((100, 1))
y = np.sin(2*np.pi * x) + np.random.rand(100, 1)
x, y = dz.Variable(x), dz.Variable(y)


model = models.MLP((10, 1))
model.plot(x)

lr = 0.2
iters = 10000

optimizer = optimizers.MomentumSGD(lr).setup(model)

for i in range(iters):
    y_pred = model(x)
    loss: dz.Variable = functions.mse(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(f'loss={loss.data}')

y_pred = model(x)

plt.scatter(x.data, y.data)
plt.plot(x.data, y_pred.data, 'r')
plt.show()
