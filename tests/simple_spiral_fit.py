import math

import matplotlib.pyplot as plt
import numpy as np

from import_dezero import *

# hyper parameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = datasets.get_spiral(train=True)

model = models.MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

losses = []
for epoch in range(max_epoch):
    idx = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_idx = idx[i*batch_size: (i+1)*batch_size]
        batch_x = x[batch_idx]
        batch_t = t[batch_idx]

        y = model(batch_x)
        loss = functions.softmax_cross_entropy_simple(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    losses.append(avg_loss)
    print(f'epoch {epoch + 1}, loss {avg_loss:.2f}')

plt.subplot(1, 2, 1)
plt.title('Loss per Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(1, 1+max_epoch), losses)

plt.subplot(1, 2, 2)
x_min = np.min(x[:, 0])
x_max = np.max(x[:, 0])
y_min = np.min(x[:, 1])
y_max = np.max(x[:, 1])
linspace = np.linspace(x_min, x_max, 100)
xv, yv = np.meshgrid(linspace, linspace)
inputs = np.stack([xv.flatten(), yv.flatten()], axis=1)
estimates = np.argmax(model(inputs).data, axis=1)
plt.contourf(xv, yv, estimates.reshape(xv.shape))
markers = ['.', '+', 'x']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    mask = t == i
    plt.scatter(x[mask, 0], x[mask, 1], c=colors[i], marker=markers[i])
plt.show()
