import math

import matplotlib.pyplot as plt
import numpy as np

from import_dezero import *

# hyper parameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 0.01

train_set = datasets.Spiral(train=True)
test_set = datasets.Spiral(train=False)

train_size = len(train_set)
test_size = len(test_set)

train_loader = datasets.DataLoader(train_set, batch_size)
test_loader = datasets.DataLoader(test_set, batch_size, shuffle=False)

model = models.MLP((hidden_size, 3))
optimizer = optimizers.Adam(lr).setup(model)

x_min, x_max, y_min, y_max = 100, -100, 100, -100

train_losses = []
train_accs = []
test_losses = []
test_accs = []
for epoch in range(max_epoch):
    sum_loss = 0
    sum_acc = 0
    for x, t in train_loader:
        y = model(x)
        loss = functions.softmax_cross_entropy_simple(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(metrics.accuracy(y, t).data) * len(t)

        local_x_min = np.min(x)
        local_x_max = np.max(x)
        local_y_min = min(np.min(y.data), np.min(t))
        local_y_max = max(np.max(y.data), np.max(t))

        if local_x_min < x_min:
            x_min = local_x_min
        if local_x_max > x_max:
            x_max = local_x_max
        if local_y_min < y_min:
            y_min = local_y_min
        if local_y_max > y_max:
            y_max = local_y_max

    train_loss = sum_loss / train_size
    train_acc = sum_acc / train_size

    sum_loss = 0
    sum_acc = 0
    for x, t in test_loader:
        with dz.disable_backprob():
            y = model(x)
            loss = functions.softmax_cross_entropy_simple(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(metrics.accuracy(y, t).data) * len(t)

        local_x_min = np.min(x)
        local_x_max = np.max(x)
        local_y_min = min(np.min(y.data), np.min(t))
        local_y_max = max(np.max(y.data), np.max(t))

        if local_x_min < x_min:
            x_min = local_x_min
        if local_x_max > x_max:
            x_max = local_x_max
        if local_y_min < y_min:
            y_min = local_y_min
        if local_y_max > y_max:
            y_max = local_y_max

    test_loss = sum_loss / test_size
    test_acc = sum_acc / test_size

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(
        f'epoch {epoch + 1:03d}: train loss={train_loss:.2f}, train_acc={train_acc:.2f}, test loss={test_loss:.2f}, test_acc={test_acc:.2f}')

plt.subplot(3, 2, 1)
plt.title('Loss per Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(1, 1+max_epoch), train_losses, label='Train Loss')
plt.plot(range(1, 1+max_epoch), test_losses, label='Test Loss')
plt.legend()

plt.subplot(3, 2, 2)
plt.title('Accuracy per Epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(range(1, 1+max_epoch), train_accs, label='Train Accuracy')
plt.plot(range(1, 1+max_epoch), test_accs, label='Test Accuracy')
plt.legend()

plt.subplot(3, 2, (3, 6))
linspace = np.linspace(x_min, x_max, 300)
xv, yv = np.meshgrid(linspace, linspace)
inputs = np.stack([xv.flatten(), yv.flatten()], axis=1)
estimates = np.argmax(model(inputs).data, axis=1)
plt.contourf(xv, yv, estimates.reshape(xv.shape))
markers = ['.', '+', 'x']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i in range(3):
    for x, t in test_loader:
        mask = t == i
        plt.scatter(x[mask, 0], x[mask, 1], c=colors[i], marker=markers[i])

plt.show()
