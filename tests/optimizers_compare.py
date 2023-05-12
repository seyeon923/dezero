import matplotlib.pyplot as plt

from import_dezero import *


def train(model, optimizer, data_loader):
    losses = []
    accs = []
    for epoch in range(max_epoch):
        sum_loss = 0
        sum_acc = 0
        ds_size = 0

        for x, t in data_loader:
            y = model(x)
            loss = functions.softmax_cross_entropy_simple(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            batch_size = len(t)

            sum_loss += float(loss.data) * batch_size
            sum_acc += float(metrics.accuracy(y, t).data) * batch_size

            ds_size += batch_size

        loss = sum_loss / ds_size
        acc = sum_acc / ds_size

        losses.append(loss)
        accs.append(acc)

        print(
            f'epoch {epoch + 1:03d}: train loss={loss:.2f}, train_acc={acc:.2f}')

    return losses, accs


# hyper parameters
max_epoch = 300
batch_size = 30
hidden_size = (10, 10)
lr = 0.01

ds = datasets.Spiral(train=True)
data_loader = datasets.DataLoader(ds, batch_size)

# SGD
model = models.MLP(hidden_size + (3,))
sgd = optimizers.SGD(lr).setup(model)
sgd_losses, sgd_accs = train(model, sgd, data_loader)

# Momentum SGD
model = models.MLP(hidden_size + (3,))
momentum = optimizers.MomentumSGD(lr).setup(model)
momentum_losses, momentum_accs = train(model, momentum, data_loader)

# RMSprop
model = models.MLP(hidden_size + (3,))
rmsprop = optimizers.RMSprop(lr).setup(model)
rmsprop_losses, rmsprop_accs = train(model, rmsprop, data_loader)

# Adam
model = models.MLP(hidden_size + (3,))
adam = optimizers.Adam(lr).setup(model)
adam_losses, adam_accs = train(model, adam, data_loader)

plt.subplot(1, 2, 1)
plt.title('Loss per Epoch')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(range(1, 1+max_epoch), sgd_losses, label='SGD')
plt.plot(range(1, 1+max_epoch), momentum_losses, label='Momentum SGD')
plt.plot(range(1, 1+max_epoch), rmsprop_losses, label='RMSprop')
plt.plot(range(1, 1+max_epoch), adam_losses, label='Adam')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('Accuracy per Epoch')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.plot(range(1, 1+max_epoch), sgd_accs, label='SGD')
plt.plot(range(1, 1+max_epoch), momentum_accs, label='Momentum SGD')
plt.plot(range(1, 1+max_epoch), rmsprop_accs, label='RMSprop')
plt.plot(range(1, 1+max_epoch), adam_accs, label='Adam')
plt.legend()

plt.show()
