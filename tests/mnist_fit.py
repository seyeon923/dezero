from import_dezero import *

max_epoch = 5
batch_size = 100
hidden_size = (1000, 1000)

train_set = datasets.MNIST(train=True)
test_set = datasets.MNIST(train=False)

train_size = len(train_set)
test_size = len(test_set)

train_loader = datasets.DataLoader(train_set, batch_size)
test_loader = datasets.DataLoader(test_set, batch_size, shuffle=False)

model = models.MLP(hidden_size + (10,), activation=functions.relu)
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)

        loss = functions.softmax_cross_entropy_simple(y, t)
        acc = metrics.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    train_loss = sum_loss / train_size
    train_acc = sum_acc / train_size

    sum_loss, sum_acc = 0, 0
    for x, t in test_loader:
        with dz.disable_backprob():
            y = model(x)
            loss = functions.softmax_cross_entropy_simple(y, t)
            acc = metrics.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    test_loss = sum_loss / test_size
    test_acc = sum_acc / test_size

    print(f'Epoch {epoch + 1}: train_loss={train_loss}, test_loss={test_loss}, train_acc={train_acc}, test_acc={test_acc}')
