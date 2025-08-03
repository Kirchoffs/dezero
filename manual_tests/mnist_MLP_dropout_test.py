if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


from dezero import no_grad, test_mode
from dezero import cuda
from dezero.datasets import MNIST
from dezero.data_loaders import DataLoader
from dezero.models import MLPWithDropout
from dezero.optimizers import SGD
from dezero.functions import softmax_cross_entropy, accuracy


max_epochs = 5
batch_size = 100
hidden_layer_size = 1000
output_layer_size = 10

mnist_train_dataset = MNIST(train = True)
mnist_test_dataset = MNIST(train = False)
mnist_train_data_loader = DataLoader(mnist_train_dataset, batch_size, shuffle = True)
mnist_test_data_loader = DataLoader(mnist_test_dataset, batch_size, shuffle = False)

model = MLPWithDropout([hidden_layer_size, output_layer_size], dropout_ratio = 0.5)
optimizer = SGD().setup(model)

if cuda.gpu_enable:
    print("Cuda Enabled!")
    mnist_train_data_loader.to_gpu()
    mnist_test_data_loader.to_gpu()
    model.to_gpu()

train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []
for epoch in range(max_epochs):
    train_loss_sum = 0
    train_acc_sum = 0
    for x, t in mnist_train_data_loader:
        y = model(x)
        loss = softmax_cross_entropy(y, t)
        acc = accuracy(y, t)

        model.clear_grad()
        loss.backward()
        optimizer.update()

        train_loss_sum += float(loss.data) * len(t)
        train_acc_sum += float(acc.data) * len(t)
    train_loss_list.append(train_loss_sum / len(mnist_train_dataset))
    train_acc_list.append(train_acc_sum / len(mnist_train_dataset))

    test_loss_sum = 0
    test_acc_sum = 0
    with no_grad(), test_mode():
        for x, t in mnist_test_data_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)

            test_loss_sum += float(loss.data) * len(t)
            test_acc_sum += float(acc.data) * len(t)
    test_loss_list.append(test_loss_sum / len(mnist_test_dataset))
    test_acc_list.append(test_acc_sum / len(mnist_test_dataset))

    print(f"epoch {epoch + 1}: train_loss {train_loss_list[-1]:.4f}, train_acc {train_acc_list[-1]:.4f}, test_loss {test_loss_list[-1]:.4f}, test_acc {test_acc_list[-1]:.4f}")
