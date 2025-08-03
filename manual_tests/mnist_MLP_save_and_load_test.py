import os
import sys
from dezero import no_grad
from dezero import cuda
from dezero.datasets import MNIST
from dezero.data_loaders import DataLoader
from dezero.models import MLP
from dezero.optimizers import SGD
from dezero.functions import softmax_cross_entropy, accuracy
import matplotlib.pyplot as plt


if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


max_epochs = 5
batch_size = 100
hidden_layer_size = 1000
output_layer_size = 10

mnist_train_dataset = MNIST(train = True)
mnist_test_dataset = MNIST(train = False)
mnist_train_data_loader = DataLoader(mnist_train_dataset, batch_size, shuffle = True)
mnist_test_data_loader = DataLoader(mnist_test_dataset, batch_size, shuffle = False)

model = MLP([hidden_layer_size, output_layer_size])
optimizer = SGD().setup(model)

if cuda.gpu_enable:
    print("Cuda Enabled!")
    mnist_train_data_loader.to_gpu()
    mnist_test_data_loader.to_gpu()
    model.to_gpu()
    
train_loss_list = []
train_acc_list = []
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

model.save_weights('mnist_DNN_save_and_load_test.npz')

model = MLP([hidden_layer_size, output_layer_size])
if cuda.gpu_enable:
    model.to_gpu()
model.load_weights('mnist_DNN_save_and_load_test.npz')

with no_grad():
    loss_sum = 0
    acc_sum = 0
    for x, t in mnist_test_data_loader:
        y = model(x)
        loss = softmax_cross_entropy(y, t)
        acc = accuracy(y, t)
        loss_sum += float(loss.data) * len(t)
        acc_sum += float(acc.data) * len(t)
    loss_sum /= len(mnist_test_dataset)
    acc_sum /= len(mnist_test_dataset)
    print(loss_sum)
    print(acc_sum)
