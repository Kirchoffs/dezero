if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


from dezero import no_grad
from dezero.datasets import MNIST
from dezero.data_loaders import DataLoader
from dezero.models import MLP
from dezero.optimizers import SGD
from dezero.functions import softmax_cross_entropy, accuracy
import matplotlib.pyplot as plt


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

        train_loss_sum += loss.data * len(t)
        train_acc_sum += acc.data * len(t)
    train_loss_list.append(train_loss_sum / len(mnist_train_dataset))
    train_acc_list.append(train_acc_sum / len(mnist_train_dataset))

    test_loss_sum = 0
    test_acc_sum = 0
    with no_grad():
        for x, t in mnist_test_data_loader:
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            acc = accuracy(y, t)

            test_loss_sum += loss.data * len(t)
            test_acc_sum += acc.data * len(t)
    test_loss_list.append(test_loss_sum / len(mnist_test_dataset))
    test_acc_list.append(test_acc_sum / len(mnist_test_dataset))


fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = 'col')

axes[0, 0].plot(range(max_epochs), train_loss_list, color = 'blue', label = 'Train Loss')
axes[0, 0].set_title('Loss Comparison', fontsize = 14)
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].grid(True, linestyle = ':', alpha = 0.6)
axes[0, 0].legend(loc = 'upper right')

axes[1, 0].plot(range(max_epochs), test_loss_list, color = 'red', label='Test Loss')
axes[1, 0].set_ylabel('Test Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].grid(True, linestyle = ':', alpha = 0.6)
axes[1, 0].legend(loc = 'upper right')

axes[0, 1].plot(range(max_epochs), train_acc_list, color = 'green', label = 'Train Acc')
axes[0, 1].set_title('Accuracy Comparison', fontsize = 14)
axes[0, 1].set_ylabel('Training Accuracy')
axes[0, 1].grid(True, linestyle = ':', alpha = 0.6)
axes[0, 1].legend(loc = 'lower right')

axes[1, 1].plot(range(max_epochs), test_acc_list, color = 'orange', label = 'Test Acc')
axes[1, 1].set_ylabel('Test Accuracy')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].grid(True, linestyle = ':', alpha = 0.6)
axes[1, 1].legend(loc = 'lower right')

plt.tight_layout()
plt.subplots_adjust(hspace = 0.1) 
plt.show()
