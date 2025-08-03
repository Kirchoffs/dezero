if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import matplotlib.pyplot as plt
from dezero import no_grad
from dezero.datasets import Dataset
from dezero.data_loaders import DataLoader
import dezero.functions as F
import dezero.models as M
import dezero.optimizers as O


def generate_spiral_data(train = True, samples_per_class = 100, num_classes = 3):
    seed = 42 if train else 6174
    np.random.seed(seed)

    features = np.zeros((samples_per_class * num_classes, 2))
    labels = np.zeros(samples_per_class * num_classes, dtype = 'uint8')
    
    angle_per_class = 2 * np.pi / num_classes
    
    for j in range(num_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)
        
        start_angle = j * angle_per_class
        end_angle = (j + 1) * angle_per_class
        
        overlap_factor = 0.2 
        expanded_start = start_angle - overlap_factor * angle_per_class
        expanded_end = end_angle + overlap_factor * angle_per_class
        
        t = np.linspace(expanded_start, expanded_end, samples_per_class) + \
            np.random.randn(samples_per_class) * 0.2
        
        features[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        labels[ix] = j
        
    return features, labels


class TripleSpiralDataset(Dataset):
    def prepare(self):
        self.data, self.labels = generate_spiral_data(train = True, samples_per_class = 100, num_classes = 3)


train_dataset = TripleSpiralDataset(train = True)
test_dataset = TripleSpiralDataset(train = False)
train_data_loader = DataLoader(train_dataset, batch_size = 30, shuffle = True)
test_data_loader = DataLoader(test_dataset, batch_size = 30, shuffle = False)

hidden_layer_size = 10
num_classes = 3
model = M.MLP([hidden_layer_size, num_classes])
lr = 0.1
optimizer = O.SGD(lr).setup(model)

max_epochs = 300
loss_sum_list = []
acc_sum_list = []
test_loss_sum_list = []
test_acc_sum_list = []
for epoch in range(max_epochs):    
    loss_sum = 0
    acc_sum = 0
    for x, t in train_data_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        model.clear_grad()
        loss.backward()
        optimizer.update()

        loss_sum += loss.data * len(t)
        acc_sum += acc.data * len(t)
    loss_sum_list.append(loss_sum)
    acc_sum_list.append(acc_sum)

    test_loss_sum = 0
    test_acc_sum = 0
    with no_grad():
        for x, t in test_data_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            test_loss_sum += loss.data * len(t)
            test_acc_sum += acc.data * len(t)
    test_loss_sum_list.append(test_loss_sum)
    test_acc_sum_list.append(test_acc_sum)


train_count = len(train_dataset)
test_count = len(test_dataset)
train_loss = [x / train_count for x in loss_sum_list]
train_acc = [x / train_count for x in acc_sum_list]
test_loss = [x / test_count for x in test_loss_sum_list]
test_acc = [x / test_count for x in test_acc_sum_list]
epochs = range(len(train_loss))

fig, axes = plt.subplots(2, 2, figsize = (12, 8), sharex = 'col')

axes[0, 0].plot(epochs, train_loss, color = 'blue', label = 'Train Loss')
axes[0, 0].set_title('Loss Comparison', fontsize = 14)
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].grid(True, linestyle = ':', alpha = 0.6)
axes[0, 0].legend(loc = 'upper right')

axes[1, 0].plot(epochs, test_loss, color = 'red', label='Test Loss')
axes[1, 0].set_ylabel('Test Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].grid(True, linestyle = ':', alpha = 0.6)
axes[1, 0].legend(loc = 'upper right')

axes[0, 1].plot(epochs, train_acc, color = 'green', label = 'Train Acc')
axes[0, 1].set_title('Accuracy Comparison', fontsize = 14)
axes[0, 1].set_ylabel('Training Accuracy')
axes[0, 1].grid(True, linestyle = ':', alpha = 0.6)
axes[0, 1].legend(loc = 'lower right')

axes[1, 1].plot(epochs, test_acc, color = 'orange', label = 'Test Acc')
axes[1, 1].set_ylabel('Test Accuracy')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].grid(True, linestyle = ':', alpha = 0.6)
axes[1, 1].legend(loc = 'lower right')

plt.tight_layout()
plt.subplots_adjust(hspace = 0.1) 
plt.show()
