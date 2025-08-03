if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import matplotlib.pyplot as plt
from dezero import no_grad
from dezero.datasets import Dataset
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


hidden_layer_size = 10
num_classes = 3
model = M.MLP([hidden_layer_size, num_classes])
lr = 0.1
optimizer = O.SGD(lr).setup(model)

triple_spiral_dataset_train = TripleSpiralDataset(train = True)

max_epochs = 300
batch_size = 30
data_size = len(triple_spiral_dataset_train)
loss_sum_list = []
for epoch in range(max_epochs):
    indices = np.random.permutation(data_size)
    
    loss_sum = 0
    for i in range(0, data_size, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch = [triple_spiral_dataset_train[idx] for idx in batch_indices]

        X_batch = np.array([sample[0] for sample in batch])
        y_batch = np.array([sample[1] for sample in batch], dtype = 'int')
        
        y_pred = model(X_batch)
        loss = F.softmax_cross_entropy(y_pred, y_batch)
        
        model.clear_grad()
        loss.backward()
        
        optimizer.update()
    
        loss_sum += loss.data * len(X_batch)
    loss_sum_list.append(loss_sum)

plt.figure(figsize = (12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(len(loss_sum_list)), loss_sum_list)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)

triple_spiral_dataset_test = TripleSpiralDataset(train = False)
features_test, labels_test = triple_spiral_dataset_test.data, triple_spiral_dataset_test.labels

x_min, x_max = features_test[:, 0].min() - 0.1, features_test[:, 0].max() + 0.1
y_min, y_max = features_test[:, 1].min() - 0.1, features_test[:, 1].max() + 0.1
xs, ys = np.meshgrid(
    np.arange(x_min, x_max, 0.01),
    np.arange(y_min, y_max, 0.01)
)

grid_points = np.c_[xs.ravel(), ys.ravel()]
with no_grad():
    score = model(grid_points)
    predict_cls = np.argmax(score.data, axis = 1)
    Z = predict_cls.reshape(xs.shape)

plt.contourf(xs, ys, Z, alpha = 0.3, cmap = plt.cm.Spectral)

markers_candidates = ['o', '^', 's', 'D', 'v', 'p']
colors_candidates = ['darkblue', 'black', 'gray', 'red', 'green', 'purple']

for i in range(num_classes):
    plt.scatter(
        features_test[labels_test == i, 0], 
        features_test[labels_test == i, 1], 
        marker = markers_candidates[i], 
        c = colors_candidates[i], 
        s = 30, 
        edgecolors = 'k',
        label = f'Class {i}'
    )

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title('Model Decision Boundary vs Original Data')

plt.legend()
plt.suptitle('Multiclass Classification with MLP on Spiral Data', fontsize = 16)
plt.tight_layout()
plt.show()
