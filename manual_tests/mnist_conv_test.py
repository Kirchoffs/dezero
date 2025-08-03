if "__file__" in globals():
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
from dezero import Model, no_grad, test_mode, cuda
from dezero.datasets import MNIST
from dezero.data_loaders import DataLoader
import dezero.layers as L
import dezero.functions as F
from dezero.functions_conv import max_pooling
from dezero.optimizers import MomentumSGD
from dezero.transforms import Compose, ToFloat, Normalize


class ConvNet(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = L.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 1, padding = 1, has_bias = True)
        self.conv2 = L.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1, has_bias = True)
        self.fc = L.Linear(out_size = 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = max_pooling(x, 2, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class DeconvNet(Model):
    def __init__(self):
        super().__init__()
        self.conv = L.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 3, stride = 2, padding = 1)
        self.deconv = L.Deconv2dSimple(out_channels = 1, kernel_size = 3, stride = 2, padding = 1, out_size = (28, 28))
        self.fc = L.Linear(out_size = 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.deconv(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def main():
    max_epochs = 4
    batch_size = 128
    
    data_transform = Compose([ToFloat(), Normalize(0., 255.)])
    
    train_set = MNIST(train = True)
    train_set.data_transform = data_transform
    train_set.data = train_set.data.reshape(-1, 1, 28, 28)
    train_loader = DataLoader(train_set, batch_size, shuffle = True)

    test_set = MNIST(train = False)
    test_set.data_transform = data_transform
    test_set.data = test_set.data.reshape(-1, 1, 28, 28)
    test_loader = DataLoader(test_set, batch_size, shuffle = False)
    
    model = ConvNet()
    optimizer = MomentumSGD(lr = 0.01).setup(model)
    
    if cuda.gpu_enable:
        print("Cuda Enabled!")
        train_loader.to_gpu()
        test_loader.to_gpu()
        model.to_gpu()
        
    print(f"Starting training with {model.__class__.__name__}...")
    for epoch in range(max_epochs):
        sum_loss, sum_acc = 0, 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            
            model.clear_grad()
            loss.backward()
            optimizer.update()
            
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)
            
        train_loss = sum_loss / len(train_set)
        train_acc = sum_acc / len(train_set)

        sum_loss, sum_acc = 0, 0
        with no_grad(), test_mode():
            for x, t in test_loader:
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)
                sum_loss += float(loss.data) * len(t)
                sum_acc += float(acc.data) * len(t)

        test_loss = sum_loss / len(test_set)
        test_acc = sum_acc / len(test_set)

        print(f"Epoch {epoch + 1}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
