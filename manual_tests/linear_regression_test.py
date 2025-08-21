if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import dezero.functions as F
from dezero import Variable


np.random.seed(42)

x = np.random.rand(128, 1)
y = 3 * x + 1 + 0.1 * np.random.rand(128, 1)

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

def predict(x):
    return F.matmul(x, W) + b

def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    return F.sum(diff ** 2) / len(diff)

lr = 0.1
epochs = 256

for _ in range(epochs):
    y_pred = predict(x)
    loss = mse_loss(y_pred, y)
    
    W.clear_grad()
    b.clear_grad()
    
    loss.backward()
    
    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data
    print(f"W: {W.data}, b: {b.data}")
