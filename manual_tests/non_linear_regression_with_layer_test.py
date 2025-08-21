if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import dezero.functions as F
import dezero.layers as L
from dezero import Variable
import matplotlib.pyplot as plt


np.random.seed(42)

x = np.random.rand(128, 1)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.rand(128, 1)

I, H, O = 1, 12, 1
l1 = L.Linear(out_size = H)
l2 = L.Linear(out_size = O)

def predict(x):
    y = l1(x)
    y = F.sigmoid(y)
    y = l2(y)
    return y

lr = 0.1
epochs = 16384

for i in range(epochs):
    y_pred = predict(x)
    loss = F.mean_square_error(y_pred, y)
    
    l1.clear_grad()
    l2.clear_grad()
    
    loss.backward()
    
    for l in (l1, l2):
        for param in l.params():
            param.data -= lr * param.grad.data

    if i % 128 == 0:
        print(f"loss: {loss.data}")

x_test = np.linspace(0, 1, 100).reshape(100, 1)
y_test_pred = predict(x_test)

plt.figure(figsize = (8, 6))
plt.plot(x_test, y_test_pred.data, color = "red", label = "Predicted")
plt.scatter(x, y, label = "Actual")
plt.title("Non-linear Regression Model with Layer")
plt.legend()
plt.show()
