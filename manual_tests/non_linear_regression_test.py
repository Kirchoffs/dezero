if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import dezero.functions as F
from dezero import Variable
import matplotlib.pyplot as plt


np.random.seed(42)

x = np.random.rand(128, 1)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.rand(128, 1)

I, H, O = 1, 12, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

lr = 0.1
epochs = 16384

for i in range(epochs):
    y_pred = predict(x)
    loss = F.mean_square_error(y_pred, y)
    
    W1.clear_grad()
    b1.clear_grad()
    W2.clear_grad()
    b2.clear_grad()
    
    loss.backward()
    
    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data

    if i % 128 == 0:
        print(f"loss: {loss.data}")

x_test = np.linspace(0, 1, 100).reshape(100, 1)
y_test_pred = predict(x_test)

plt.figure(figsize = (8, 6))
plt.plot(x_test, y_test_pred.data, color = "red", label = "Predicted")
plt.scatter(x, y, label = "Actual")
plt.title("Non-linear Regression Model")
plt.legend()
plt.show()
