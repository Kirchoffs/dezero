if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import dezero.functions as F
import dezero.models as M
import matplotlib.pyplot as plt


np.random.seed(42)

x = np.random.rand(128, 1)
y = np.sin(2 * np.pi * x) + 0.1 * np.random.rand(128, 1)

hidden_size = 12

model = M.MLP([hidden_size, 1], activation = F.sigmoid)

lr = 0.1
epochs = 16384

for i in range(epochs):
    y_pred = model(x)
    loss = F.mean_square_error(y_pred, y)
    
    model.clear_grad()
    loss.backward()
    
    for param in model.params():
        param.data -= lr * param.grad.data

    if i % 128 == 0:
        print(f"loss: {loss.data}")

x_test = np.linspace(0, 1, 100).reshape(100, 1)
y_test_pred = model(x_test)

plt.figure(figsize = (8, 6))
plt.plot(x_test, y_test_pred.data, color = "red", label = "Predicted")
plt.scatter(x, y, label = "Actual")
plt.title("Non-linear Regression Model with 'MLP'")
plt.legend()
plt.show()
