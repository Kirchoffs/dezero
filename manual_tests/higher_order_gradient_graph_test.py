if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F
from dezero import Variable


x = Variable(np.linspace(-5 * np.pi, 5 * np.pi, 256))
y = F.sin(x)
y.backward(create_graph = True)

logs = [y.data]
for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.clear_grad()
    gx.backward(create_graph = True)

lables = ["y = sin(x)", "y'", "y''", "y'''"]
for i, log in enumerate(logs):
    plt.plot(x.data, log, label = lables[i])
plt.title("Higher-order Gradients of sin(x)")
plt.legend(loc = "lower right")
plt.show()
