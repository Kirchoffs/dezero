if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import sin


x = Variable(np.linspace(-5 * np.pi, 5 * np.pi, 256))
y = sin(x)
y.backward(create_graph = True)

logs = [y.data]
for i in range(3):
    logs.append(x.grad.data)
    gx = x.grad
    x.clean_grad()
    gx.backward(create_graph = True)

lables = ["y = sin(x)", "y'", "y''", "y'''"]
for i, log in enumerate(logs):
    plt.plot(x.data, log, label = lables[i])
plt.legend(loc = "lower right")
plt.show()
