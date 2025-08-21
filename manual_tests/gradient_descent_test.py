if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable


def rosenbrock_gradient_descent():
    def rosenbrock(x0, x1):
        a = 1.0
        b = 100.0
        return b * (x1 - x0 ** 2) ** 2 + (a - x0) ** 2

    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))

    lr = 0.001
    iters = 10000

    x0_list = []
    x1_list = []
    for _ in range(iters):
        y = rosenbrock(x0, x1)

        x0.clean_grad()
        x1.clean_grad()

        y.backward()

        x0.data -= lr * x0.grad
        x1.data -= lr * x1.grad

        x0_list.append(x0.data.copy())
        x1_list.append(x1.data.copy())

    plt.figure(figsize = (6, 6))
    plt.plot(x0_list, x1_list, marker = 'o')
    plt.plot(1, 1, marker = 'x', color = 'red', markersize = 12)  # Minimum point
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.title("Rosenbrock Function Optimization")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    rosenbrock_gradient_descent()
