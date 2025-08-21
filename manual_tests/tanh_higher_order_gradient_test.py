if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
from dezero import Variable
from dezero import plot_dot_graph
from dezero import tanh


x = Variable(np.array(np.pi / 4))
y = tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph = True)

iters = 2
for _ in range(iters):
    gx = x.grad
    x.clean_grad()
    gx.backward(create_graph = True)


gx = x.grad
gx.name = 'gx' + str(iters + 1)
plot_dot_graph(gx, verbose = False, to_file = "tanh_higher_order_gradient_order_" + str(iters + 1) + ".png")
