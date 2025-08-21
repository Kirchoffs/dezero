if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np
from dezero import Variable
from dezero import sin

x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward(create_graph = True)

for i in range(3):
    gx = x.grad
    x.clean_grad()
    gx.backward(create_graph = True)
    print(f"{i + 2} order derivative:", x.grad.data)
