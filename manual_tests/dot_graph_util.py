import numpy as np
from dezero import Variable
from dezero import plot_dot_graph


def goldstein_price(x, y):
    return (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
           (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))


x = Variable(np.array(1.0))
y = Variable(np.array(1.0))
z = goldstein_price(x, y)
z.backward()

x.name = "x"
y.name = "y"
z.name = "z"
plot_dot_graph(z, verbose = True, to_file = "goldstein_price.png")
