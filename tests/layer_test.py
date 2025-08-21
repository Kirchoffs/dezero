from dezero import Parameter, Layer
import numpy as np


def test_add_parameter():
    layer = Layer()
    
    layer.p1 = Parameter(np.array(1))
    layer.p2 = Parameter(np.array(2))
    layer.p3 = Parameter(np.array(3))

    print("Parameters in layer:", layer._params)
    assert layer._params == {"p1", "p2", "p3"}
