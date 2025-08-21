if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np

from dezero import Config
from dezero import using_config, no_grad
from dezero import Variable
from dezero import square, add

def test_config():
    Config.enable_backprop = False

    x = Variable(np.random.rand(1))
    y = square(x)

    try:
        y.backward()
    except RuntimeError as e:
        assert str(e) == "Backpropagation is disabled. Set 'Config.enable_backprop' to True to enable it."
    finally:
        Config.enable_backprop = True


def test_using_config():
    with using_config("enable_backprop", False):
        x = Variable(np.random.rand(1))
        y = square(x)

        try:
            y.backward()
        except RuntimeError as e:
            assert str(e) == "Backpropagation is disabled. Set 'Config.enable_backprop' to True to enable it."

    assert Config.enable_backprop is True


def test_no_grad():
    with no_grad():
        x = Variable(np.random.rand(1))
        y = square(x)

        try:
            y.backward()
        except RuntimeError as e:
            assert str(e) == "Backpropagation is disabled. Set 'Config.enable_backprop' to True to enable it."

    assert Config.enable_backprop is True
