is_core_basic = True

if is_core_basic:
    from .core_basic import Variable
    from .core_basic import setup_variable
    from .core_basic import Function
    from .core_basic import Add, Exp, Square
    from .core_basic import add, exp, square
    from .core_basic import Config
    from .core_basic import using_config, no_grad
    from .derivative import numerical_derivative
else:
    pass

setup_variable()
from .utils import plot_dot_graph
