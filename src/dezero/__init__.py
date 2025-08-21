is_core_basic = True

if is_core_basic:
    from .core_basic import Variable
    from .core_basic import setup_variable
    from .core_basic import Function
    from .core_basic import Add, Sub, Mul, Div, Exp, Pow, Square, Neg, Sin
    from .core_basic import add, sub, rsub, mul, div, rdiv, exp, pow, square, neg, sin, sin_maclaurin
    from .core_basic import Config
    from .core_basic import using_config, no_grad
    from .core_basic import numerical_derivative
else:
    pass

from .utils import plot_dot_graph

setup_variable()
