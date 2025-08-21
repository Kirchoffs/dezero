from .core import Variable, Parameter
from .core import Function
from .core import Config
from .core import using_config, no_grad
from .layers import Layer
from .models import Model
from .functions import Add, Sub, Mul, Div, Exp, Pow, Square, Neg, Sin, Cos, Tanh, Reshape, Transpose, Sum
from .functions import setup_variable
from .utils import plot_dot_graph


setup_variable()
