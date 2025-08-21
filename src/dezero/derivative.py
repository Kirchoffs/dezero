from .core_basic import Variable


def numerical_derivative(f, x, eps=1e-6):
    x_minus = Variable(x.data - eps)
    x_plus = Variable(x.data + eps)
    return (f(x_plus).data - f(x_minus).data) / (2 * eps)
