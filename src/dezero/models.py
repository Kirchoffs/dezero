from .layers import Layer, Linear
from .utils import plot_dot_graph
from .functions import relu


class Model(Layer):
    def plot(self, *inputs, to_file = "graph.png"):
        y = self.forward(*inputs)
        return plot_dot_graph(y, True, to_file)
    

class MLP(Model):
    def __init__(self, fc_out_sizes, activation = relu):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_out_sizes):
            layer = Linear(out_size = out_size)
            setattr(self, f"l{i+1}", layer)
            self.layers.append(layer)
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
