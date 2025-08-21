import numpy as np
from dezero import Variable, Model
import dezero.layers as L
import dezero.functions as F

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(out_size = hidden_size)
        self.l2 = L.Linear(out_size = out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


x = Variable(np.random.randn(5, 16), name = 'x')
model = TwoLayerNet(128, 8)
model.plot(x, to_file = 'two_layer_network.png')
