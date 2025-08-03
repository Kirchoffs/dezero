from .layers import Layer, Linear, Conv2d, RNN
from .utils import plot_dot_graph
from .functions import relu, dropout
from .functions_conv import max_pooling


class Model(Layer):
    def plot(self, *inputs, to_file = "graph.png", clear_grad = True):
        y = self.forward(*inputs)
        if clear_grad:
            self.clear_grad()
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


class MLPWithDropout(MLP):
    def __init__(self, fc_out_sizes, activation = relu, dropout_ratio = 0.5):
        super().__init__(fc_out_sizes, activation)
        self.dropout_ratio = dropout_ratio

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
            x = dropout(x, self.dropout_ratio)
        return self.layers[-1](x)


class VGG16(Model):
    def __init__(self, out_size = 1000):
        super().__init__()

        self.conv1_1 = Conv2d(64, kernel_size = 3, stride = 1, padding = 1)
        self.conv1_2 = Conv2d(64, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_1 = Conv2d(128, kernel_size = 3, stride = 1, padding = 1)
        self.conv2_2 = Conv2d(128, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_1 = Conv2d(256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_2 = Conv2d(256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_3 = Conv2d(256, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_1 = Conv2d(512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_2 = Conv2d(512, kernel_size = 3, stride = 1, padding = 1)
        self.conv4_3 = Conv2d(512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_1 = Conv2d(512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_2 = Conv2d(512, kernel_size = 3, stride = 1, padding = 1)
        self.conv5_3 = Conv2d(512, kernel_size = 3, stride = 1, padding = 1)
        self.fc6 = Linear(out_size = 4096)
        self.fc7 = Linear(out_size = 4096)
        self.fc8 = Linear(out_size = out_size)

    def forward(self, x):
        x = relu(self.conv1_1(x))
        x = relu(self.conv1_2(x))
        x = max_pooling(x, 2, 2)
        x = relu(self.conv2_1(x))
        x = relu(self.conv2_2(x))
        x = max_pooling(x, 2, 2)
        x = relu(self.conv3_1(x))
        x = relu(self.conv3_2(x))
        x = relu(self.conv3_3(x))
        x = max_pooling(x, 2, 2)
        x = relu(self.conv4_1(x))
        x = relu(self.conv4_2(x))
        x = relu(self.conv4_3(x))
        x = max_pooling(x, 2, 2)
        x = relu(self.conv5_1(x))
        x = relu(self.conv5_2(x))
        x = relu(self.conv5_3(x))
        x = max_pooling(x, 2, 2)
        x = x.reshape(x.shape[0], -1)
        x = relu(self.fc6(x))
        x = dropout(x)
        x = relu(self.fc7(x))
        x = dropout(x)
        x = self.fc8(x)
        return x


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()

        self.rnn = RNN(hidden_size)
        self.fc = Linear(in_size = None, out_size = out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        return self.fc(h)
