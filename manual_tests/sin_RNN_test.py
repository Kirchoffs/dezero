if "__file__" in globals():
    import os
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))


import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dezero.datasets import SinCurve
from dezero.models import SimpleRNN
from dezero.optimizers import Adam
from dezero import no_grad

import dezero.functions as F


train_set = SinCurve(train = True)
xs = [data[0] for data in train_set]
ts = [data[1] for data in train_set]

plt.plot(np.arange(len(xs)), xs, label = 'xs')
plt.plot(np.arange(len(ts)), ts, label = 'ts')
plt.legend()

output_file = 'sin_curve.png'
plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)
plt.close()

max_epoch = 99
hidden_size = 99
bptt_length = 35
seq_length = len(train_set)

model = SimpleRNN(hidden_size, 1)
optimizer = Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()

    loss, count = 0, 0
    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_square_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seq_length:
            model.clear_grad()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()
        
    avg_loss = float(loss.data) / count
    print('| epoch %d | loss %f |' % (epoch + 1, avg_loss))

xs = np.cos(np.linspace(0, 4 * np.pi, 999))
model.reset_state()
ys = []

with no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        ys.append(y.data.item())

plt.plot(np.arange(len(xs)), xs, label = 'Target')
plt.plot(np.arange(len(xs)), ys, label = 'Predict')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

output_file = 'sin_curve_pred.png'
plt.savefig(output_file, bbox_inches = 'tight', dpi = 300)
plt.close()
