import numpy as np
import dezero
from dezero import Variable, test_mode as dezero_test_mode
from dezero.models import MLP, MLPWithDropout

def test_mlp_with_dropout_train_test_mode():
    x = Variable(np.ones((10, 100)))
    model = MLPWithDropout([100, 10], dropout_ratio=0.5)
    
    # In train mode (default), dropout should be applied
    y1 = model(x)
    y2 = model(x)
    # Since dropout is random, y1 and y2 should be different
    assert not np.array_equal(y1.data, y2.data)

    # In test mode, dropout should not be applied
    with dezero_test_mode():
        y3 = model(x)
        y4 = model(x)
        # In test mode, it should be deterministic
        assert np.array_equal(y3.data, y4.data)
        # In test mode, it should be different from train mode (usually)
        # though there is a tiny chance it could be the same, but very unlikely
        assert not np.array_equal(y1.data, y3.data)

def test_mlp_with_dropout_params():
    model = MLPWithDropout([100, 10], dropout_ratio=0.5)
    params = list(model.params())
    # 2 layers, each has W and b
    assert len(params) == 4
    for p in params:
        assert isinstance(p, dezero.Parameter)

def test_mlp_forward():
    x = Variable(np.ones((2, 5)))
    model = MLP([10, 3])
    y = model(x)
    assert y.shape == (2, 3)
