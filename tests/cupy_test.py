import numpy as np
import cupy as cp

def test_cupy():
    x = cp.arange(6).reshape(2, 3)
    y = x.sum(axis = 1, keepdims = True)

    assert np.array_equal(cp.asnumpy(y), np.array([[3], [12]]))

def test_cupy_numpy_transform():
    tensor_np = np.array([1, 2, 3])
    tensor_cp = cp.asarray(tensor_np)
    assert type(tensor_cp) == cp.ndarray

    tensor_cp = cp.array([1, 2, 3])
    tensor_np = cp.asnumpy(tensor_cp)
    assert type(tensor_np) == np.ndarray

    x = np.array([1, 2, 3])
    xp = cp.get_array_module(x)
    assert xp is np

    x = cp.array([1, 2, 3])
    xp = cp.get_array_module(x)
    assert xp is cp
