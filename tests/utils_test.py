import numpy as np
from dezero.functions_conv import image_to_column


def test_image_to_column_basic():
    print("\nTest image_to_column Basic")

    n, c, h, w = 1, 1, 3, 3
    kh, kw = 2, 2
    image = np.array([[[[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]]]], dtype = np.float32)
    
    expected_column = np.array([
        [0, 1, 3, 4],
        [1, 2, 4, 5],
        [3, 4, 6, 7],
        [4, 5, 7, 8]
    ], dtype = np.float32)
    
    column = image_to_column(image, (kh, kw), stride = 1, padding = 0, to_matrix = True)

    assert column.shape == (4, 4)
    assert np.allclose(column, expected_column)


def test_image_to_column_stride():
    print("\nTest image_to_column Stride")

    n, c, h, w = 1, 1, 4, 4
    kh, kw = 2, 2
    sh, sw = 2, 2
    image = np.arange(16).reshape(1, 1, 4, 4).astype(np.float32)
    
    expected_column = np.array([
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15]
    ], dtype = np.float32)
    
    column = image_to_column(image, (kh, kw), stride = (sh, sw), padding = 0, to_matrix = True)

    assert column.shape == (4, 4)
    assert np.allclose(column, expected_column)


def test_image_to_column_pad():
    print("\nTest im2col Padding")

    n, c, h, w = 1, 1, 2, 2
    kh, kw = 2, 2
    ph, pw = 1, 1
    x = np.arange(4).reshape(1, 1, 2, 2).astype(np.float32)

    column = image_to_column(x, (kh, kw), stride = 1, padding = (ph, pw), to_matrix = True)

    assert column.shape == (9, 4)
    assert np.allclose(column[4], [0, 1, 2, 3])


def test_image_to_column_6d():
    print("\nTest image_to_column 6D Output")
    
    n, c, h, w = 2, 3, 5, 5
    kh, kw = 3, 3
    image = np.random.randn(n, c, h, w).astype(np.float32)
    
    column = image_to_column(image, (kh, kw), stride = 1, padding = 0, to_matrix = False)

    assert column.shape == (2, 3, 3, 3, 3, 3)
