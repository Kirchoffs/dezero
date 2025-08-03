from typing import override
import numpy as np
import cupy as cp
from .utils import pair
from .core import Function, as_variable
from .functions import linear, sum as dezero_sum
from .cuda import get_array_module


def conv2d(image, kernel, bias = None, stride = 1, padding = 0):
    return Conv2d(stride, padding)(image, kernel, bias)


def conv2d_simple(image, kernel, bias = None, stride = 1, padding = 0):
    image, kernel = as_variable(image), as_variable(kernel)

    n, c, h, w = image.shape
    oc, c, kh, kw = kernel.shape
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    oh, ow = get_conv_out_size(h, kh, sh, ph), get_conv_out_size(w, kw, sw, pw)

    column = image_to_column(image, (kh, kw), stride, padding, to_matrix = True) # (n * oh * ow, c * kh * kw)
    kernel = kernel.reshape(oc, -1).transpose()                                  # (c * kh * kw, oc)

    intermediate_vector = linear(column, kernel, bias)
    processed_image = intermediate_vector.reshape(n, oh, ow, oc).transpose(0, 3, 1, 2)
    return processed_image


def deconv2d(image, kernel, bias = None, stride = 1, padding = 0, out_size = None):
    return Deconv2d(stride, padding, out_size)(image, kernel, bias)


def deconv2d_simple(image, kernel, bias = None, stride = 1, padding = 0, out_size = None):
    image, kernel = as_variable(image), as_variable(kernel)
    c, oc, kh, kw = kernel.shape
    n, c, h, w = image.shape
    sh, sw = pair(stride)
    ph, pw = pair(padding)

    if out_size is None:
        oh = get_deconv_out_size(h, kh, sh, ph)
        ow = get_deconv_out_size(w, kw, sw, pw)
    else:
        oh, ow = pair(out_size)

    weight = kernel.reshape(c, -1)
    column = linear(image.transpose(0, 2, 3, 1).reshape(-1, c), weight, None)
    y = column_to_image(column, (n, oc, oh, ow), (kh, kw), stride, padding, to_matrix = True)

    if bias is not None:
        y += bias.reshape(1, oc, 1, 1)
    return y


def max_pooling(image, kernel_size, stride = 1, padding = 0):
    return MaxPooling(kernel_size, stride, padding)(image)


def max_pooling_simple(image, kernel_size, stride = 1, padding = 0):
    image = as_variable(image)

    n, c, h, w = image.shape
    kh, kw = pair(kernel_size)
    ph, pw = pair(padding)
    sh, sw = pair(stride)
    oh, ow = get_conv_out_size(h, kh, sh, ph), get_conv_out_size(w, kw, sw, pw)

    column = image_to_column(image, kernel_size, stride, padding, to_matrix = True)   # (n * oh * ow, c * kh * kw)
    column = column.reshape(-1, kh * kw)                                              # (n * oh * ow * c, kh * kw)
    intermediate_vector = column.max(axis = 1)                                        # (n * oh * ow * c,)
    processed_image = intermediate_vector.reshape(n, oh, ow, c).transpose(0, 3, 1, 2) # (n, c, oh, ow)
    return processed_image


class Conv2d(Function):
    def __init__(self, stride = 1, padding = 0):
        super().__init__()
        self.stride = pair(stride)
        self.padding = pair(padding)
    
    def forward(self, x, k, b):
        xp = get_array_module(x)

        kh, kw = k.shape[2:] # W: (oc, c, kh, kw)
        x = image_to_column_array(x, (kh, kw), self.stride, self.padding, to_matrix = False) # (n, c, kh, kw, oh, ow)

        y = xp.tensordot(x, k, ((1, 2, 3), (1, 2, 3))) # (n, oh, ow, oc) or NHWC
        if y is not None:
            y += b
        y = xp.rollaxis(y, 3, 1) # (n, oc, oh, ow) or NCHW
        # Below method also works:
        # y = np.transpose(y, (0, 3, 1, 2))
        return y

    def backward(self, gy):
        x, k, b = self.inputs

        gx = deconv2d(gy, k, bias = None, stride = self.stride, padding = self.padding, out_size = (x.shape[2], x.shape[3]))

        gk = Conv2DKernelGrad(self)(x, gy)

        gb = None
        if b.data is not None:
            gb = dezero_sum(gy, axis = (0, 2, 3))

        return gx, gk, gb


def get_conv_out_size(image_size, kernel_size, stride, padding):
    return (image_size + padding * 2 - kernel_size) // stride + 1


class Deconv2d(Function):
    def __init__(self, stride = 1, padding = 0, out_size = None):
        super().__init__()
        self.stride = pair(stride)
        self.padding = pair(padding)
        self.out_size = out_size

    def forward(self, x, k, b):
        xp = get_array_module(x)

        sh, sw = self.stride
        ph, pw = self.padding
        c, oc, kh, kw = k.shape
        n, c, h, w = x.shape

        if self.out_size is None:
            oh = get_deconv_out_size(h, kh, sh, ph)
            ow = get_deconv_out_size(w, kw, sw, pw)
        else:
            oh, ow = pair(self.out_size)
        
        image_shape = (n, oc, oh, ow)
        x = xp.tensordot(k, x, (0, 1)) # (oc, kh, kw, n, h, w)
        x = xp.rollaxis(x, 3)          # (n, oc, kh, kw, h, w)

        y = column_to_image_array(x, image_shape, (kh, kw), self.stride, self.padding, to_matrix = False) # (n, oc, oh, ow)
        if b is not None:
            y += b.reshape((1, b.size, 1, 1))

        return y

    def backward(self, gy):
        x, k, b = self.inputs
        
        gx = conv2d(gy, k, bias = None, stride = self.stride, padding = self.padding)

        f = Conv2DKernelGrad(self)
        gk = f(gy, x)

        gb = None
        if b.data is not None:
            gb = dezero_sum(gy, axis = (0, 2, 3))
        return gx, gk, gb


def get_deconv_out_size(image_size, kernel_size, stride, padding):
    return stride * (image_size - 1) + kernel_size - 2 * padding


class MaxPooling(Function):
    def __init__(self, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        column = image_to_column_array(x, self.kernel_size, self.stride, self.padding, to_matrix = False)
        
        n, c, kh, kw, oh, ow = column.shape
        column = column.reshape(n, c, kh * kw, oh, ow)
        self.max_indexes = column.argmax(axis = 2)
        y = column.max(axis = 2)

        return y

    def backward(self, gy):
        return MaxPooling2dGrad(self)(gy)


class ImageToColumn(Function):
    def __init__(self, kernel_size, stride, padding, to_matrix):
        super().__init__()
        self.image_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.to_matrix = to_matrix

    def forward(self, x):
        self.image_shape = x.shape
        return image_to_column_array(x, self.kernel_size, self.stride, self.padding, self.to_matrix)

    def backward(self, gy):
        return column_to_image(gy, self.image_shape, self.kernel_size, self.stride, self.padding, self.to_matrix)


def image_to_column(image, kernel_size, stride, padding, to_matrix = True):
    return ImageToColumn(kernel_size, stride, padding, to_matrix)(image)


class ColumnToImage(Function):
    def __init__(self, column_shape, kernel_size, stride, padding, to_matrix):
        super().__init__()
        self.column_shape = column_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.to_matrix = to_matrix

    def forward(self, x):
        return column_to_image_array(x, self.column_shape, self.kernel_size, self.stride, self.padding, self.to_matrix)

    def backward(self, gy):
        return image_to_column(gy, self.kernel_size, self.stride, self.padding, self.to_matrix)


def column_to_image(column, column_shape, kernel_size, stride=1, pad=0, to_matrix = True):
    return ColumnToImage(column_shape, kernel_size, stride, pad, to_matrix)(column)


class Conv2DKernelGrad(Function):
    def __init__(self, conv2d):
        k = conv2d.inputs[1]
        kh, kw = k.shape[2:]
        self.kernel_size = (kh, kw)
        self.stride = conv2d.stride
        self.padding = conv2d.padding

    def forward(self, x, gy):
        xp = get_array_module(x)

        x = image_to_column_array(x, self.kernel_size, self.stride, self.padding, to_matrix = False) # (n, c, kh, kw, oh, ow)
        # gy: (n, oc, oh, ow)
        gk = xp.tensordot(gy, x, ((0, 2, 3), (0, 4, 5))) # gk: (oc, c, oh, ow)
        return gk

    def backward(self, ggk):
        x, gy = self.inputs
        xh, xw = x.shape[2:]

        gx = deconv2d(gy, ggk, stride = self.stride, padding = self.padding, out_size = (xh, xw))
        ggy = conv2d(x, ggk, stride = self.stride, padding = self.padding)
        return gx, ggy


class MaxPooling2dGrad(Function):
    def __init__(self, pooling2d):
        pooling2d_inputs, = pooling2d.inputs

        self.pooling2d = pooling2d
        self.kernel_size = pooling2d.kernel_size
        self.stride = pooling2d.stride
        self.padding = pooling2d.padding
        self.input_shape = pooling2d_inputs.shape
        self.dtype = pooling2d_inputs.dtype
        self.max_indexes = pooling2d.max_indexes # (n, c, oh, ow)

    def forward(self, gy):
        xp = get_array_module(gy)

        n, c, oh, ow = gy.shape
        n, c, h, w = self.input_shape
        kh, kw = pair(self.kernel_size)

        g_column = xp.zeros((n * c * oh * ow * kh * kw), dtype = self.dtype)

        max_indexes = self.max_indexes.ravel() + xp.arange(0, self.max_indexes.size * kh * kw, kh * kw)
        
        g_column[max_indexes] = gy.ravel()
        g_column = g_column.reshape(n, c, oh, ow, kh, kw)
        g_column = xp.swapaxes(g_column, 2, 4)
        g_column = xp.swapaxes(g_column, 3, 5)
        # g_column = g_column.transpose(0, 1, 4, 5, 2, 3)
        # g_column: (n, c, kh, kw, oh, ow)

        gx = column_to_image_array(
            g_column, 
            (n, c, h, w), 
            self.kernel_size, 
            self.stride,
            self.padding, 
            to_matrix = False
        )
        return gx

    def backward(self, ggx):
        return MaxPooling2dWithIndexes(self.pooling2d)(ggx)


class MaxPooling2dWithIndexes(Function):
    def __init__(self, pooling2d):
        pooling2d_inputs, = pooling2d.inputs

        self.kernel_size = pooling2d.kernel_size
        self.stride = pooling2d.stride
        self.padding = pooling2d.padding
        self.input_shape = pooling2d_inputs.shape
        self.dtype = pooling2d_inputs.dtype
        self.max_indexes = pooling2d.max_indexes

    def forward(self, ggx):
        column = image_to_column_array(
            ggx, self.kernel_size, self.stride, self.padding,
            to_matrix = False
        )

        n, c, kh, kw, oh, ow = column.shape
        column = column.reshape(n, c, kh * kw, oh, ow)
        column = column.transpose(0, 1, 3, 4, 2).reshape(-1, kh * kw)
        max_indexes = self.max_indexes.ravel()
        column = column[np.arange(len(max_indexes)), max_indexes]
        return column.reshape(n, c, oh, ow)


def image_to_column_array(image, kernel_size, stride, padding, to_matrix = True):
    n, c, h, w = image.shape
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    oh = get_conv_out_size(h, kh, sh, ph)
    ow = get_conv_out_size(w, kw, sw, pw)

    xp = get_array_module(image)
    if xp is not np:
        column = _image_to_column_gpu(image, kernel_size, stride, padding)
    else:
        image = np.pad(image, ((0, 0), (0, 0), (ph, ph + sh - 1), (pw, pw + sw - 1)), mode = 'constant', constant_values = (0,))
        column = np.ndarray((n, c, kh, kw, oh, ow), dtype = image.dtype)

        for j in range(kh):
            j_boundary = j + sh * oh # j + sh * oh = j + sh * [(h + 2ph - kh) / sh + 1] <= j + h + 2ph - kh + sh <= h + 2ph + sh - 1
            for i in range(kw):
                i_boundary = i + sw * ow
                column[:, :, j, i, :, :] = image[:, :, j:j_boundary:sh, i:i_boundary:sw]
        
    if to_matrix:
        column = column.transpose((0, 4, 5, 1, 2, 3)).reshape((n * oh * ow, -1)) # (n * oh * ow, c * kh * kw)
    
    return column

    # Q1: Why not:
    # column = np.ndarray((n, c, oh, ow, kh, kw), dtype = image.dtype)
    # for j in range(kh):
    #     j_boundary = j + sh * oh
    #     for i in range(kw):
    #         i_boundary = i + sw * ow
    #         column[:, :, :, :, j, i] = image[:, :, j:j_boundary:sh, i:i_boundary:sw]
    # A1: For every pixel you write, you must skip a space of length kh * kw before writing the next pixel.
    #
    # Q2: Why not:
    # column = np.ndarray((n, c, oh, ow, kh, kw), dtype = image.dtype)
    # for y in range(oh):
    #     h_start = y * sh
    #     h_end = h_start + kh
    #     for x in range(ow):
    #         w_start = x * sw
    #         w_end = w_start + kw
    #         column[:, :, y, x, :, :] = image[:, :, h_start:h_end, w_start:w_end]
    # A2: The number of pairs (oh, ow) is much larger than (kh, kw).


def column_to_image_array(column, image_shape, kernel_size, stride, padding, to_matrix = True):
    n, c, h, w = image_shape
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    oh = get_conv_out_size(h, kh, sh, ph)
    ow = get_conv_out_size(w, kw, sw, pw)

    if to_matrix:
        column = column.reshape(n, oh, ow, c, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    xp = get_array_module(column)
    if xp is not np:
        image = _column_to_image_gpu(column, sh, sw, ph, pw, h, w)
        return image
    else:
        image = np.zeros(
            (n, c, h + 2 * ph + sh - 1, w + 2 * pw + sw - 1),
            dtype = column.dtype
        )

        for j in range(kh):
            j_boundary = j + sh * oh # j + sh * oh = j + sh * [(h + 2ph - kh) / sh + 1] <= j + h + 2ph - kh + sh <= h + 2ph + sh - 1
            for i in range(kw):
                i_boundary = i + sw * ow
                image[:, :, j:j_boundary:sh, i:i_boundary:sw] += column[:, :, j, i, :, :]
        return image[:, :, ph:h + ph, pw:w + pw]


def _image_to_column_gpu(image, kernel_size, stride, padding):
    n, c, h, w = image.shape
    kh, kw = pair(kernel_size)
    sh, sw = pair(stride)
    ph, pw = pair(padding)
    oh, ow = get_conv_out_size(h, kh, sh, ph), get_conv_out_size(w, kw, sw, pw)
    dy, dx = 1, 1
    column = cp.empty((n, c, kh, kw, oh, ow), dtype = image.dtype)

    # i <=> (ni, ci, ky, kx, oy, ox)
    # i = ni * (c * kh * kw * oh * ow) + ci * (kh * kw * oh * ow) + ky * (kw * oh * ow) + kx * (oh * ow) + oy * (ow) + ox
    #
    # dimension collapsing for batch and channel: nc
    # i <=> (nci, ky, kx, oy, ox)
    # i = nci * (kh * kw * oh * ow) + ky * (kw * oh * ow) + kx * (oh * ow) + oy * (ow) + ox
    cp.ElementwiseKernel(
        'raw T image, int32 h, int32 w, int32 oh, int32 ow,'
        'int32 kh, int32 kw, int32 sh, int32 sw, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T column',
        '''
            int nci = i / (kh * kw * oh * ow);
            int ky = i / (kw * oh * ow) % kh;
            int kx = i / (oh * ow) % kw;
            int oy = i / ow % oh;
            int ox = i % ow;
            int yi = ky * dy + oy * sh - ph;
            int xi = kx * dx + ox * sw - pw;

            if (yi >= 0 && yi < h && xi >= 0 && xi < w) {
                // int idx = nci * h * w + yi * w + xi;
                int idx = xi + w * (yi + h * nci);
                column = image[idx];
            } else {
                column = 0;
            }
        ''',
        'image_to_column'
    )(image.reduced_view(), h, w, oh, ow, kh, kw, sh, sw, ph, pw, dy, dx, column)

    return column


def _column_to_image_gpu(column, sh, sw, ph, pw, h, w):
    n, c, kh, kw, oh, ow = column.shape
    dx, dy = 1, 1
    image = cp.empty((n, c, h, w), dtype = column.dtype)

    # i <=> (ni, ci, yi, xi)
    # i = ni * (c * h * w) + ci * (h * w) + yi * (w) + xi
    #
    # dimension collapsing for batch and channel: nc
    # i <=> (nci, yi, xi)
    # i = nci * (h * w) + yi * (w) + xi
    cp.ElementwiseKernel(
        'raw T column, int32 h, int32 w, int32 oh, int32 ow,'
        'int32 kh, int32 kw, int32 sh, int32 sw, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T image',
        '''
            int nci = i / (h * w);
            int yi  = i / w % h;
            int xi  = i % w;

            T res = 0;
            for (int ky = 0; ky < kh; ++ky) {
                int oy = (yi + ph - ky * dy);
                if (oy < 0 || oy >= oh * sh) continue;
                if (oy % sh != 0) continue;
                oy /= sh;

                for (int kx = 0; kx < kw; ++kx) {
                    int ox = (xi + pw - kx * dx);
                    if (ox < 0 || ox >= ow * sw) continue;
                    if (ox % sw != 0) continue;
                    ox /= sw;

                    // int idx = nci * (kh * kw * oh * ow) + ky * (kw * oh * ow) + kx * (oh * ow) + oy * ow + ox;
                    int idx = ox + ow * (oy + oh * (kx + kw * (ky + kh * nci)));
                    res = res + column[idx];
                }
            }

            image = res;
        ''',
        'column_to_image'
    )(column.reduced_view(), h, w, oh, ow, kh, kw, sh, sw, ph, pw, dx, dy, image)
    
    return image
