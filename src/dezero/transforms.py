import numpy as np
from PIL import Image
from .utils import pair


class Compose:
    def __init__(self, transforms = []):
        self.transforms = transforms

    def __call__(self, data):
        if not self.transforms:
            return data

        for transform in self.transforms:
            data = transform(data)
        return data


#
# Transforms for PIL Image
#

class Convert:
    def __init__(self, mode = 'RGB'):
        self.mode = mode

    def __call__(self, img):
        if self.mode == 'BGR':
            img = img.convert('RGB')
            r, g, b = img.split()
            img = Image.merge('RGB', (b, g, r))
            return img
        else:
            return img.convert(self.mode)
        

class Resize:
    def __init__(self, size, mode = Image.BILINEAR):
        self.size = size

    def __call__(self, img):
        return img.resize(self.size)


class CenterCrop:
    def __init__(self, size):
        self.size = pair(size)

    def __call__(self, img):
        w, h = img.size
        th, tw = self.size

        left = (w - tw) // 2
        right = w - ((w - tw) // 2 + (w - tw) % 2)
        top = (h - th) // 2
        bottom = h - ((h - th) // 2 + (h - th) % 2)

        return img.crop((left, top, right, bottom))
    

class ToArray:
    def __init__(self, dtype = np.float32):
        self.dtype = dtype

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            return img
        elif isinstance(img, Image.Image):
            img = np.asarray(img)
            
            # Convert from HWC (Height Width Channel) to CHW (Channel Height Width)
            # HWC is the default format for images in PIL, while CHW is commonly used in PyTorch.
            # If grayscale (HW), convert to (1, H, W)
            if img.ndim == 2:
                img = img[np.newaxis, :, :]
            else:
                img = img.transpose((2, 0, 1))

            img = img.astype(self.dtype)
            return img
        else:
            raise TypeError("Unsupported type for ToArray transform")


class ToPIL:
    def __call__(self, array):
        data = array.transpose((1, 2, 0))
        if data.shape[2] == 1:
            data = data.squeeze(axis = 2)
        return Image.fromarray(data)

class RandomHorizontalFlip:
    pass


#
# Transforms for NumPy ndarray
#

class Normalize:
    def __init__(self, mean = 0.0, std = 1.0, channel_axis = 0):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.axis = channel_axis

    def __call__(self, array):
        array = np.asanyarray(array)
        
        mean = self._prepare_param(self.mean, array.ndim)
        std = self._prepare_param(self.std, array.ndim)

        return (array - mean) / std

    def _prepare_param(self, param, ndim):
        if param.ndim == 0:
            return param
        
        new_shape = [1] * ndim
        try:
            new_shape[self.axis] = param.size
            return param.reshape(*new_shape)
        except IndexError:
            raise ValueError(f"Axis {self.axis} is out of bounds for array of dimension {ndim}")


class Flatten:
    def __call__(self, array):
        return array.flatten()


class AsType:
    def __init__(self, dtype = np.float32):
        self.dtype = dtype

    def __call__(self, array):
        return array.astype(self.dtype)


class ToInt(AsType):
    def __init__(self, dtype = int):
        self.dtype = dtype


ToFloat = AsType
