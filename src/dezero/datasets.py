import numpy as np

from .transforms import Compose, Flatten, ToFloat, Normalize
from .utils import download_and_parse


class Dataset:
    def __init__(self, train = True, data_transform = lambda x: x, labels_transform = lambda x: x):
        self.train = train
        self.data_transform = data_transform
        self.labels_transform = labels_transform
        self.data = None
        self.labels = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)

        if self.labels is None:
            return self.data_transform(self.data[index]), None
        return self.data_transform(self.data[index]), self.labels_transform(self.labels[index])
    
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        raise NotImplementedError("prepare method must be implemented in subclasses")


class MNIST(Dataset):
    def __init__(self, train = True):
        data_transform = Compose([Flatten(), ToFloat(), Normalize(0., 255.)])
        super().__init__(train, data_transform)

    def prepare(self):
        self.data, self.labels = self._load_mnist_raw(train = self.train)
        print(f"MNIST {'train' if self.train else 'test'} loaded. Total: {len(self.data)}")

    def _load_mnist_raw(self, save_dir = '/tmp/mnist', train = True):
            base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
            
            image_filename = 'train-images-idx3-ubyte.gz' if train else 't10k-images-idx3-ubyte.gz'
            label_filename = 'train-labels-idx1-ubyte.gz' if train else 't10k-labels-idx1-ubyte.gz'
            
            x = download_and_parse(base_url, image_filename, save_dir, offset = 16)
            y = download_and_parse(base_url, label_filename, save_dir, offset = 8)
            
            return x.reshape(-1, 28, 28), y
