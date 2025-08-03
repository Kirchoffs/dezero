import math
import numpy as np
import cupy as cp
import cuda


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle = True, gpu = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)
        self.gpu = gpu

        self.reset()

    def reset(self):
        self.current_iter = 0
        if self.shuffle:
            self.indices = np.random.permutation(self.data_size)
        else:
            self.indices = np.arange(self.data_size)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_iter >= self.max_iter:
            self.reset()
            raise StopIteration
        
        current_iter, batch_size = self.current_iter, self.batch_size
        batch_indices = self.indices[current_iter * batch_size : (current_iter + 1) * batch_size]
        batch = [self.dataset[i] for i in batch_indices]

        xp = cp if self.gpu else np
        x = xp.array([item[0] for item in batch])
        t = xp.array([item[1] for item in batch])

        self.current_iter += 1
        return x, t
    
    def next(self):
        return self.__next__()

    def to_cpu(self):
        self.gpu = False

    def to_gpu(self):
        self.gpu = True
