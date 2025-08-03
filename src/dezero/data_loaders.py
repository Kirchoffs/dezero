import math
import numpy as np


class DataLoader:
    def __init__(self, dataset, batch_size, shuffle = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / self.batch_size)

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
        x = np.array([item[0] for item in batch])
        t = np.array([item[1] for item in batch])

        self.current_iter += 1
        return x, t
    
    def next(self):
        return self.__next__()
