import numpy as np
from tensorflow.keras.utils import Sequence
import os

class DataGenerator(Sequence):
    def __init__(self, path, X_id, Y, target_shape=None, batch_size=32, num_classes=None, shuffle=True):
        self.batch_size = batch_size
        self.path = path
        self.target_shape = target_shape
        self.Y = Y
        self.indices = X_id
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        l = int(len(self.indices) / self.batch_size)
        if l*self.batch_size < len(self.indices):
            l += 1
        return l

    def __getitem__(self, index):
        index = self.index[index * self.batch_size:(index + 1) * self.batch_size]
        batch = self.indices[index]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.index = np.arange(len(self.indices))
        if self.shuffle == True:
            np.random.shuffle(self.index)

    def __get_data(self, batch):
        t_d, ch_d = self.target_shape

        X = np.empty((len(batch), ch_d, t_d))
        y = np.empty((len(batch), self.num_classes))

        for i, id in enumerate(batch):
            path = os.path.join(self.path, '{}.npy'.format(id))
            X[i,] = np.load(path)
            y[i] = self.Y[id-1]

        X = np.swapaxes(X, 1,2)
        return X, y
