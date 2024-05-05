from copy import deepcopy
from glob import glob
import tensorflow as tf
import numpy as np
import json
import yaml
import math
import os

class Data_Loader():
    def __init__(self, sim, num, target_cols, delta_t, batch_size, is_normalize=False, is_standardize=False, shuffle=True):
        self.sim = sim
        self.num = num
        self.target_cols = target_cols
        self.delta_t = delta_t
        self.batch_size = batch_size
        self.is_normalize = is_normalize
        self.is_standardize = is_standardize
        self.shuffle = shuffle

    def get_data(self):
        path = os.path.join(os.getcwd(), 'data', self.sim, f'num_{self.num}', 'sim_result.json')
        with open(path, 'r') as f:
            f = f.read()
            dict_r = json.loads(f)
        
        self.s, self.s_next = self.get_s_AND_s_next(dict_r)
        self.indices = np.arange(len(self.s))

    def get_s_AND_s_next(self, dict_result):
        cycle_time = round(dict_result['time'][1], 2)
        patience = int(self.delta_t / cycle_time)

        data = []
        for col in self.target_cols:
            data.append(dict_result[col])
        
        data = np.transpose(np.array(data, dtype=np.float32))

        if self.is_normalize:
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
        
        elif self.is_standardize:
            self.mean = np.mean(data, axis=0)
            self.max = np.std(data, axis=0)

        return deepcopy(data[:len(data)-patience-1,:]), deepcopy(data[patience:,:])
    
    def split_train_val(self, val_ratio):
        val_idx = np.random.choice(self.indices, size = int(len(self.indices)*val_ratio), replace=False)
        self.s_val = self.s[val_idx]
        self.s_next_val = self.s_next[val_idx]
        self.indices_val = np.arange(len(self.s_val))

        self.s_train = np.delete(self.s, val_idx, axis=0)
        self.s_next_train = np.delete(self.s_next, val_idx, axis=0)
        self.indices_train = np.arange(len(self.s_train))

        self.len_train = int(math.ceil(len(self.s_train) / self.batch_size))
        self.len_val = int(math.ceil(len(self.s_val) / self.batch_size))

        del self.s, self.s_next

    def get_train_batch(self, idx):
        batch_idx = self.indices_train[idx*self.batch_size:(idx+1)*self.batch_size]

        X = self.s_train[batch_idx]
        Y = self.s_next_train[batch_idx]

        if self.is_normalize:
            X = np.divide((X - self.min), (self.max - self.min))
            Y = np.divide((Y - self.min), (self.max - self.min))
            Y = np.subtract(Y, X)

        elif self.is_standardize:
            X = (X - self.mean) / self.std
            Y = (Y - self.mean) / self.std
            Y = np.subtract(Y, X)

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)

    def get_val_batch(self, idx):
        batch_idx = self.indices_val[idx*self.batch_size:(idx+1)*self.batch_size]

        X = self.s_val[batch_idx]
        Y = self.s_next_val[batch_idx]

        if self.is_normalize:
            X = np.divide((X - self.min), (self.max - self.min))
            Y = np.divide((Y - self.min), (self.max - self.min))
            Y = np.subtract(Y, X)

        elif self.is_standardize:
            X = (X - self.mean) / self.std
            Y = (Y - self.mean) / self.std
            Y = np.subtract(Y, X)

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)

    def on_epoch_end(self):
        self.indices_train = np.arange(len(self.s_train))
        self.indices_val = np.arange(len(self.s_val))
        if self.shuffle == True:
            np.random.shuffle(self.indices_train)
            np.random.shuffle(self.indices_val)

if __name__ == '__main__':
    dataloader = Data_Loader(
        sim = 'freefall',
        num = 2,
        target_cols=['s', 'v'],
        delta_t=0.1,
        batch_size=128
    )

    dataloader.get_data()
    dataloader.split_train_val(0.2)
    x, y = dataloader.get_train_batch(0)
    print(x.shape)