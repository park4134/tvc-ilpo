from copy import deepcopy
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import json
# import yaml
import math
import os

class DataLoader():
    def __init__(self, sim, expert_num, data_num, target_cols, delta_t, batch_size, shuffle=True):
        self.sim = sim
        self.expert_num = expert_num
        self.data_num = data_num
        self.target_cols = target_cols
        self.delta_t = delta_t
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_data(self):
        self.s_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}',f's_{self.delta_t}.npy')
        self.s_next_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}',f's_next_{self.delta_t}.npy')
        self.s = np.load(self.s_path)
        self.s_next = np.load(self.s_next_path)

        self.indices = np.arange(len(self.s))
        self.len = int(math.ceil(len(self.s) / self.batch_size))

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

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(np.subtract(Y, X), dtype=tf.float32)

    def get_val_batch(self, idx):
        batch_idx = self.indices_val[idx*self.batch_size:(idx+1)*self.batch_size]

        X = self.s_val[batch_idx]
        Y = self.s_next_val[batch_idx]

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(np.subtract(Y, X), dtype=tf.float32)

    def get_test_batch(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        X = self.s[batch_idx]
        Y = self.s_next[batch_idx]

        return tf.convert_to_tensor(X, dtype=tf.float32), tf.convert_to_tensor(np.subtract(Y, X), dtype=tf.float32), tf.convert_to_tensor(Y, dtype=tf.float32)

    def on_epoch_end(self):
        self.indices_train = np.arange(len(self.s_train))
        self.indices_val = np.arange(len(self.s_val))
        if self.shuffle == True:
            np.random.shuffle(self.indices_train)
            np.random.shuffle(self.indices_val)

class SeqDataLoader(DataLoader):
    def __init__(self, sim, expert_num, data_num, target_cols, seq, delta_t, batch_size, shuffle=True):
        super().__init__(
            sim = sim,
            expert_num = expert_num,
            data_num = data_num,
            shuffle = shuffle,
            target_cols = target_cols,
            delta_t = delta_t,
            batch_size = batch_size
        )
        self.seq = seq
    
    def get_data(self):
        self.s_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}', f'seq{self.seq}_s_{self.delta_t}.npy')
        self.s_next_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}', f'seq{self.seq}_s_next_{self.delta_t}.npy')
        self.s = np.load(self.s_path)
        self.s_next = np.load(self.s_next_path)

        self.indices = np.arange(len(self.s))
        self.len = int(math.ceil(len(self.s) / self.batch_size))
            
    def get_train_batch(self, idx):
        batch_idx = self.indices_train[idx*self.batch_size:(idx+1)*self.batch_size]

        seq_s = self.s_train[batch_idx]
        s_next = self.s_next_train[batch_idx]
        s = self.s_train[batch_idx,-1,:]
        s = np.reshape(s, (s.shape[0], s.shape[-1]))

        return tf.convert_to_tensor(seq_s, dtype=tf.float32), tf.convert_to_tensor(np.subtract(s_next, s), dtype=tf.float32)

    def get_val_batch(self, idx):
        batch_idx = self.indices_val[idx*self.batch_size:(idx+1)*self.batch_size]

        seq_s = self.s_val[batch_idx]
        s_next = self.s_next_val[batch_idx]
        s = self.s_val[batch_idx,-1,:]
        s = np.reshape(s, (s.shape[0], s.shape[-1]))

        return tf.convert_to_tensor(seq_s, dtype=tf.float32), tf.convert_to_tensor(np.subtract(s_next, s), dtype=tf.float32)

    def get_test_batch(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        seq_s = self.s[batch_idx]
        s_next = self.s_next[batch_idx]
        s = self.s[batch_idx,-1,:]
        s = np.reshape(s, (s.shape[0], s.shape[-1]))

        return tf.convert_to_tensor(seq_s, dtype=tf.float32), tf.convert_to_tensor(np.subtract(s_next, s), dtype=tf.float32), tf.convert_to_tensor(s_next, dtype=tf.float32)

if __name__ == '__main__':
    # dataloader = DataLoader(
    #     sim = 'LunarLander-v2',
    #     num = 0,
    #     target_cols=['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'w'],
    #     delta_t=0.2,
    #     batch_size=128
    # )

    dataloader = SeqDataLoader(
        sim = 'LunarLander-v2',
        num = 0,
        target_cols=['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'w'],
        seq = 10,
        delta_t=0.02,
        batch_size=128
    )

    dataloader.get_sequenec_data()
    dataloader.split_train_val(0.2)
    x, y = dataloader.get_train_batch(0)
    print(x.shape)