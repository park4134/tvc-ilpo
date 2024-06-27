import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from glob import glob
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import json
import os

class Preprocessor():
    def __init__(self, sim, num, cycle_time, delta_t, target_cols, is_normalize=False, is_standardize=False):
        self.sim = sim
        self.num = num
        self.cycle_time = cycle_time
        self.delta_t = delta_t
        self.target_cols = target_cols
        self.is_normalize = is_normalize
        self.is_standardize = is_standardize
        self.patience = int(self.delta_t / self.cycle_time)
        self.get_save_path()
    
    def get_save_path(self):
        number = len(glob(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.num}', 'policy', 'data*')))
        self.save_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.num}', 'policy', f'data_{number}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def get_data(self):
        self.data_paths = glob(os.path.join(os.getcwd(), 'data', 'row', self.sim, f'expert_data_{self.num}', 'senario*'))
        self.s = np.array([])
        self.s_next = np.array([])

        for p in tqdm(self.data_paths):
            with open(p, 'r') as f:
                dict_r = json.load(f)
            s_, s_next_ = self.get_s_AND_s_next(dict_r)

            if len(self.s) == 0:
                self.s = s_
                self.s_next = s_next_
            else:
                self.s = np.concatenate((self.s, s_), axis=0)
                self.s_next = np.concatenate((self.s_next, s_next_), axis=0)

        self.indices = np.arange(len(self.s))

        if self.is_normalize:
            self.min = np.array([-1.5, -1.5, -5., -5., -3.14, -5.])
            self.max = np.array([1.5, 1.5, 5., 5., 3.14, 5.])
            # self.min = np.min(self.s, axis=0)
            # self.max = np.max(self.s, axis=0)

            self.s = np.divide(np.subtract(self.s, self.min), np.subtract(self.max, self.min))
            self.s_next = np.divide(np.subtract(self.s_next, self.min), np.subtract(self.max, self.min))
        
        elif self.is_standardize:
            self.mean = np.mean(self.s, axis=0)
            self.std = np.std(self.s, axis=0)

            self.s = np.divide(np.subtract(self.s, self.std), self.mean)
            self.s_next = np.divide(np.subtract(self.s_next, self.std), self.mean)

    def get_s_AND_s_next(self, dict_result):
        self.patience = int(self.delta_t / self.cycle_time)

        data = []
        for col in self.target_cols:
            data.append(dict_result[col])
        
        data = np.transpose(np.array(data, dtype=np.float32))

        return deepcopy(data[:len(data)-self.patience,:]), deepcopy(data[self.patience:,:])
    
    def save_data(self):
        np.save(os.path.join(self.save_path, f's_{self.delta_t}.npy'), self.s)
        np.save(os.path.join(self.save_path, f's_next_{self.delta_t}.npy'), self.s_next)
        if self.is_normalize:
            np.save(os.path.join(self.save_path, f'min_{self.delta_t}.npy'), self.min)
            np.save(os.path.join(self.save_path, f'max_{self.delta_t}.npy'), self.max)
        if self.is_standardize:
            np.save(os.path.join(self.save_path, f'mean_{self.delta_t}.npy'), self.mean)
            np.save(os.path.join(self.save_path, f'std_{self.delta_t}.npy'), self.std)

class SeqPreprocessor():
    def __init__(self, sim, num, seq, cycle_time, delta_t, target_cols, is_normalize=False, is_standardize=False):
        self.sim = sim
        self.num = num
        self.seq = seq
        self.cycle_time = cycle_time
        self.delta_t = delta_t
        self.target_cols = target_cols
        self.is_normalize = is_normalize
        self.is_standardize = is_standardize
        self.patience = int(self.delta_t / self.cycle_time)
        self.get_save_path()
    
    def get_save_path(self):
        number = len(glob(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.num}', 'policy', 'data*')))
        self.save_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.num}', 'policy', f'data_{number}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_data(self):
        self.data_paths = glob(os.path.join(os.getcwd(), 'data', 'row', self.sim, f'expert_data_{self.num}', 'senario*'))
        self.s = np.array([])
        self.s_next = np.array([])

        for p in tqdm(self.data_paths):
            with open(p, 'r') as f:
                dict_r = json.load(f)
            # print(f"dict length : {len(dict_r['v_x'])}")
            s_, s_next_ = self.get_s_AND_s_next(dict_r)
            # print(f"s length : {len(s_)}")
            # print(f"s_next length : {len(s_next_)}")
            if len(self.s) == 0:
                self.s = s_
                self.s_next = s_next_
            else:
                self.s = np.concatenate((self.s, s_), axis=0)
                self.s_next = np.concatenate((self.s_next, s_next_), axis=0)
        
        self.indices = np.arange(len(self.s))

        if self.is_normalize:
            # self.min = np.array([-1.5, -1.5, -5., -5., -3.14, -5.])
            # self.max = np.array([1.5, 1.5, 5., 5., 3.14, 5.])
            s_origin = np.reshape(self.s[:, -1, :], (self.s.shape[0], self.s.shape[-1]))
            self.min = np.min(s_origin, axis=0)
            self.max = np.max(s_origin, axis=0)

            self.s = np.divide(np.subtract(self.s, self.min), np.subtract(self.max, self.min))
            self.s_next = np.divide(np.subtract(self.s_next, self.min), np.subtract(self.max, self.min))
        
        elif self.is_standardize:
            self.mean = np.mean(self.s, axis=0)
            self.std = np.std(self.s, axis=0)

            self.s = np.divide(np.subtract(self.s, self.std), self.mean)
            self.s_next = np.divide(np.subtract(self.s_next, self.std), self.mean)

    
    def get_s_AND_s_next(self, dict_result):
        data = np.array([])
        for s in range(len(dict_result['v_x']) - self.seq + 1):
            sequence = []
            for col in self.target_cols:
                sequence.append(dict_result[col][s:s + self.seq])
            sequence = np.transpose(np.array(sequence, dtype=np.float32))
            if len(data) == 0:
                data = np.expand_dims(sequence, axis=0)
            else:
                data = np.concatenate((data, np.expand_dims(sequence, axis=0)), axis=0)
        # print(data.shape)

        return deepcopy(data[:len(data)-self.patience,:]), deepcopy(np.reshape(data[self.patience:,-1,:], (len(data)-self.patience, data.shape[-1])))
    
    def save_data(self):
        print(f"S : {self.s.shape} , S_next : {self.s_next.shape}")
        np.save(os.path.join(self.save_path, f'seq{self.seq}_s_{self.delta_t}.npy'), self.s)
        np.save(os.path.join(self.save_path, f'seq{self.seq}_s_next_{self.delta_t}.npy'), self.s_next)
        if self.is_normalize:
            np.save(os.path.join(self.save_path, f'seq{self.seq}_min_{self.delta_t}.npy'), self.min)
            np.save(os.path.join(self.save_path, f'seq{self.seq}_max_{self.delta_t}.npy'), self.max)
        if self.is_standardize:
            np.save(os.path.join(self.save_path, f'seq{self.seq}_mean_{self.delta_t}.npy'), self.mean)
            np.save(os.path.join(self.save_path, f'seq{self.seq}_std_{self.delta_t}.npy'), self.std)

class ClusterPreprocessor(Preprocessor):
    def __init__(self, sim, num, cycle_time, delta_t, target_cols, is_normalize=False, is_standardize=False):
        self.sim = sim
        self.num = num
        self.cycle_time = cycle_time
        self.delta_t = delta_t
        self.target_cols = target_cols
        self.is_normalize = is_normalize
        self.is_standardize = is_standardize
        self.patience = int(self.delta_t / self.cycle_time)
        self.get_save_path()

    def get_save_path(self):
        number = len(glob(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.num}', 'cluster', 'data*')))
        self.save_path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.num}', 'cluster', f'data_{number}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_data(self):
        self.data_paths = glob(os.path.join(os.getcwd(), 'data', 'row', self.sim, f'expert_data_{self.num}', 'senario*'))
        self.s = np.array([])
        self.s_next = np.array([])
        self.a = np.array([])

        for p in tqdm(self.data_paths):
            with open(p, 'r') as f:
                dict_r = json.load(f)
            s_, s_next_, a_ = self.get_s_AND_s_next(dict_r)

            if len(self.s) == 0:
                self.s = s_
                self.s_next = s_next_
                self.a = a_
            else:
                self.s = np.concatenate((self.s, s_), axis=0)
                self.s_next = np.concatenate((self.s_next, s_next_), axis=0)
                self.a = np.concatenate((self.a, a_), axis=0)
        
        if self.is_normalize:
            self.min = np.array([-1.5, -1.5, -5., -5., -3.14, -5.], dtype = np.float32)
            self.max = np.array([1.5, 1.5, 5., 5., 3.14, 5.], dtype = np.float32)
            # self.min = np.min(self.s, axis=0)
            # self.max = np.max(self.s, axis=0)

            self.s = np.divide(np.subtract(self.s, self.min), np.subtract(self.max, self.min))
            self.s_next = np.divide(np.subtract(self.s_next, self.min), np.subtract(self.max, self.min))
        
        elif self.is_standardize:
            self.mean = np.mean(self.s, axis=0)
            self.std = np.std(self.s, axis=0)

            self.s = np.divide(np.subtract(self.s, self.std), self.mean)
            self.s_next = np.divide(np.subtract(self.s_next, self.std), self.mean)
        
        self.delta_s = np.subtract(self.s_next, self.s)
    
    def get_s_AND_s_next(self, dict_result):
        data = [dict_result[col] for col in self.target_cols]
        action = dict_result["action"]

        data = np.transpose(np.array(data, dtype=np.float32))
        action = np.array(action, dtype=np.float32).reshape(-1)

        return deepcopy(data[:len(data)-self.patience,:]), deepcopy(data[self.patience:,:]), deepcopy(action)

    def save_data(self):
        np.save(os.path.join(self.save_path, f'delta_s_{self.delta_t}.npy'), self.delta_s)
        np.save(os.path.join(self.save_path, f'action_{self.delta_t}_{self.patience}.npy'), self.a)
        if self.is_normalize:
            np.save(os.path.join(self.save_path, f'min_{self.delta_t}.npy'), self.min)
            np.save(os.path.join(self.save_path, f'max_{self.delta_t}.npy'), self.max)
        if self.is_standardize:
            np.save(os.path.join(self.save_path, f'mean_{self.delta_t}.npy'), self.mean)
            np.save(os.path.join(self.save_path, f'std_{self.delta_t}.npy'), self.std)

if __name__ == "__main__":
    preprocessor = Preprocessor(sim = 'LunarLander-v2',
                                num = 0,
                                cycle_time = 0.02,
                                delta_t = 0.02,
                                target_cols = ['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'w'],
                                is_normalize = True
                                )
    
    # preprocessor = SeqPreprocessor(sim = 'LunarLander-v2',
    #                             num = 1,
    #                             seq = 10,
    #                             cycle_time = 0.02,
    #                             delta_t = 0.02,
    #                             target_cols = ['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'w'],
    #                             is_normalize = True
    #                             )

    # preprocessor = ClusterPreprocessor(sim = 'LunarLander-v2',
    #                             num = 0,
    #                             cycle_time = 0.02,
    #                             delta_t = 0.02,
    #                             target_cols = ['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'w']
    #                             )
    
    preprocessor.get_data()
    preprocessor.save_data()