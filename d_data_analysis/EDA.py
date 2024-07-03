from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import os

class EDA:
    def __init__(self, sim, expert_num, data_num, target_cols):
        self.sim = sim
        self.expert_num = expert_num
        self.data_num = data_num
        self.target_cols = target_cols
        self.get_data()
        self.get_save_path()
        self.dict_param = {}
        for p in range(len(self.target_cols)):
            self.dict_param[p] = self.target_cols[p]
    
    def get_data(self):
        base_path = os.path.join(os.getcwd(), 'data', 'preprocessed', 'LunarLander-v2', f'expert_data_{self.expert_num}', 'cluster', f'data_{self.data_num}')
        self.action = np.load(glob(os.path.join(base_path, 'action*'))[0])
        self.delta_s = np.load(glob(os.path.join(base_path, 'delta_s*'))[0])
        self.s = np.load(glob(os.path.join(base_path, 's*'))[0])
        self.s_next = np.load(glob(os.path.join(base_path, 's_next*'))[0])
    
    def get_save_path(self):
        num = len(glob(os.path.join(os.getcwd(), 'runs', 'eda', self.sim, 'result_*')))
        self.save_path = os.path.join(os.getcwd(), 'runs', 'eda', self.sim, f'result_{num}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def deltas_action_plot(self):
        for i in range(len(self.dict_param)):
            for j in range(len(self.dict_param)):
                if i == j:
                    continue
                else:
                    param_idx = [i, j]
                    plt.figure(figsize=(16,12))
                    n_action = len(np.unique(self.action))
                    xlim = round(max(np.max(self.s[:, param_idx[0]]), abs(np.min(self.s[:, param_idx[0]]))), 2)
                    ylim = round(max(np.max(self.s[:, param_idx[1]]), abs(np.min(self.s[:, param_idx[1]]))), 2)
                    for n in range(n_action):
                        plt.subplot(2,2,n+1)
                        idx = np.where(self.action == n)
                        plt.scatter(self.delta_s[:, param_idx][idx,0], self.delta_s[:, param_idx][idx,1], s=10, alpha=0.05, label=f'action_{n}', color = plt.cm.tab10(n))
                        plt.legend(fontsize=14)
                        plt.xlim(-xlim, xlim)
                        plt.ylim(-ylim, ylim)
                        plt.xlabel(self.dict_param[param_idx[0]], fontsize=14)
                        plt.ylabel(self.dict_param[param_idx[1]], fontsize=14)
                    
                    plt.savefig(os.path.join(self.save_path, f'deltas_action_{self.dict_param[param_idx[0]]}_{self.dict_param[param_idx[1]]}.png'))
                    plt.close()

    def s_action_plot(self):
        for i in range(len(self.dict_param)):
            for j in range(len(self.dict_param)):
                if i == j:
                    continue
                else:
                    param_idx = [i, j]
                    plt.figure(figsize=(16,12))
                    n_action = len(np.unique(self.action))
                    xlim = round(max(np.max(self.s[:, param_idx[0]]), abs(np.min(self.s[:, param_idx[0]]))), 2)
                    ylim = round(max(np.max(self.s[:, param_idx[1]]), abs(np.min(self.s[:, param_idx[1]]))), 2)
                    for n in range(n_action):
                        plt.subplot(2,2,n+1)
                        idx = np.where(self.action == n)
                        plt.scatter(self.s[:, param_idx][idx,0], self.s[:, param_idx][idx,1], s=10, alpha=0.05, label=f'action_{n}', color = plt.cm.tab10(n))
                        plt.legend(fontsize=14)
                        plt.xlim(-xlim, xlim)
                        plt.ylim(-ylim, ylim)
                        plt.xlabel(self.dict_param[param_idx[0]], fontsize=14)
                        plt.ylabel(self.dict_param[param_idx[1]], fontsize=14)
                    
                    plt.savefig(os.path.join(self.save_path, f's_action_{self.dict_param[param_idx[0]]}_{self.dict_param[param_idx[1]]}.png'))
                    plt.close()

if __name__ == '__main__':
    eda = EDA(
        sim='LunarLander-v2',
        expert_num=2,
        data_num=1,
        target_cols=['pos_x', 'pos_y', 'v_x', 'v_y', 'angle', 'w']
    )

    eda.deltas_action_plot()
    eda.s_action_plot()