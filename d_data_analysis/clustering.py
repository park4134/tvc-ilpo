from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import os

class Cluster:
    def __init__(self):
        self.get_configs()
        
        self.get_save_path()
    
    def get_configs(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--sim", default="LunarLander-v2", help="env of data for clustering.")
        parser.add_argument("--expert_num", default=0, type=int, help="expert data info.")
        parser.add_argument("--data_num", default=0, type=int, help="data number.")
        parser.add_argument("--num_cluster", default=4, type=int, help="number of cluster.")
        parser.add_argument("--pca_dim", default=0, type=int, help="dimension of pca vectors.")
        parser.add_argument("--method", default="kmeans", help="method of clustering.")
        args = parser.parse_args()

        self.sim = args.sim
        self.expert_num = args.expert_num
        self.data_num = args.data_num
        self.num_cluster = args.num_cluster
        self.pca_dim = args.pca_dim
        self.method = args.method
    
    def get_save_path(self):
        num = len(glob(os.path.join(os.getcwd(), 'runs', 'clustering', self.sim, 'result_*')))
        self.save_path = os.path.join(os.getcwd(), 'runs', 'clustering', self.sim, f'result_{num}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_data(self):
        path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', f'data_{self.data_num}')
        self.delta_s = np.load(glob(os.path.join(path, 'delta_s*'))[0])
        self.action = np.load(glob(os.path.join(path, 'action*'))[0])

    def pca(self):
        pca = PCA(n_components=self.pca_dim)
        pca.fit(self.delta_s)
        self.delta_s_pca = pca.transform(self.delta_s)
    
    def kmeans_clustering(self):
        kmeans = KMeans(n_clusters=self.num_cluster)

        kmeans.fit(self.delta_s)
        self.result = kmeans.predict(self.delta_s_pca)
        self.center_point = kmeans.cluster_centers_
    
    def agglomerative_clustering(self):
        agglomerative = AgglomerativeClustering(n_clusters=self.num_cluster)

        agglomerative.fit(self.delta_s)
        self.result = agglomerative.predict(self.delta_s_pca)
        self.center_point = agglomerative.cluster_centers_
    
    def draw_cluster_plots(self):
        plt.figure(figsize=(8,6))
        for n in range(self.num_cluster):
            idx = np.where(self.result == n)
            plt.scatter(self.delta_s_pca[idx], label=f'cluster_{n}')

        plt.savefig(os.path.join(self.save_path, 'cluster_{self.mode}.png'))
    
    def draw_action_plots(self):
        n_action = len(np.unique(self.action))
        plt.figure(figsize=(8,6))
        for n in range(n_action):
            idx = np.where(self.action == n)
            plt.scatter(self.delta_s_pca[idx], label=f'action_{n}')

        plt.savefig(os.path.join(self.save_path, 'distributed_by_action.png'))

    def save_result(self):
        dict_result = {}
        for d in range(self.delta_s.shape[-1]):
            dict_result[f'delta_s_{d}'] = self.delta_s[:,d]
        if self.pca_dim != 0:
            for d_pca in range(self.pca_dim):
                dict_result[f'delta_s_pca_{d_pca}'] = self.delta_s_pca[:, d_pca]
        dict_result['action'] = self.action
    
    def run(self):
        self.get_data()
        if self.pca_dim != 0:
            self.pca()
        if self.mode == 'kmeans':
            self.kmeans_clustering()
        elif self.mode == 'agglomerative':
            self.agglomerative_clustering()
        self.draw_action_plots()
        self.draw_cluster_plots()
        self.save_result()

if __name__=='__main__':
    cluster = Cluster()
    cluster.run()

# python d_data_analysis\clustering.py --sim LunarLander --expert_num 0 --data_num 0 --num_cluster 4 --pca_dim 2 --method kmeans