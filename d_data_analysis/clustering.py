from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        parser.add_argument("--dim", default=0, type=int, help="dimension of reduction vectors.")
        parser.add_argument("--c_method", default="kmeans", help="method of clustering.")
        parser.add_argument("--r_method", default="pca", help="method of dimension reduction.")
        args = parser.parse_args()

        self.sim = args.sim
        self.expert_num = args.expert_num
        self.data_num = args.data_num
        self.num_cluster = args.num_cluster
        self.dim = args.dim
        self.c_method = args.c_method
        self.r_method = args.r_method
    
    def get_save_path(self):
        num = len(glob(os.path.join(os.getcwd(), 'runs', 'clustering', self.sim, 'result_*')))
        self.save_path = os.path.join(os.getcwd(), 'runs', 'clustering', self.sim, f'result_{num}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_data(self):
        path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'cluster',f'data_{self.data_num}')
        self.delta_s = np.load(glob(os.path.join(path, 'delta_s*'))[0])
        self.action = np.load(glob(os.path.join(path, 'action*'))[0])

    def pca(self):
        pca = PCA(n_components=self.dim)
        pca.fit(self.delta_s)
        self.delta_s_r = pca.transform(self.delta_s)
    
    def tsne(self):
        tsne = TSNE(n_components=self.dim)
        tsne.fit(self.delta_s)
        self.delta_s_r = tsne.transform(self.delta_s)
    
    def kmeans_clustering(self):
        kmeans = KMeans(n_clusters=self.num_cluster)

        kmeans.fit(self.delta_s_r)
        self.result = kmeans.predict(self.delta_s_r)
        self.center_point = kmeans.cluster_centers_
    
    def agglomerative_clustering(self):
        agglomerative = AgglomerativeClustering(n_clusters=self.num_cluster)

        agglomerative.fit(self.delta_s)
        self.result = agglomerative.predict(self.delta_s_r)
        self.center_point = agglomerative.cluster_centers_
    
    def draw_cluster_plots(self):
        plt.figure(figsize=(8,6))
        plt.title(f"Cluster of delta_s_{self.r_method}")
        for n in range(self.num_cluster):
            idx = np.where(self.result == n)
            plt.scatter(self.delta_s_r[idx,0], self.delta_s_r[idx,1], s=20, label=f'cluster_{n}')
        plt.legend()

        plt.savefig(os.path.join(self.save_path, f'cluster_{self.c_method}.png'))
    
    def draw_action_plots(self):
        n_action = len(np.unique(self.action))
        plt.figure(figsize=(8,6))
        plt.title(f"Action of delta_s_{self.r_method}")
        for n in range(n_action):
            idx = np.where(self.action == n)
            plt.scatter(self.delta_s_r[idx,0], self.delta_s_r[idx,1], s=20, label=f'action_{n}')
        plt.legend()

        plt.savefig(os.path.join(self.save_path, 'distributed_by_action.png'))

    def save_result(self):
        dict_result = {}
        for d in range(self.delta_s.shape[-1]):
            dict_result[f'delta_s_{d}'] = self.delta_s[:,d]
        if self.dim != 0:
            for d_pca in range(self.dim):
                dict_result[f'delta_s_r_{d_pca}'] = self.delta_s_r[:, d_pca]
        dict_result['action'] = self.action
    
    def run(self):
        self.get_data()
        if self.dim != 0:
            if self.r_method == 'pca':
                self.pca()
            elif self.r_method == 'tsne':
                self.tsne()
        if self.c_method == 'kmeans':
            self.kmeans_clustering()
        elif self.c_method == 'agglomerative':
            self.agglomerative_clustering()
        self.draw_action_plots()
        self.draw_cluster_plots()
        self.save_result()

if __name__=='__main__':
    cluster = Cluster()
    cluster.run()

# python d_data_analysis\clustering.py --sim LunarLander --expert_num 0 --data_num 0 --num_cluster 4 --dim 2 --method kmeans