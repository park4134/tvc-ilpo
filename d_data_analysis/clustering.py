from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from glob import glob

import numpy as np
import os

class Cluster:
    def __init__(self, sim, expert_num, data_num):
        self.sim = sim
        self.expert_num = expert_num
        self.data_num = data_num
    
    def get_save_path(self):
        num = len(glob(os.path.join(os.getcwd(), 'runs', 'clustering', self.sim, 'result_*')))
        self.save_path = os.path.join(os.getcwd(), 'runs', 'clustering', self.sim, f'result_{num}')
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_data(self):
        path = os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', f'data_{self.data_num}')
        self.delta_s = np.load(glob(os.path.join(path, 'delta_s*'))[0])
        self.action = np.load(glob(os.path.join(path, 'action*'))[0])

    def pca(self, num_components):
        pca = PCA(n_components=num_components)
        pca.fit(self.delta_s)
        self.delta_s_pca = pca.transform(self.delta_s)
    
    def kmeans_clustering(self, num_k):
        kmeans = KMeans(n_clusters=num_k)

        kmeans.fit(self.delta_s)
        self.result = kmeans.predict(self.delta_s_pca)
        self.center_point = kmeans.cluster_centers_
    
