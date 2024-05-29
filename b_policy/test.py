import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import PolicyNetwork, SeqPolicyNetwork
from utils import gpu_limit, get_test_metric
from tensorflow.keras.optimizers import Adam
from DataLoader import DataLoader, SeqDataLoader
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import yaml

class tester():
    def __init__(self):
        self.config_dic = self.get_config()
        self.n_state = self.config_dic['n_state']
        self.n_latent_action = self.config_dic['n_latent_action']
        self.batch_size = self.config_dic['batch_size']
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.lrelu = self.config_dic['lrelu']
        self.seq = self.config_dic['seq']
        self.sim = self.config_dic['sim']
        self.delta_t = self.config_dic['dt']
        self.target_cols = self.config_dic['target_cols']

        self.get_save_path()

        if self.seq == 1:
            self.dataloader = DataLoader(
                sim=self.sim,
                num=1,
                target_cols=self.target_cols,
                delta_t=self.delta_t,
                batch_size=self.batch_size,
            )

            self.model = PolicyNetwork(
                n_state=self.n_state,
                n_latent_action=self.n_latent_action,
                units = self.units,
                layer_num=self.layer_num,
                batch_size=self.batch_size,
                lrelu=self.lrelu,
            )
            self.model.build_graph()
            self.model.load_weights(os.path.join(os.path.dirname(self.config_path), 'best_weights'))
        
        else:
            self.dataloader = SeqDataLoader(
                sim=self.sim,
                num=1,
                target_cols=self.target_cols,
                seq = self.seq,
                delta_t=self.delta_t,
                batch_size=self.batch_size,
            )

            self.model = SeqPolicyNetwork(
                n_state=self.n_state,
                n_latent_action=self.n_latent_action,
                seq = self.seq,
                units = self.units,
                layer_num=self.layer_num,
                batch_size=self.batch_size,
                lrelu=self.lrelu,
            )
            self.model.build_graph()
            self.model.load_weights(os.path.join(os.path.dirname(self.config_path), 'best_weights'))

        self.test_metric = np.array([])
        self.test_relative_metric = np.array([])
        self.test_s = np.array([])
        self.test_s_next = np.array([])
        self.test_z = np.array([])
        self.test_delta_s_hat = np.array([])

    def get_config(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--sim", default="LunarLander-v2", help="simulator of trained model.")
        parser.add_argument("--model_dir", default="model_0", help="yaml file of trained model.")
        args = parser.parse_args()

        self.config_path = os.path.join(os.getcwd(), 'runs', 'policy', args.sim, 'train', args.model_dir, 'config.yaml')
        with open(self.config_path) as f:
            config_dic = yaml.load(f, Loader=yaml.FullLoader)
        
        self.sim = args.sim
        self.model_dir = args.model_dir

        return deepcopy(config_dic)

    def get_save_path(self):
        self.save_path = os.path.join(os.getcwd(), 'runs', 'policy', self.sim, 'test', self.model_dir)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def serialize(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    def save_test_results(self):
        print(self.test_metric.shape)
        print(self.test_z.shape)
        print(self.test_s.shape)
        print(self.test_s_next.shape)
        print(self.test_relative_metric.shape)
        df_test = pd.DataFrame({
            f'test_s_{self.target_cols[0]}' : self.test_s[:,0],
            f'test_s_{self.target_cols[1]}' : self.test_s[:,1],
            f'test_s_{self.target_cols[2]}' : self.test_s[:,2],
            f'test_s_{self.target_cols[3]}' : self.test_s[:,3],
            f'test_s_{self.target_cols[4]}' : self.test_s[:,4],
            f'test_s_{self.target_cols[5]}' : self.test_s[:,5],
            f'test_s_next_{self.target_cols[0]}' : self.test_s_next[:,0],
            f'test_s_next_{self.target_cols[1]}' : self.test_s_next[:,1],
            f'test_s_next_{self.target_cols[2]}' : self.test_s_next[:,2],
            f'test_s_next_{self.target_cols[3]}' : self.test_s_next[:,3],
            f'test_s_next_{self.target_cols[4]}' : self.test_s_next[:,4],
            f'test_s_next_{self.target_cols[5]}' : self.test_s_next[:,5],
            'test_metric' : self.test_metric,
            'test_z' : self.test_z,
            f'rel_metric_{self.target_cols[0]}' : self.test_relative_metric[:,0],
            f'rel_metric_{self.target_cols[1]}' : self.test_relative_metric[:,1],
            f'rel_metric_{self.target_cols[2]}' : self.test_relative_metric[:,2],
            f'rel_metric_{self.target_cols[3]}' : self.test_relative_metric[:,3],
            f'rel_metric_{self.target_cols[4]}' : self.test_relative_metric[:,4],
            f'rel_metric_{self.target_cols[5]}' : self.test_relative_metric[:,5]
                    })
        # print(df_test.head())


        df_test.to_csv(os.path.join(os.path.dirname(self.save_path), self.model_dir, f'test_result_p{int(self.delta_t / 0.02)}_z{self.n_latent_action}.csv'))

    def test(self):
        self.dataloader.get_data()

        '''Test step'''
        for j in tqdm(range(self.dataloader.len)):
            s, delta_s, s_next = self.dataloader.get_test_batch(j)

            z_p, delta_s_hat = self.model(s, training=False)
            max_z = tf.argmax(z_p, axis=-1)
            max_z_p = tf.expand_dims(tf.one_hot(max_z, z_p.shape[-1]), axis=-1)

            max_delta_s_hat = tf.reduce_sum(tf.multiply(delta_s_hat, max_z_p), axis=1)

            metric, relative_metric = get_test_metric(delta_s, z_p, delta_s_hat)

            if self.seq > 1:
                s = s[:,-1,:]
                s = np.reshape(s, (s.shape[0], s.shape[-1]))
                # print(s.shape)

            if len(self.test_metric) == 0:
                self.test_metric = metric
                self.test_relative_metric = relative_metric
                self.test_s = s
                self.test_s_next = s_next
                self.test_z = np.array(max_z)
                self.test_delta_s_hat = np.array(max_delta_s_hat)

            else:
                self.test_metric = np.concatenate((self.test_metric, metric), axis=0)
                self.test_relative_metric = np.concatenate((self.test_relative_metric, relative_metric), axis=0)
                self.test_s = np.concatenate((self.test_s, s), axis=0)
                self.test_s_next = np.concatenate((self.test_s_next, s_next), axis=0)
                self.test_z = np.concatenate((self.test_z, np.array(max_z)), axis=0)
                self.test_delta_s_hat = np.concatenate((self.test_delta_s_hat, np.array(max_delta_s_hat)), axis=0)
        # print(self.test_s.shape)

        self.save_test_results()

        print(f"Test metric : {np.mean(self.test_metric):.5f}")
        print(f"Test relative metric : {self.test_relative_metric}")

if __name__=="__main__":
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    gpu_limit(4)
    np.random.seed(42)

    Tester = tester()

    Tester.test()