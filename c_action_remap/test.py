import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from b_policy.model import PolicyNetwork
from model import ActionRemapNetwork
from utils import get_loss_map, gpu_limit
from tensorflow.keras.optimizers import Adam
from collections import deque
from copy import deepcopy
from glob import glob
from tqdm import tqdm

import tensorflow as tf
import gymnasium as gym
import numpy as np
import random
import json
import yaml
import math

class tester():
    def __init__(self):
        self.get_config()
        self.get_save_path()
        self.env = gym.make(self.sim, render_mode="rgb_array")
        self.n_action = self.env.action_space.n
        self.get_models()

        if self.is_normalize:
            self.min = np.load(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}', f'min_{self.delta_t}.npy'))
            self.max = np.load(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}', f'max_{self.delta_t}.npy'))
            self.min = tf.convert_to_tensor(self.min, dtype=tf.float32)
            self.max = tf.convert_to_tensor(self.max, dtype=tf.float32)
            # self.min = tf.constant([-1.5, -1.5, -5., -5., -3.14, -5.], dtype=tf.float32)
            # self.max = tf.constant([1.5, 1.5, 5., 5., 3.14, 5.], dtype=tf.float32)
        self.test_z = []
        self.test_terminated = []
        self.test_total_score = []
        self.test_avg_metric = []
        self.test_action = []

        self.force = 0.001
        self.gravity = 0.0025
    
    def get_config(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--sim", default="Acrobot-v1", help="environment for test.")
        parser.add_argument("--model", default="model_0", help="model directory for test.")
        parser.add_argument("--episodes", default=10, type=int, help="model directory for test.")
        args = parser.parse_args()
        self.sim = args.sim
        self.model_arm_dir = args.model
        self.n_episodes = args.episodes

        self.config_path = os.path.join(os.getcwd(), 'runs', 'action_remap', self.sim, 'train', self.model_arm_dir, 'config.yaml')
        with open(self.config_path) as f:
            self.config_dic = yaml.load(f, Loader=yaml.FullLoader)
        
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.max_patience = self.config_dic['patience']
        self.model_lpn_dir = self.config_dic['model_name']
        
        self.config_path_p = os.path.join(os.getcwd(), 'runs', 'policy', self.sim, 'train', self.model_lpn_dir, 'config.yaml')
        with open(self.config_path_p) as f:
            self.config_dic_p = yaml.load(f, Loader=yaml.FullLoader)
        
        self.n_state = self.config_dic_p['n_state']
        self.n_latent_action = self.config_dic_p['n_latent_action']
        self.units_p = self.config_dic_p['units']
        self.layer_num_p = self.config_dic_p['layer_num']
        self.lrelu = self.config_dic_p['lrelu']
        self.expert_num = self.config_dic_p['expert_num']
        self.data_num = self.config_dic_p['data_num']
        self.delta_t = self.config_dic_p['dt']
        self.is_normalize = self.config_dic_p['is_normalize']
        self.is_standardize = self.config_dic_p['is_standardize']
    
    def get_save_path(self):
        self.save_path = os.path.join(os.getcwd(), 'runs', 'action_remap', self.sim, 'test', self.model_arm_dir)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_models(self):
        self.model_lpn = PolicyNetwork(
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units=self.units_p,
            layer_num=self.layer_num_p,
            batch_size=1,
            lrelu=self.lrelu
        )
        self.model_lpn.build_graph()
        self.model_lpn.load_weights(os.path.join(os.path.dirname(self.config_path_p), 'best_weights'))

        self.model_arm = ActionRemapNetwork(
            n_action=self.n_action,
            units=self.units,
            layer_num=self.layer_num,
            batch_size=1,
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units_p=self.units_p,
            layer_num_p=self.layer_num_p,
            lrelu=self.lrelu
        )
        self.model_arm.build_graph()
        # self.model_arm.load_weights(os.path.join(os.path.dirname(self.config_path), 'best_weights_metric'))
        self.model_arm.load_weights(os.path.join(os.path.dirname(self.config_path), 'best_weights_score'))
    
    def serialize(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
    def save_test_results(self):
        dict_test = {
            'z' : self.test_z,
            'terminated' : self.test_terminated,
            'score' : self.test_total_score,
            'metric' : self.test_avg_metric,
            'n_latent_action' : self.n_latent_action,
            'action' : self.test_action
        }

        with open(os.path.join(self.save_path, 'test_results.json'), 'w') as f:
            json.dump(dict_test, f, default=self.serialize)

    def env_init(self, n):
        self.done = False
        if self.sim == 'LunarLander-v2':
            self.state = self.env.reset()[0][:6]
        # elif self.sim == 'MountainCar-v0':
        #     obs = self.env.reset()[0]
        #     self.state = np.concatenate((obs, [0.0]))
        else:
            self.state = self.env.reset()[0]
        self.video = VideoRecorder(self.env, path=os.path.join(self.save_path,f"video_{n}.mp4"))
        self.env.render()
    
    def test_step(self):
        self.video.capture_frame()

        if self.is_normalize:
            self.state = tf.divide(tf.subtract(self.state, self.min), tf.subtract(self.max, self.min))
        
        self.z_p, delta_s_hat = self.model_lpn(tf.expand_dims(self.state, axis=0)) # z_p : (batch, n_latent_action) / delta_s_hat : (batch, n_latent_action, n_state)
        max_z = tf.one_hot(tf.argmax(self.z_p, axis=-1), self.n_latent_action)
        max_delta_s_hat = tf.reshape(delta_s_hat, (self.n_latent_action, self.n_state))[tf.argmax(tf.reshape(self.z_p, (self.n_latent_action,)), axis=-1).numpy()]

        action_mapped = self.model_arm((tf.expand_dims(self.state, axis=0), max_z))
        action = tf.argmax(tf.reshape(action_mapped, (self.n_action, )), axis=-1).numpy()
        print(action_mapped)

        next_state, reward, self.terminated, truncated, info = self.env.step(action)
        self.env.render()
        self.done = self.terminated or truncated

        if self.sim == 'LunarLander-v2':
            next_state = next_state[:6]
        # elif self.sim == 'MountainCar-v0':
        #     acc = (action - 1) * self.force + math.cos(3 * self.state[0]) * (-self.gravity)
        #     next_state = np.concatenate((next_state, [acc]))
        
        if self.is_normalize:
            next_state = tf.divide(tf.subtract(next_state, self.min), tf.subtract(self.max, self.min))
        
        metric = tf.subtract(next_state, max_delta_s_hat)
        metric = tf.reduce_mean(tf.square(metric))

        self.state = next_state
        self.test_score += reward
        self.test_metric.append(metric)
        self.test_z.append(int(tf.argmax(tf.reshape(self.z_p, (self.n_latent_action,)), axis=-1).numpy()))
        self.test_action.append(int(action))
    
    def test(self):
        for n in range(self.n_episodes):
            self.env_init(n)

            self.test_score = 0
            self.test_metric = []

            while not self.done:
                self.test_step()
            
            print(f'Episode {n} :\tTest Score >> {self.test_score:.4f}')
            if self.terminated:
                self.test_terminated.append(1)
            else:
                self.test_terminated.append(0)
            
            self.test_avg_metric.append(np.mean(self.test_metric))
            self.test_total_score.append(self.test_score)

            self.env.close()
            self.video.close()

        self.save_test_results()

if __name__ == '__main__':
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    gpu_limit(2)
    np.random.seed(42)

    Tester = tester()

    Tester.test()