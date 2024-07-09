import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

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

class trainer():
    def __init__(self):
        self.get_config()
        self.get_save_path()
        self.env = gym.make(self.sim, render_mode="rgb_array")
        self.n_action = self.env.action_space.n
        print(self.n_action)
        self.get_models()

        if self.is_normalize:
            self.min = np.load(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}', f'min_{self.delta_t}.npy'))
            self.max = np.load(os.path.join(os.getcwd(), 'data', 'preprocessed', self.sim, f'expert_data_{self.expert_num}', 'policy', f'data_{self.data_num}', f'max_{self.delta_t}.npy'))
            self.min = tf.convert_to_tensor(self.min, dtype=tf.float32)
            self.max = tf.convert_to_tensor(self.max, dtype=tf.float32)
            # self.min = tf.constant([-1.5, -1.5, -5., -5., -3.14, -5.], dtype=tf.float32)
            # self.max = tf.constant([1.5, 1.5, 5., 5., 3.14, 5.], dtype=tf.float32)
        
        # if self.is_standardize

        self.train_loss_step = []
        self.train_episode = []

        self.train_loss = []
        self.train_total_score = []

        self.val_loss_step = []
        self.val_z = []
        self.val_episode = []

        self.val_terminated = []
        self.val_loss = []
        self.val_total_score = [-10000]
        self.val_avg_metric = [np.inf]
    
    def get_config(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config.yaml", help="yaml file for train configuration.")
        parser.add_argument("--model", default="model_0", help="model directory for policy network.")
        args = parser.parse_args()

        config_path = os.path.join(os.getcwd(), 'c_action_remap', 'configs', args.config)
        with open(config_path) as f:
            self.config_dic = yaml.load(f, Loader=yaml.FullLoader)
        
        self.sim = self.config_dic['sim']
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.batch_size = self.config_dic['batch_size']
        self.learning_rate = self.config_dic['learning_rate']
        self.epochs = self.config_dic['epochs']
        self.max_patience = self.config_dic['patience']
        self.buffer = deque(maxlen=self.config_dic['buffer_size'])
        self.min_buffer_size = self.config_dic['min_buffer_size']
        self.e = self.config_dic['e_init']
        self.e_decay_factor = self.config_dic['e_decay_factor']
        self.e_min = self.config_dic['e_min']
        
        self.config_path_p = os.path.join(os.getcwd(), 'runs', 'policy', self.sim, 'train', args.model, 'config.yaml')
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
        number = len(glob(os.path.join(os.getcwd(), 'runs', 'action_remap', self.sim, 'train', 'model_*')))
        save_dir = os.path.join(os.getcwd(), 'runs', 'action_remap', self.sim, 'train', f'model_{number}')
        self.save_path = save_dir

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_models(self):
        self.model_lpn = PolicyNetwork(
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units=self.units_p,
            layer_num=self.layer_num_p,
            batch_size=self.batch_size,
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
    
    def save_config(self):
        self.config_dic['model_name'] = self.config_path_p.split(os.sep)[-2]
        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config_dic, f)
    
    def serialize(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
    def save_train_results(self):
        self.save_config()

        dict_train = {
            'loss_step' : self.train_loss_step,
            'episode' : self.train_episode,
            'loss' : self.train_loss,
            'score' : self.train_total_score
        }

        dict_val = {
            'loss_step' : self.val_loss_step,
            'episode' : self.val_episode,
            'z' : self.val_z,

            'terminated' : self.val_terminated,
            'score' : self.val_total_score[1:],
            'metric' : self.val_avg_metric[1:],
            'loss' : self.val_loss
        }

        with open(os.path.join(self.save_path, 'train_results.json'), 'w') as f:
            json.dump(dict_train, f, default=self.serialize)

        with open(os.path.join(self.save_path, 'validation_results.json'), 'w') as f:
            json.dump(dict_val, f, default=self.serialize)
    
    def e_greedy_action(self, state, max_z):
        if np.random.uniform(0, 1) <= self.e:
            action = np.random.choice(np.arange(self.n_action))
            action_mapped = tf.one_hot(action, self.n_action)
        else:
            action_mapped = self.model_arm((tf.expand_dims(state, axis=0), max_z))
            action_mapped = tf.reshape(action_mapped, (self.n_action, ))
            action = tf.argmax(tf.reshape(action_mapped, (self.n_action, )), axis=-1).numpy()
        
        return action, action_mapped

    def update_model(self):
        batch = random.sample(self.buffer, self.batch_size)

        state, action_mapped, delta_s_hat, next_state = zip(*batch)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action_mapped = tf.convert_to_tensor(action_mapped, dtype=tf.float32)
        delta_s_hat = tf.convert_to_tensor(delta_s_hat, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)

        delta_s = tf.subtract(next_state, state)
        delta_s = tf.tile(delta_s, [1, self.n_latent_action])
        delta_s = tf.reshape(delta_s, (self.batch_size, self.n_latent_action, self.n_state))

        dist = tf.norm(tf.subtract(delta_s_hat, delta_s), axis=-1) # (batch, n_latent_action)
        min_z = tf.one_hot(tf.argmin(dist, axis=-1), self.n_latent_action) # (batch, n_latent_action)

        with tf.GradientTape() as tape:
            action_label = self.model_arm((state, min_z))

            loss = get_loss_map(action_mapped, action_label)
        
        grads = tape.gradient(loss, self.model_arm.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_arm.trainable_variables))
        
        return loss

    def env_init(self):
        self.done = False
        if self.sim == 'LunarLander-v2':
            self.state = self.env.reset()[0][:6]
        else:
            self.state = self.env.reset()[0]
    
    def train_step(self, n):
        if self.is_normalize:
            self.state = tf.divide(tf.subtract(self.state, self.min), tf.subtract(self.max, self.min))

        self.z_p, delta_s_hat = self.model_lpn(tf.expand_dims(self.state, axis=0)) # z_p : (batch, n_latent_action) / delta_s_hat : (batch, n_latent_action, n_state)
        max_z = tf.one_hot(tf.argmax(self.z_p, axis=-1), self.n_latent_action)
        delta_s_hat = tf.reshape(delta_s_hat, (self.n_latent_action, self.n_state))

        action, action_mapped = self.e_greedy_action(self.state, max_z)

        next_state, reward, terminated, truncated, info = self.env.step(action)
        self.done = terminated or truncated

        if self.sim == 'LunarLander-v2':
            next_state = next_state[:6]

        if self.is_normalize:
            next_state = tf.divide(tf.subtract(next_state, self.min), tf.subtract(self.max, self.min))
        
        if not self.done:
            self.buffer.append((self.state, action_mapped, delta_s_hat, next_state))

        self.state = next_state
        self.train_score += reward

        if len(self.buffer) >= self.min_buffer_size:
            loss = self.update_model()
            self.train_loss_episode.append(loss.numpy())
            self.train_loss_step.append(loss.numpy())
            self.train_episode.append(n)
    
    def validate_step(self, n):
        if self.is_normalize:
            self.state = tf.divide(tf.subtract(self.state, self.min), tf.subtract(self.max, self.min))
        
        self.z_p, delta_s_hat = self.model_lpn(tf.expand_dims(self.state, axis=0)) # z_p : (batch, n_latent_action) / delta_s_hat : (batch, n_latent_action, n_state)
        max_z = tf.one_hot(tf.argmax(self.z_p, axis=-1), self.n_latent_action)
        max_delta_s_hat = tf.reshape(delta_s_hat, (self.n_latent_action, self.n_state))[tf.argmax(tf.reshape(self.z_p, (self.n_latent_action,)), axis=-1).numpy()]

        action_mapped = self.model_arm((tf.expand_dims(self.state, axis=0), max_z))
        action = tf.argmax(tf.reshape(action_mapped, (self.n_action, )), axis=-1).numpy()

        next_state, reward, self.terminated, truncated, info = self.env.step(action)
        self.done = self.terminated or truncated

        if self.sim == 'LunarLander-v2':
            next_state = next_state[:6]
        
        if self.is_normalize:
            next_state = tf.divide(tf.subtract(next_state, self.min), tf.subtract(self.max, self.min))
        
        metric = tf.subtract(next_state, max_delta_s_hat)
        metric = tf.reduce_mean(tf.square(metric))

        delta_s = tf.subtract(next_state, self.state) # (n_state, )
        delta_s = tf.tile(delta_s, [self.n_latent_action])
        delta_s = tf.reshape(delta_s, (self.n_latent_action, self.n_state))

        dist = tf.norm(tf.subtract(tf.reshape(delta_s_hat, (self.n_latent_action, self.n_state)), delta_s), axis=-1) # (n_latent_action,)
        min_z = tf.one_hot(tf.argmin(dist, axis=-1), self.n_latent_action) # (n_latent_action,)

        action_label = self.model_arm((tf.expand_dims(self.state, axis=0), tf.expand_dims(min_z, axis=0)))

        loss = get_loss_map(action_mapped, action_label)

        self.state = next_state
        self.val_score += reward
        self.val_metric.append(metric)
        self.val_z.append(int(tf.argmax(tf.reshape(self.z_p, (self.n_latent_action,)), axis=-1).numpy()))
        if len(self.buffer) >= self.min_buffer_size:
            self.val_loss_episode.append(loss.numpy())
            self.val_loss_step.append(loss.numpy())
            self.val_episode.append(n)
    
    def train(self):
        self.optimizer = Adam(learning_rate=self.learning_rate)
        patience = 0

        for n in range(self.epochs):
            self.env_init()

            self.e = max(self.e_min, self.e * self.e_decay_factor)
            # self.e = 0.2

            self.train_score = 0
            self.train_loss_episode = []

            '''Train'''
            while not self.done:
                self.train_step(n)
            
            self.train_total_score.append(self.train_score)

            print('----------------------------------------------------------------')
            print(f'Episode >> {n}\t\tEpsilon >> {self.e:.4f}\t\tBuffer Size >> {len(self.buffer)}')
            if len(self.buffer) >= self.min_buffer_size:
                self.train_loss.append(np.mean(self.train_loss_episode))
                print(f'Train Loss >> {self.train_loss[-1]:.4f}\tTrain Score >> {self.train_score:.4f}', end='')
            else:
                print(f'Train Loss >>    -   \tTrain Score >> {self.train_score:.4f}', end='')

            '''Validation'''
            self.env_init()

            self.val_score = 0
            self.val_metric = []
            self.val_loss_episode = []

            while not self.done:
                self.validate_step(n)

            if len(self.buffer) >= self.min_buffer_size:
                self.val_loss.append(np.mean(self.val_loss_episode))
                if np.mean(self.val_metric) <= np.min(self.val_avg_metric):
                    self.model_arm.save_weights(os.path.join(self.save_path, 'best_weights_metric'))
                
                if self.val_score >= np.max(self.val_total_score):
                    patience = 0
                    self.model_arm.save_weights(os.path.join(self.save_path, 'best_weights_score'))
                else:
                    patience += 1
            
            print(f'\tValidation Score >> {self.val_score:.4f}')
            if self.terminated:
                self.val_terminated.append(1)
            else:
                self.val_terminated.append(0)
            
            self.val_avg_metric.append(np.mean(self.val_metric))
            self.val_total_score.append(self.val_score)
            
            if patience >= self.max_patience:
                print(f"Train finished : minimun of val_metric : {np.min(self.val_avg_metric)}")
                self.save_train_results()
                break
        
        # self.model_arm.save_weights(os.path.join(self.save_path, 'best_weights'))
        self.save_train_results()

if __name__ == '__main__':
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    gpu_limit(2)
    np.random.seed(42)

    Trainer = trainer()

    Trainer.train()