import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from b_policy.model import Policy_Network
from model import Action_Remap_Network
from utils import get_loss_map
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
        self.config_dic, self.config_dic_p, self.config_path_p = self.get_config()
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.batch_size = self.config_dic['batch_size']
        self.learning_rate = self.config_dic['learning_rate']
        self.epochs = self.config_dic['epochs']
        self.buffer = deque(maxlen=self.config_dic['buffer_size'])
        self.min_buffer_size = self.config_dic['min_buffer_size']
        self.e = self.config_dic['e_init']
        self.e_decay_factor = self.config_dic['e_decay_factor']
        self.e_min = self.config_dic['e_min']
        # self.max_patience = self.config_dic['max_patience']

        self.n_state = self.config_dic_p['n_state']
        self.n_latent_action = self.config_dic_p['n_latent_action']
        self.units_p = self.config_dic_p['units']
        self.layer_num_p = self.config_dic_p['layer_num']
        self.lrelu = self.config_dic_p['lrelu']
        self.is_normalize = self.config_dic_p['is_normalize']
        self.is_standardize = self.config_dic_p['is_standardize']
        self.sim = self.config_dic_p['sim']
        self.get_models()

        if self.is_normalize:
            self.min = np.array([-1.5, -1.5, -5., -5., -3.14, -5.])
            self.max = np.array([1.5, 1.5, 5., 5., 3.14, 5.])
        
        # if self.is_standardize

        self.train_loss = []
        self.train_score = []

        self.val_terminated = []
        self.val_score = []
    
    def get_config(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config.yaml", help="yaml file for train configuration.")
        parser.add_argument("--model", default="model_0", help="model directory for policy network.")
        args = parser.parse_args()

        config_path = os.path.join(os.getcwd(), 'c_action_remap', 'configs', args.config)
        with open(config_path) as f:
            config_dic = yaml.load(f, Loader=yaml.FullLoader)
        
        config_path_p = os.path.join(os.getcwd(), 'runs', 'policy', args.model, 'config.yaml')
        with open(config_path_p) as f:
            config_dic_p = yaml.load(f, Loader=yaml.FullLoader)

        return config_dic, config_dic_p, config_path_p
    
    def get_save_path(self):
        save_dir = os.path.join(os.getcwd(), 'runs', 'action_remap', self.sim, self.model_name)
        self.save_path = save_dir

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
    
    def get_models(self):
        self.model_lpn = Policy_Network(
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units=self.units_p,
            layer_num=self.layer_num_p,
            batch_size=self.batch_size,
            lrelu=self.lrelu,
            is_normalize=self.is_normalize,
            is_standardize=self.is_standardize
        )
        self.model_lpn.build_graph()
        self.model_lpn.load_weights(os.path.join(os.dirname(self.config_path_p), 'best_weights'))

        self.model_arm = Action_Remap_Network(
            n_action=self.n_action,
            units=self.units,
            layer_num=self.layer_num,
            batch_size=1,
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units_p=self.units_p,
            layer_num_p=self.layer_num_p,
            lrelu=self.lrelu,
            is_normalize=self.is_normalize,
            is_standardize=self.is_standardize
        )
        self.model_arm.build_graph()
    
    def save_config(self):
        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config_dic, f)
    
    def save_train_results(self):
        self.save_config()

        dict_train = {
            'loss' : self.train_loss,
            'score' : self.train_score
        }

        dict_val = {
            'terminated' : self.val_terminated,
            'score' : self.val_score
        }

        with open(os.path.join(self.save_path, 'train_results.json'), 'w') as f:
            json.dump(dict_train, f, default=self.serialize)

        with open(os.path.join(self.save_path, 'validation_results.json'), 'w') as f:
            json.dump(dict_val, f, default=self.serialize)
    
    def e_greedy_action(self, state, max_z):
        if np.random.uniform(0, 1) <= self.e:
            action = np.random.choice(np.arange(self.n_action))
            action_mapped = tf.one_hot([action], self.n_action)
        else:
            action_mapped = self.model_arm((state, max_z))
            action = tf.argmax(tf.reshape(action, (self.n_action, )), axis=-1)
        
        return action, action_mapped

    def update_model(self):
        batch = random.sample(self.buffer, self.batch_size)

        state, action_mapped, delta_s_hat, next_state = zip(*batch)

        delta_s = tf.subtract(next_state, state)
        delta_s = tf.tile(delta_s, [1, self.n_latent_actions])
        delta_s = tf.reshape(delta_s, (self.batch_size, self.n_latent_actions, self.n_state))

        dist = tf.norm(tf.subtract(delta_s_hat, delta_s), axis=-1) # (batch, n_latent_action)
        min_z = tf.one_hot(tf.argmin(dist, axis=-1), self.n_latent_action) # (batch, n_latent_action)

        with tf.GradientTape() as tape:
            action_label = self.model_arm((state, min_z))

            loss = get_loss_map(action_label, action_mapped)
        self.train_loss.append(loss)
        
        grads = tape.gradient(loss, self.model_arm.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model_arm.trainable_variables))
        print(f'Train Loss >> {tf.reduce_mean(loss).numpy():.4f}')
    
    def train(self):
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")
        self.n_action = self.env.action_space.n

        for n in range(self.epochs):
            done = False
            state = self.env.reset()[0,:6]
            score = 0

            self.e = max(self.e_min, self.e * (self.e_decay_factor ** n))

            '''Train'''
            while not done:
                if self.is_normalize:
                    state = tf.divide(tf.subtract(state, self.min), tf.subtract(self.max, self.min))

                z_p, delta_s_hat = self.model_lpn(tf.expand_dims(state, axis=0)) # z_p : (batch, n_latent_action) / delta_s_hat : (batch, n_latent_action, n_state)
                max_z = tf.one_hot(tf.argmax(z_p, axis=-1), self.n_latent_actions)

                action, action_mapped = self.e_greedy_action(state, max_z)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = next_state[:6]
                if self.is_normalize:
                    next_state = tf.divide(tf.subtract(next_state, self.min), tf.subtract(self.max, self.min))
                
                if not done:
                    self.buffer.append((state, action_mapped, delta_s_hat, next_state))

                state = next_state
                score += reward

                if len(self.buffer) >= self.min_buffer_size:
                    self.update_model()
            
            self.train_score.append(score)
            print(f'Train Score >> {score}')

            '''Validation'''
            done = False
            state = self.env.reset()[0,:6]
            val_score = 0

            while not done:
                if self.is_normalize:
                    state = tf.divide(tf.subtract(state, self.min), tf.subtract(self.max, self.min))
                
                z_p, delta_s_hat = self.model_lpn(tf.expand_dims(state, axis=0)) # z_p : (batch, n_latent_action) / delta_s_hat : (batch, n_latent_action, n_state)
                max_z = tf.one_hot(tf.argmax(z_p, axis=-1), self.n_latent_actions)

                action_mapped = self.model_arm((state, max_z))
                action = tf.argmax(tf.reshape(action, (self.n_action, )), axis=-1)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                next_state = next_state[:6]
                if self.is_normalize:
                    next_state = tf.divide(tf.subtract(next_state, self.min), tf.subtract(self.max, self.min))
                
                state = next_state
                val_score += reward
            
            self.val_score.append(val_score)
            print(f'Validation Score >> {val_score}')
            if terminated:
                self.val_terminated.append(1)
            else:
                self.val_terminated.append(0)
