import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from b_policy.model import Policy_Network
from model import Action_Remap_Network
from tensorflow.keras.optimizers import Adam
from DataLoader import Data_Loader
from copy import deepcopy
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import gymnasium as gym
import numpy as np
import json
import yaml

class trainer():
    def __init__(self):
        self.config_dic, self.config_dic_p, self.config_path_p = self.get_config()
        self.n_action = len(self.action)
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.batch_size = self.config_dic['batch_size']
        self.learning_rate = self.config_dic['learning_rate']
        self.epochs = self.config_dic['epochs']
        self.e_init = self.config_dic['e_init']
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
        self.train_metric = []
        self.train_epochs = []

        self.val_loss = []
        self.val_metric = []
        self.val_epochs = []
    
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
            'metric' : self.train_metric,
            'epochs' : self.train_epochs
        }

        dict_val = {
            'loss' : self.val_loss,
            'metric' : self.val_metric,
            'epochs' : self.val_epochs
        }

        with open(os.path.join(self.save_path, 'train_results.json'), 'w') as f:
            json.dump(dict_train, f, default=self.serialize)

        with open(os.path.join(self.save_path, 'validation_results.json'), 'w') as f:
            json.dump(dict_val, f, default=self.serialize)
    
    def e_greedy_action(self, state, max_z):

    
    def train(self):
        self.optimizer = Adam(learning_rate=self.learning_rate)
        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")

        for e in range(self.epochs):
            done = False
            state = self.env.reset()[0,:6]

            while not done:
                if self.is_normalize:
                    state = np.divide(np.subtract(state, self.min), np.subtract(self.max, self.min))

                z_p, delta_s = self.model_lpn(state)
                
