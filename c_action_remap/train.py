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
import numpy as np
import json
import yaml
import gym

class trainer():
    def __init__(self):
        self.config_dic, self.config_dic_p, self.model_name = self.get_config()
        self.action = self.config_dic['action']
        self.n_action = len(self.action)
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.batch_size = self.config_dic['batch_size']
        self.learning_rate = self.config_dic['learning_rate']
        self.epochs = self.config_dic['epochs']
        # self.max_patience = self.config_dic['max_patience']

        self.n_state = self.config_dic_p['n_state']
        self.n_latent_action = self.config_dic_p['n_latent_action']
        self.units_p = self.config_dic_p['units']
        self.layer_num_p = self.config_dic_p['layer_num']
        self.lrelu = self.config_dic_p['lrelu']
        self.is_normalize = self.config_dic_p['is_normalize']
        self.is_standardize = self.config_dic_p['is_standardize']
        self.sim = self.config_dic_p['sim']
        
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

        self.model_arm = Action_Remap_Network(
            n_action=self.n_action,
            units=self.units,
            layer_num=self.layer_num,
            batch_size=self.batch_size,
            n_state=self.n_state,
            n_latent_action=self.n_latent_action,
            units_p=self.units_p,
            layer_num_p=self.layer_num_p,
            lrelu=self.lrelu,
            is_normalize=self.is_normalize,
            is_standardize=self.is_standardize
        )

        self.model_arm.build_graph()
        self.model_lpn.build_graph()
    
    def get_config(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="config.yaml", help="yaml file for train configuration.")
        parser.add_argument("--model", default="model_0", help="yaml file for train configuration.")
        args = parser.parse_args()

        config_path = os.path.join(os.getcwd(), 'c_action_remap', 'configs', args.config)
        with open(config_path) as f:
            config_dic = yaml.load(f, Loader=yaml.FullLoader)
        
        config_path_p = os.path.join(os.getcwd(), 'runs', 'policy', args.model, 'config.yaml')
        with open(config_path_p) as f:
            config_dic_p = yaml.load(f, Loader=yaml.FullLoader)

        return config_dic, config_dic_p, args.model
    
    def get_save_path(self):
        save_dir = os.path.join(os.getcwd(), 'runs', 'action_remap', self.sim, self.model_name)
        self.save_path = save_dir

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

