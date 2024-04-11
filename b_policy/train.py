from model import State_Embedding, Policy_Network
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import json
import yaml
import os

# class trainer():
#     def __init__(self, n_latent_action, units, layer_num, batch_size, lrelu=0.2, is_normalize=False, is_standardize=False):