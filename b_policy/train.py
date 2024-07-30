import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from model import PolicyNetwork, SeqPolicyNetwork
from utils import get_loss_min, get_loss_exp, get_metric, gpu_limit, get_relative_metric
from tensorflow.keras.optimizers import Adam
from DataLoader import DataLoader, SeqDataLoader
from copy import deepcopy
from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json
import yaml

class trainer():
    def __init__(self):
        self.config_dic = self.get_config()
        self.seq = self.config_dic['seq']
        self.n_latent_action = self.config_dic['n_latent_action']
        self.units = self.config_dic['units']
        self.layer_num = self.config_dic['layer_num']
        self.batch_size = self.config_dic['batch_size']
        self.lrelu = self.config_dic['lrelu']
        self.epochs = self.config_dic['epochs']
        self.max_patience = self.config_dic['max_patience']
        self.learning_rate = self.config_dic['learning_rate']
        self.sim = self.config_dic['sim']
        self.expert_num = self.config_dic['expert_num']
        self.data_num = self.config_dic['data_num']
        self.delta_t = self.config_dic['dt']
        self.n_state = len(self.config_dic['target_cols'])
        self.target_cols = self.config_dic['target_cols']

        self.get_save_path()

        if self.seq == 1:
            self.dataloader = DataLoader(
                sim=self.sim,
                expert_num=self.expert_num,
                data_num=self.data_num,
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
        
        else:
            self.dataloader = SeqDataLoader(
                sim=self.sim,
                expert_num=self.expert_num,
                data_num=self.data_num,
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

        self.train_loss_min = []
        self.train_loss_exp = []
        self.train_loss_total = []
        self.train_metric = []
        self.train_epochs = []

        self.val_loss_min = []
        self.val_loss_exp = []
        self.val_loss_total = []
        self.val_metric = []
        self.val_epochs = []

        self.val_metric_epoch_total = [np.inf]
        self.val_loss_epoch_total = [np.inf]

    def get_config(self):
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--config", default="freefall_0.yaml", help="yaml file for train configuration.")
        args = parser.parse_args()

        config_path = os.path.join(os.getcwd(), 'b_policy', 'configs', args.config)
        with open(config_path) as f:
            config_dic = yaml.load(f, Loader=yaml.FullLoader)

        return deepcopy(config_dic)

    def get_save_path(self):
        save_dir = os.path.join(os.getcwd(), 'runs', 'policy', self.sim, 'train')
        self.number = len(glob(os.path.join(save_dir, 'model*')))
        file_name = f'model_{self.number}'
        self.save_path = os.path.join(save_dir, file_name)

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

    def save_config(self):
        self.config_dic['n_state'] = self.n_state
        self.config_dic['units'] = self.units
        self.config_dic['layer_num'] = self.layer_num
        self.config_dic['batch_size'] = self.batch_size
        self.config_dic['lrelu'] = self.lrelu
        self.config_dic['epochs'] = self.epochs
        self.config_dic['max_patience'] = self.max_patience
        self.config_dic['learning_rate'] = self.learning_rate
        self.config_dic['epoch'] = self.epoch
        self.config_dic['min_val_metric'] = float(self.min_val_metric)
        self.config_dic['min_val_loss'] = float(self.min_val_loss)

        with open(os.path.join(self.save_path, 'config.yaml'), 'w') as f:
            yaml.dump(self.config_dic, f)
    
    def save_result_fig(self, dict_train, dict_val):
        train_loss_step = np.array(dict_train['loss_total'])
        train_metric_step = np.array(dict_train['metric'])
        train_epochs = np.array(dict_train['epochs'])
        val_loss_step = np.array(dict_val['loss_total'])
        val_metric_step = np.array(dict_val['metric'])
        val_epochs = np.array(dict_val['epochs'])

        train_loss = [np.mean(train_loss_step[np.where(train_epochs == e)]) for e in range(np.max(dict_train['epochs']))]
        val_loss = [np.mean(val_loss_step[np.where(val_epochs == e)]) for e in range(np.max(dict_val['epochs']))]
        train_metric = [np.mean(train_metric_step[np.where(train_epochs == e)]) for e in range(np.max(dict_train['epochs']))]
        val_metric = [np.mean(val_metric_step[np.where(val_epochs == e)]) for e in range(np.max(dict_val['epochs']))]

        plt.figure(figsize=(12,4.5))
        plt.subplot(1,2,1)
        plt.title(f"Loss of model_{self.number}")
        plt.plot(train_loss, label='Train loss')
        plt.plot(val_loss, label='Validation loss')
        plt.legend()

        plt.subplot(1,2,2)
        plt.title(f"Metric of model_{self.number}")
        plt.plot(train_metric, label='Train metric')
        plt.plot(val_metric, label='Validation metric')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'result.png'))

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
            'loss_total' : self.train_loss_total,
            'loss_min' : self.train_loss_min,
            'loss_exp' : self.train_loss_exp,
            'metric' : self.train_metric,
            'epochs' : self.train_epochs
                    }

        dict_val = {
            'loss_total' : self.val_loss_total,
            'loss_min' : self.val_loss_min,
            'loss_exp' : self.val_loss_exp,
            'metric' : self.val_metric,
            'epochs' : self.val_epochs
                    }
        
        self.save_result_fig(dict_train, dict_val)

        with open(os.path.join(self.save_path, 'train_results.json'), 'w') as f:
            json.dump(dict_train, f, default=self.serialize)

        with open(os.path.join(self.save_path, 'validation_results.json'), 'w') as f:
            json.dump(dict_val, f, default=self.serialize)

    def train(self):
        self.optimizer_min = Adam(learning_rate=self.learning_rate)
        self.optimizer_exp = Adam(learning_rate=self.learning_rate)
        self.dataloader.get_data()
        self.dataloader.split_train_val(val_ratio=0.2)

        patience = 0

        for epoch in range(self.epochs):
            train_metric_epoch = []
            '''Train step'''
            for i in tqdm(range(self.dataloader.len_train)):
                s, delta_s = self.dataloader.get_train_batch(i)

                with tf.GradientTape(persistent=True) as tape:
                    z_p, delta_s_hat = self.model(s, training=True)

                    loss_min = get_loss_min(delta_s, delta_s_hat)
                    loss_exp = get_loss_exp(delta_s, z_p, delta_s_hat)
                    loss_total = loss_min + loss_exp
                metric, relative_metric = get_metric(delta_s, z_p, delta_s_hat)

                self.train_loss_min.append(loss_min.numpy())
                self.train_loss_exp.append(loss_exp.numpy())
                self.train_loss_total.append(loss_total.numpy())
                self.train_metric.append(metric.numpy())
                self.train_epochs.append(epoch)
                train_metric_epoch.append(metric)

                grads_min = tape.gradient(loss_min, self.model.state_embedding.trainable_variables + self.model.generator.trainable_variables)
                grads_exp = tape.gradient(loss_exp, self.model.state_embedding.trainable_variables + self.model.latent_policy.trainable_variables)

                self.optimizer_min.apply_gradients(zip(grads_min, self.model.state_embedding.trainable_variables + self.model.generator.trainable_variables))
                self.optimizer_exp.apply_gradients(zip(grads_exp, self.model.state_embedding.trainable_variables + self.model.latent_policy.trainable_variables))

            print(f"Epoch : {epoch}")
            print(f"Train metric : {np.mean(train_metric_epoch):.5f}")

            del s, delta_s, z_p, delta_s_hat, loss_min, loss_exp, loss_total, metric, grads_min, grads_exp

            val_metric_epoch = []
            val_loss_epoch = []

            '''Validation step'''
            for j in tqdm(range(self.dataloader.len_val)):
                s, delta_s = self.dataloader.get_val_batch(j)

                z_p, delta_s_hat = self.model(s, training=False)

                loss_min = get_loss_min(delta_s, delta_s_hat)
                loss_exp = get_loss_exp(delta_s, z_p, delta_s_hat)
                loss_total = loss_min + loss_exp
                metric, relative_metric = get_metric(delta_s, z_p, delta_s_hat)

                self.val_loss_min.append(loss_min.numpy())
                self.val_loss_exp.append(loss_exp.numpy())
                self.val_loss_total.append(loss_total.numpy())
                self.val_metric.append(metric.numpy())
                self.val_epochs.append(epoch)
                val_metric_epoch.append(metric)
                val_loss_epoch.append(loss_total.numpy())

            avg_metric = np.mean(val_metric_epoch)
            avg_loss = np.mean(val_loss_epoch)
            print(f"Validation metric : {avg_metric:.5f} \t Validation loss : {avg_loss:.5f}")

            # if avg_metric <= np.min(self.val_metric_epoch_total):
            #     print(f"Update weights : Current >> {avg_metric:.5f}, Previous >> {self.val_metric_epoch_total[-1]:.5f}, Minimum >> {np.min(self.val_metric_epoch_total):.5f}")
            #     print(f"Target columns relative metric :", end=' ')
            #     print(relative_metric)
            if avg_loss <= np.min(self.val_loss_epoch_total):
                print(f"Update weights : Current >> {avg_loss:.5f}, Previous >> {self.val_loss_epoch_total[-1]:.5f}, Minimum >> {np.min(self.val_loss_epoch_total):.5f}")
                print(f"Target columns relative metric :", end=' ')
                print(relative_metric)
                self.model.save_weights(os.path.join(self.save_path, 'best_weights'))
                self.model.save(os.path.join(self.save_path, 'best_model'))
                patience = 0

            else:
                patience += 1
                # print(f"Maintain weights : Current >> {avg_metric:.5f}, Previous >> {self.val_metric_epoch_total[-1]:.5f}, Minimum >> {np.min(self.val_metric_epoch_total):.5f}")
                # print(f"Target columns relative metric :", end=' ')
                print(f"Maintain weights : Current >> {avg_loss:.5f}, Previous >> {self.val_loss_epoch_total[-1]:.5f}, Minimum >> {np.min(self.val_loss_epoch_total):.5f}")
                print(f"Target columns relative metric :", end=' ')
                print(relative_metric)
                if patience >= self.max_patience:
                    print(f"Train finished : minimun of val_metric : {np.min(self.val_metric_epoch_total)}")
                    print(f"Train finished : minimun of val_loss : {np.min(self.val_loss_epoch_total)}")
                    self.epoch = epoch
                    self.min_val_metric = np.min(self.val_metric_epoch_total)
                    self.min_val_loss = np.min(self.val_loss_epoch_total)
                    self.save_train_results()
                    break

            self.dataloader.on_epoch_end()
            self.val_metric_epoch_total.append(avg_metric)
            self.val_loss_epoch_total.append(avg_loss)
        
        self.epoch = epoch
        self.min_val_metric = np.min(self.val_metric_epoch_total)
        self.min_val_loss = np.min(self.val_loss_epoch_total)
        self.save_train_results()

if __name__=="__main__":
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)

    gpu_limit(2)
    np.random.seed(42)

    Trainer = trainer()

    Trainer.train()