import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from glob import glob
from tqdm import tqdm
import numpy as np
import gymnasium as gym
import json
import os

class DataGenerator():
    def __init__(self, env_name, model_dir, observe_episodes=10000):
        '''mass : kg, cycle_time & run_time : sec, action_seq : list of newtons, s_init : meter'''
        self.env_name = env_name
        self.model_dir = model_dir
        self.observe_episodes = observe_episodes
        self.get_save_path()

        self.state_list = []
        self.action_list = []

    def get_save_path(self):
        file_name = 'expert_data'
        save_path = os.path.join(os.getcwd(), 'data', 'row', self.env_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        num = len(glob(os.path.join(save_path, f'{file_name}*')))
        file_name = f'{file_name}_{num}'

        if not os.path.isdir(os.path.join(save_path, file_name)):
            os.makedirs(os.path.join(save_path, file_name))
        
        self.file_name = file_name
        self.save_path = save_path
    
    def serialize(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    def save_result(self):
        self.state_list = np.array(self.state_list, dtype=np.float32)
        if self.env_name == 'LunarLander-v2':
            dict_state = {'pos_x' : list(self.state_list[:, 0]),
                        'pos_y' : list(self.state_list[:, 1]),
                        'v_x' : list(self.state_list[:, 2]),
                        'v_y' : list(self.state_list[:, 3]),
                        'angle' : list(self.state_list[:, 4]),
                        'w' : list(self.state_list[:, 5]),
                        'is_grounded_left' : list(self.state_list[:, 6]),
                        'is_grounded_right' : list(self.state_list[:, 7]),
                        'action' : self.action_list
                        }
            
        elif self.env_name == 'MountainCar-v0':
            dict_state = {'pos' : list(self.state_list[:, 0]),
                        'v' : list(self.state_list[:, 1]),
                        'action' : self.action_list
                        }
        
        elif self.env_name == 'CartPole-v1':
            dict_state = {'pos' : list(self.state_list[:, 0]),
                        'v' : list(self.state_list[:, 1]),
                        'angle' : list(self.state_list[:, 2]),
                        'w' : list(self.state_list[:, 3]),
                        'action' : self.action_list
                        }
            
        num = len(glob(os.path.join(self.save_path, self.file_name, 'senario*')))
        
        with open(os.path.join(self.save_path, self.file_name, f'senario_{num}.json'), 'w') as f:
            json.dump(dict_state, f, default=self.serialize)
        
        self.state_list = []
        self.action_list = []

    def generate(self):
        env = gym.make(self.env_name, render_mode="rgb_array")
        model_path = os.path.join(os.getcwd(), "runs/expert/", self.env_name, self.model_dir)
        model = PPO.load(model_path, env=env)
        dones = False

        vec_env = model.get_env()
        obs = vec_env.reset()
        self.state_list.append(obs[0])
        c = 0

        for e in tqdm(range(self.observe_episodes)):
            while not dones:
                action, _states = model.predict(obs, deterministic=True)  # 상태 저장 옵션 추가
                obs, rewards, dones, info = vec_env.step(action)
                if not dones:
                    self.state_list.append(obs[0])
                    self.action_list.append(action)
                    vec_env.render()
            self.save_result()
            obs = vec_env.reset()
            dones = False
        vec_env.close()

if __name__ == "__main__":
    data_gen = DataGenerator(
                            env_name = 'CartPole-v1',
                            model_dir = 'PPO_cartpole_500.0',
                            observe_episodes = 100
                            )
    data_gen.generate()