from glob import glob

import gymnasium as gym
import numpy as np
import pygame
import json
import time
import os

# CustomMountainCarEnv path : /home/pcy/anaconda3/envs/ilpo/lib/python3.11/site-packages/gymnasium/envs/classic_control/custom_mntcar.py
    
class MountainCarGame:
    def __init__(self, episodes=10, mode='play', action_space=[-0.001, 0.0, 0.001], gravity=0.0025):
        self.episodes = episodes
        self.mode = mode
        self.env = gym.make('CustomMountainCar-v0', render_mode='human', action=action_space, gravity = gravity)
        self.env_name = 'CustomMountainCar-v0'
        
        if self.mode == 'collect':
            self.get_save_path()
            self.state_list = []
            self.action_list = []
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.action = 0
                elif event.key == pygame.K_RIGHT:
                    self.action = 2
                elif event.type == pygame.K_BACKSPACE:
                    self.running = False
            elif event.type == pygame.KEYUP:
                self.action = 1
    
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
    
    def save_result(self):
        self.state_list = np.array(self.state_list, dtype=np.float32)
        dict_state = {
                    'pos' : list(self.state_list[:, 0]),
                    'v' : list(self.state_list[:, 1]),
                    'action' : self.action_list
                    }
        
        num = len(glob(os.path.join(self.save_path, self.file_name, 'senario*')))
        
        with open(os.path.join(self.save_path, self.file_name, f'senario_{num}.json'), 'w') as f:
            json.dump(dict_state, f, default=self.serialize)
        
        self.state_list = []
        self.action_list = []

    def serialize(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
    def play(self):
        pygame.init()
        pygame.display.set_caption("MountainCar Game")
        self.running = True
        self.action = 0

        for e in range(self.episodes):
            dones = False
            obs = self.env.reset()[0]
            
            if self.mode == 'collect':
                self.state_list.append(obs)
            
            while not dones:
                self.handle_events()
                if not self.running:
                    break

                obs, rewards, terminated, truncated, info = self.env.step(self.action)
                dones = terminated or truncated

                if self.mode == 'collect' and not dones:
                    self.state_list.append(obs)
                    self.action_list.append(self.action)
                
            time.sleep(2)
            
            if not self.running:
                break
            
            if self.running and self.mode == 'collect':
                self.save_result()

        pygame.quit()
        self.env.close()

if __name__ == "__main__":
    # game = MountainCarGame(episodes=30, mode='collect')
    game = MountainCarGame(episodes=10, mode='play', action_space=[-0.00075, 0.0, 0.00075])
    game.play()
