import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Sim import Freefall
from glob import glob
from tqdm import tqdm
import numpy as np
import json
import os
import time

class Data_Generator():
    def __init__(self, mass, cycle_time, run_time, action_seq, s_init = 0):
        '''mass : kg, cycle_time & run_time : sec, action_seq : list of newtons, s_init : meter'''
        self.cycle_time = cycle_time
        self.action_seq = action_seq
        self.run_time = run_time
        self.get_save_path()

        self.time = []
        self.state_list = []

        self.Sim = Freefall(mass=mass, s_init=s_init, cycle_time=self.cycle_time)

    def get_save_path(self):
        file_name = 'num'
        save_path = os.path.join(os.getcwd(), 'data', 'freefall')
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        num = len(glob(os.path.join(save_path, f'{file_name}*')))
        file_name = f'num_{num}'

        if not os.path.isdir(os.path.join(save_path, file_name)):
            os.makedirs(os.path.join(save_path, file_name))
        
        self.file_name = file_name
        self.save_path = save_path
    
    def serialize(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    def save_result(self):
        self.state_list = np.array(self.state_list, dtype=np.float64)
        self.action_seq = np.array(self.action_seq, dtype=np.float64)

        dict_result = {
                        'time' : self.time,
                        's' : self.state_list[:,0],
                        'v' : self.state_list[:,1],
                        'action' : self.action_seq
                        }
        
        with open(os.path.join(self.save_path, self.file_name, 'sim_result.json'), 'w') as f:
            json.dump(dict_result, f, default=self.serialize)

    def generate(self):
        iteration = int(self.run_time / self.cycle_time)

        '''Initial state'''
        self.state_list.append([self.Sim.s, self.Sim.v])
        self.time.append(0.0)
        for i in tqdm(range(iteration)):
            state = self.Sim.get_data(action = self.action_seq[i])
            self.time.append(self.Sim.t)
            self.state_list.append(state)
        
        '''Equalize length'''
        self.action_seq.append(0)
        self.save_result()

if __name__ == "__main__":
    data_gen = Data_Generator(
                            mass=1.0, 
                            cycle_time=0.01,
                            run_time=1000,
                            action_seq=[0.0]*100000,
                            s_init=10000.0
                            )
    
    data_gen.generate()