from glob import glob
import numpy as np
import json
import os
import time

class Freefall():
    def __init__(self, mass, s_init, cycle_time = 0.01, v_init = 0.0, a_init = -9.81):
        self.mass = mass
        self.cycle_time = cycle_time
        self.action = 0.0
        self.a_init = a_init
        self.t = 0.0
        self.s = s_init
        self.v = v_init
        self.a = self.a_init

        self.time = []
        self.state_total = []
        self.action_total = []

        self.flag_first = True

        self.get_save_path()

    def get_save_path(self):
        file_name = 'num'
        save_path = os.path.join(os.getcwd(), 'runs', 'freefall')
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
        self.state_total = np.array(self.state_total, dtype=np.float64)
        self.action_total = np.array(self.action_total, dtype=np.float64)

        dict_result = {
                        'time' : self.time,
                        's' : self.state_total[:,0],
                        'v' : self.state_total[:,1],
                        'action' : self.action_total
                        }
        
        with open(os.path.join(self.save_path, self.file_name, 'sim_result.json'), 'w') as f:
            json.dump(dict_result, f, default=self.serialize)
    
    def update(self):
        self.a = self.a_init + self.action / self.mass # acceleration update
        self.v = self.v + self.a * self.cycle_time # velocity update
        self.s = self.s + self.v * self.cycle_time + 0.5 * self.a * (self.cycle_time ** 2) # state update
        self.t = self.t + self.cycle_time

    def get_data(self, action):
        if self.flag_first:
            self.flag_first = False
            self.action_total.append(0.0)
        else:
            self.action = action
            self.action_total.append(self.action)
            self.update()
        self.time.append(self.t)
        state = [self.s, self.v]
        self.state_total.append(state)

        return state

if __name__== "__main__":
    Sim = Freefall(mass = 1.0, s_init = 100)

    for i in range(1000):
        Sim.get_data(action = 0)
    
    Sim.save_result()