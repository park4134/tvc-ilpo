import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from Sim import Freefall
from ActionGenerator import Action_Generator
from DataGenerator import Data_Generator
from glob import glob
from tqdm import tqdm
import numpy as np
import json
import os
import time

class Simulator():
    def __init__(self, mass, cycle_time, run_time, s_init, mode, time_section, action):
        '''mass : kg, cycle_time & run_time : sec, action_seq : list of newtons, s_init : meter'''
        '''mode : simulation mode at each time section, time_section : length of each time section, action : action value at each time section'''
        self.mass = mass
        self.cycle_time = cycle_time
        self.run_time = run_time
        self.s_init = s_init
        self.mode = mode
        self.time_section = time_section
        self.action = action

    def run(self):
        self.action_generator = Action_Generator(cycle_time=self.cycle_time, run_time=self.run_time)
        self.action_generator.generate(
            mode = self.mode,
            time_section = self.time_section,
            action = self.action
        )

        self.data_generator = Data_Generator(
            mass= self.mass,
            cycle_time=self.cycle_time,
            run_time=self.run_time,
            action_seq=self.action_generator.action_seq,
            s_init=self.s_init
            )
        self.data_generator.generate()
        
        self.data_generator.save_result()

if __name__ == "__main__":
    Sim = Simulator(
        mass = 1.0,
        cycle_time = 0.01,
        run_time = 100,
        s_init = 5000.0,
        mode = ['step'] * 10,
        time_section = [10] * 10,
        action = [5.0, 15.0] * 5
    )

    Sim.run()