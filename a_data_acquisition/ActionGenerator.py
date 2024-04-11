from glob import glob
from tqdm import tqdm
import numpy as np
import json
import os
import time

class Action_Generator():
    def __init__(self, cycle_time, run_time):
        self.cycle_time = cycle_time
        self.run_time = run_time
        self.time_left = run_time
        self.action_seq = []
    
    def time_scheduler(self, setting_time):
        self.time_left = self.time_left - setting_time

        if self.time_left < 0:
            print(f"Setting time is too long")
            print(f"Time left : {self.time_left}")
            return False
        else:
            return True

    def step(self, time, action):
        # self.action_seq = self.action_seq + [action] * (time / self.cycle_time)
        self.action_seq.extend([action] * int(time / self.cycle_time))

    # def linear(self):

    # def quadratic(self):

    # def sinus(self):

    def generate(self, mode, time_section, action):
        for i in range(len(mode)):
            if mode[i] == 'step':
                if self.time_scheduler(time_section[i]):
                    self.step(time_section[i], action[i])
                else:
                    self.step(self.time_left, action[i])
                    break

if __name__ == "__main__":
    mode = ['step'] * 4
    time_section = [250] * 4
    action = [5.0, 15.0, 5.0, 15.0]

    act_gen = Action_Generator(
        cycle_time=0.01,
        run_time=1000
    )

    act_gen.generate(
        mode = mode,
        time_section=time_section,
        action = action
        )

    print(set(act_gen.action_seq))