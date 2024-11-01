# COMP 4010 Project
# Carleton University
# Members:
# Aidan MacGillivary (101181223)
# Fraser Rankin (101192297)
# Angus MacGillivary (101161464)
# Zhi Ou Yang (101161808)
# Vasily Inkovskiy (101084058)

import numpy as np

class Env():
    def __init__(self, state_length, num_actions, num_states):
        self.state_length = state_length
        self.num_actions = num_actions
        self.num_states = num_states
    
    def reset(self):
        self.state = np.zeros((self.state_length, self.num_actions - 2), int)
        return self.state
    
    def step(self, action):
        # shift columns left by 1 for new interval
        for i in range(len(self.state)):
            if i < len(self.state) - 1:
                self.state[i] = self.state[i+1]
            else:
                # initialize new interval
                self.state[i] = 0

        # update state based on action
        if action < self.num_actions - 2: # play note
            self.state[len(self.state) - 1][action] = 1

        elif action == self.num_actions - 2: # sustain
            for i in range(self.num_actions - 2):
                if self.state[len(self.state) - 2][i] != 0:
                    self.state[len(self.state) - 1][i] = 2
        
        # else rest
                

        # TO-DO reward structure
        reward = 0

        return self.state, reward