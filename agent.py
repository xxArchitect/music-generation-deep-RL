# COMP 4010 Project
# Carleton University
# Members:
# Aidan MacGillivary (101181223)
# Fraser Rankin (101192297)
# Angus MacGillivary (101161464)
# Zhi Ou Yang (101161808)
# Vasily Inkovskiy (101084058)

import random
import numpy as np

class Agent():
    def __init__(self, gamma, step_size, epsilon, num_actions, num_states):
        self.gamma = gamma
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_states = num_states

        self.q_table = np.full((num_states, num_actions), 0)

        # for indexing the 2D array states
        self.known_states = {}
        self.next_index = 0
    
    def choose_action(self, state):
        state_index = self.get_index(state)

        if random.random() > self.epsilon:
            return np.argmax(self.q_table[state_index]) # exploit
        else:
            return random.randint(0, self.num_actions - 1) # explore

    def update(self, state, action, reward, next_state):
        state_index = self.get_index(state)
        next_state_index = self.get_index(next_state)

        # QLearning function
        optimal_action = np.argmax(self.q_table[next_state_index])
        TD_target = reward + self.gamma * self.q_table[next_state_index][optimal_action]
        TD_error = TD_target - self.q_table[state_index][action]
        self.q_table[state_index][action] += self.step_size * TD_error
        return
    
    def get_index(self, state):
        # convert state to indexable data type (tuple)
        state_tuple = tuple(state.flatten())
        
        # assign an index to the tuple if it's new
        if state_tuple not in self.known_states:
            self.known_states[state_tuple] = self.next_index
            self.next_index += 1
        
        return self.known_states[state_tuple]