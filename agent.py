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
    """
    Represents an agent that can choose actions to affect an environment object. Based
    on received rewards, it updates its q-table using a QLearning algorithm.

    Attributes:
        gamma (float): The discount factor.
        step_size (float): The learning rate.
        epsilon (float): The probability of exploring.
        num_actions (int): The number of actions to choose from.
        num_states (int): The total number of possible states the environment could be in.
        q_table (np.ndarray): A table that stores the q-values of each state-action pair.
        known_states (dict): A dictionary containing the index of each state the agent has
        visited.
        next_index (int): A counter to keep track of how many states have been indexed.

    Methods:
        __init__(self, gamma, step_size, epsilon, num_actions, num_states):
            Initializes the agent with the specified parameters.
        choose_action(self, state):
            Chooses the agent's action based on the current state and epsilon value.
        update(self, state, action, reward, next_state):
            Updates the agent's q-table based on the reward received for its last action.
        get_index(self, state):
            Gets the index of a state and creates one if it doesn't exist.
    """

    def __init__(self, gamma, step_size, epsilon, num_actions, num_states):
        """
        Initializes the agent with the specified parameters.

        Parameters:
            gamma (float): The discount factor.
            step_size (float): The learning rate.
            epsilon (float): The probability of exploring.
            num_actions (int): The number of actions to choose from.
            num_states (int): The total number of possible states the environment could be in.
            q_table (np.ndarray): A table that stores the q-values of each state-action pair.

        Returns:
            None
        """

        # Initialize standard RL agent variables
        self.gamma = gamma
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_states = num_states
        self.q_table = np.zeros((num_states, num_actions))

        # For indexing states
        self.known_states = {}
        self.next_index = 0
    
    def choose_action(self, state):
        """
        Chooses the agent's action based on the current state and epsilon value.

        Parameters:
            state (np.ndarray): The current state.

        Returns:
            (int): The action the agent chose.
        """

        # Get index of current state
        self.state_index = self.get_index(state)

        if random.random() > self.epsilon:
            # Exploit
            return np.argmax(self.q_table[self.state_index])
        else:
            # Explore
            return random.randint(0, self.num_actions - 1) 

    def update(self, state, action, reward, next_state):
        """
        Updates the agent's q-table based on the reward received for its last action.

        Parameters:
            state (np.ndarray): The current state.
            action (int): The action the agent chose.
            reward (float): The reward value associated with the action.
            next_state (np.ndarray): The next state.

        Returns:
            (np.ndarray): The q-table.
        """

        # Get index of current and next state
        self.state_index = self.get_index(state)
        self.next_state_index = self.get_index(next_state)

        # QLearning function
        optimal_action = np.argmax(self.q_table[self.next_state_index])
        TD_target = reward + self.gamma * self.q_table[self.next_state_index][optimal_action]
        TD_error = TD_target - self.q_table[self.state_index][action]
        
        self.q_table[self.state_index][action] += self.step_size * TD_error

        return self.q_table
    
    def get_index(self, state):
        """
        Gets the index of a state and creates one if it doesn't exist.

        Parameters:
            state (np.ndarray): A state.

        Returns:
            (int): The q-table index of the given state.
        """

        # Convert state to indexable data type (tuple)
        state_tuple = tuple(state.flatten())
        
        # Assign an index to the tuple if it's new
        if state_tuple not in self.known_states:
            self.known_states[state_tuple] = self.next_index
            self.next_index += 1
        
        return self.known_states[state_tuple]