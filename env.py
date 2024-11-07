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
    """
    Represents a environment that can maintain a state and generate rewards based on actions
    received from an Agent object.

    Attributes:
        num_actions (int): The number of actions an Agent can choose from.
        num_states (int): The total number of possible states the environment could be in.
        state_length (int): The number of intervals the state holds.
        PLAY ([int]): Each action that is associated with playing a note.
        SUSTAIN: The action that is associated with sustaining the last note.
        REST: The action that is associated with resting; playing no note.

    Methods:
        __init__(self, state_length, num_actions, num_states):
            Initializes the environment with the specified parameters.
        reset(self):
            Resets the environment to its initial state.
        step(self, action):
            Updates the state based on the action received then returns the new state and 
            reward.
    """

    def __init__(self, state_length, num_actions, num_states):
        """
        Initializes the environment with the specified parameters.

        Parameters:
            state_length (int): The number of intervals the state holds.
            num_actions (int): The number of actions an Agent can choose from.
            num_states (int): The total number of possible states the environment could be 
            in.

        Returns:
            None
        """

        # Initialize standard RL environment variables
        self.num_actions = num_actions
        self.num_states = num_states
        self.state_length = state_length

        # Flag constants
        self.PLAY = []
        for action in range(num_actions - 2):
            self.PLAY.append(action)
        
        self.SUSTAIN = num_actions - 2
        self.REST = num_actions - 1

    def reset(self):
        """
        Resets the environment to its initial state.

        Parameters:
            None

        Returns:
            [[int]]: An empty state.
        """

        # Reset state to 2D array of zeros
        self.state = np.zeros((self.state_length, self.num_actions - 2), int)

        return self.state
    
    def step(self, action):
        """
        Updates the state based on the action received then returns the new state and reward.

        Parameters:
            action (int): The action chosen by the agent.

        Returns:
            (state, reward): A tuple containing:
                - state (np.ndarray) The next state of the environment.
                - reward (float): The reward for the agent based on its action.
        """

        # Shift columns left by 1 for new interval
        for i in range(len(self.state)):
            if i < len(self.state) - 1:
                self.state[i] = self.state[i+1]
            else:
                # Initialize new interval
                self.state[i] = 0

        # Update state based on action:

        # If a play action is detected, put a 1 at the corresponding pitch
        if action in self.PLAY:
            self.state[-1][action] = 1

        # If a sustain action is detected, put a 2 at the corresponding pitch
        # only if there is a note before it to be sustained (not a 0)
        elif action == self.SUSTAIN:
            for i in range(self.num_actions - 2):
                if self.state[-1][i] != 0:
                    self.state[-1][i] = 2
        
        # If a rest action is detected, the new interval is already initialized
        # to 0, so do nothing
        elif action == self.REST:
            pass

        # TO-DO reward structure
        reward = 0
        

        return self.state, reward