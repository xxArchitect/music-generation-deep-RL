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
        q_threshold_percent (float): The percent threshold of q-values are considered 
        good/bad.
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

    def __init__(self, gamma, step_size, epsilon, num_actions, num_states, 
                 q_threshold_percent, weight):
        """
        Initializes the agent with the specified parameters.

        Parameters:
            gamma (float): The discount factor.
            step_size (float): The learning rate.
            epsilon (float): The probability of exploring.
            num_actions (int): The number of actions to choose from.
            num_states (int): The total number of possible states the environment could be in.
            q_threshold_percent (float): The percent threshold of q-values which are 
            considered good/bad.
            weight (float): The amount of weight to put on action probabilities.

        Returns:
            None
        """

        # Initialize standard RL agent variables
        self.gamma = gamma
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_actions = num_actions
        self.num_states = num_states
        self.q_table = np.zeros((num_states * 5, num_actions))

        # For action selection
        self.q_threshold_percent = q_threshold_percent
        self.weight = weight

        # For indexing states
        self.known_states = {}
        self.next_index = 0
    
    def choose_action(self, state, episode, max_episodes):
        """
        Chooses the agent's action based on the current state and epsilon value.

        Parameters:
            state (dict): The current state of the environment, containing:
                - "last_note" (dict): The last note the agent played, containing:
                    - "pitch" (int): The pitch of the last note.
                    - "interval" (int): The time interval the note was played on.
                - "current_interval" (int): The current time interval of this episode.
                - "playing" (bool): The playing state of the agent.
            episode (int): The current episode in this training run.
            max_episodes (int): The total number of episodes to be ran.

        Returns:
            (int): The action the agent chose.
        """
        

        # Get index of current state
        self.state_index = self.get_index(state)

        # Randomly choose (w/ weighted probs) well-performing actions on last episode
        # This approach strikes a balance between the agent over-exploiting the same 
        # notes or exploring and playing off-key notes.
        if (episode == max_episodes - 1):
            q_min = np.min(self.q_table[self.state_index])

            # Q-values above this are considered "good" enough to consider selecting
            q_threshold = q_min * self.q_threshold_percent

            # Find all the good actions:
            good_actions = []
            good_q_values = np.array([])
            for i in range(len(self.q_table[self.state_index])):
                q_value = self.q_table[self.state_index][i]
                if q_value > q_threshold:
                    good_actions.append(i)
                    good_q_values = np.append(good_q_values, self.q_table[self.state_index][i])

            if not good_actions: # empty list; no good actions found
                # Default to highest q-value
                good_actions.append(np.argmax(self.q_table[self.state_index]))
                good_q_values = np.append(good_q_values, self.q_table[self.state_index][good_actions[0]])

            exp_q = np.exp(good_q_values / self.weight)
            probabilities = exp_q / np.sum(exp_q)
            return np.random.choice(good_actions, p=probabilities)
        
        # Otherwise, standard epsilon-based exploit/explore algorithm
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
            state (dict): The current state of the environment, containing:
                - "last_note" (dict): The last note the agent played, containing:
                    - "pitch" (int): The pitch of the last note.
                    - "interval" (int): The time interval the note was played on.
                - "current_interval" (int): The current time interval of this episode.
                - "playing" (bool): The playing state of the agent.
            action (int): The action the agent chose.
            reward (int): The reward value associated with the action.
            next_state (dict): The next state of the environment, containing:
                - "last_note" (dict): The last note the agent played, containing:
                    - "pitch" (int): The pitch of the last note.
                    - "interval" (int): The time interval the note was played on.
                - "current_interval" (int): The next time interval of this episode.
                - "playing" (bool): The playing state of the agent.

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
            state (dict): The current state of the environment, containing:
                - "last_note" (dict): The last note the agent played, containing:
                    - "pitch" (int): The pitch of the last note.
                    - "interval" (int): The time interval the note was played on.
                - "current_interval" (int): The current time interval of this episode.
                - "playing" (bool): The playing state of the agent.

        Returns:
            (int): The q-table index of the given state.
        """

        # Convert state to indexable data type (tuple)
        state_tuple = (state["last_note"]["pitch"], 
                           state["last_note"]["interval"], 
                           state["current_interval"],
                           state["playing"])
        
        # Assign an index to the tuple if it's new
        if state_tuple not in self.known_states:
            self.known_states[state_tuple] = self.next_index
            self.next_index += 1
        
        
        return self.known_states[state_tuple]