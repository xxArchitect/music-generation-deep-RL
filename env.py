# COMP 4010 Project
# Carleton University
# Members:
# Aidan MacGillivary (101181223)
# Fraser Rankin (101192297)
# Angus MacGillivary (101161464)
# Zhi Ou Yang (101161808)
# Vasily Inkovskiy (101084058)

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
        NOTE_PENALTIES: The reward values associated with each pitch.

    Methods:
        __init__(self, state_length, num_actions, num_states):
            Initializes the environment with the specified parameters.
        reset(self):
            Resets the environment to its initial state.
        step(self, action):
            Updates the state based on the action received then returns the new state and 
            reward.
        update_piano_roll(self, piano_roll, action, interval_count):
            Updates the piano roll for the current episode based on the chosen action.
    """

    def __init__(self, num_actions, num_states):
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

        # Flag constants
        self.PLAY = []
        for action in range(num_actions - 2):
            self.PLAY.append(action)
        
        self.SUSTAIN = num_actions - 2
        self.REST = num_actions - 1

        # in order of least to greatest pitch (I/unison, ii/min 2nd, ... , VII/maj 7th)
        self.NOTE_PENALTIES = [0, 10, 1, 10, 0, 0, 10, 0, 10, 1, 10, 1]

    def reset(self):
        """
        Resets the environment to its initial state.

        Parameters:
            None

        Returns:
            dict: An empty state, containing:
                    - "last_note" (dict): The last note the agent played, containing:
                        - "pitch" (int): The pitch of the last note.
                        - "interval" (int): The time interval the note was played on.
                    - "current_interval" (int): The current time interval of this episode.
                    - "playing" (bool): The playing state of the agent.
        """

        # Reset state to initial values
        self.state = {
            "last_note": {"pitch": -1, "interval": -1},
            "current_interval": 0,
            "playing": False
        }

        return self.state
    
    def step(self, action, rhythmic_precision, episode_length):
        """
        Updates the state based on the action received then returns the new state and reward.

        Parameters:
            action (int): The action chosen by the agent.
            rhythmic_precision (int): The number of time intervals per beat.
            episode_length (int): The number of time intervals per episode.

        Returns:
            (state, reward): A tuple containing:
                - state (dict): The current state of the environment, containing:
                    - "last_note" (dict): The last note the agent played, containing:
                        - "pitch" (int): The pitch of the last note.
                        - "interval" (int): The time interval the note was played on.
                    - "current_interval" (int): The current time interval of this episode.
                    - "playing" (bool): The playing state of the agent.
                - reward (float): The reward for the agent based on its action.
        """
        
        ####################
        # REWARD STRUCTURE #
        ####################

        reward = 0

        # Reward based on whether the note is in key
        if action in self.PLAY:
            reward -= self.NOTE_PENALTIES[action % 12]

        # Penalize playing too many notes, keep a good ratio with sustain and rest actions
        if action in self.PLAY:
            reward -= 4
        
        # Penalize playing the same pitch consecutively
        if action == self.state["last_note"]["pitch"]:
            reward -= 2
        
        # Penalize sustaining a note longer than one beat
        if action == self.SUSTAIN:
            if (self.state["current_interval"] - self.state["last_note"]["interval"] >= 1 * 
                rhythmic_precision) and self.state["playing"]:
                reward -= 3

        # Penalize sustaining if a note has not been played yet
        if action == self.SUSTAIN:      
            if (self.state["last_note"]["pitch"] == -1):
                reward -= 5

        # Penalize sustaining when there is no note to sustain
        if action == self.SUSTAIN:  
            if (self.state["playing"] == False):
                reward -= 4

        # Slightly penalize resting, keep a good ratio with other actions
        if action == self.REST:
            reward -= 3

        # Reward being on beat
        if action in self.PLAY and self.state["current_interval"] % rhythmic_precision == 0:
            reward += 2

        # Penalize notes right after a beat because it can de-emphasize the on-beat note
        elif action in self.PLAY and self.state["current_interval"] % rhythmic_precision == 1:
            reward -= 1

        # Reward pickup notes into next beat
        elif action in self.PLAY and self.state["current_interval"] % rhythmic_precision > 1:
            reward += 1

        # Reward last note being the tonic
        if (self.state["last_note"]["pitch"] % 12 == 0 and 
            self.state["current_interval"] == episode_length - 1 and
            action not in self.PLAY):
            reward += 5

        ####################
        #   STATE UPDATE   #
        ####################

        # Update state based on action:
        # If a play action is detected, store it as the last note
        if action in self.PLAY:
            self.state["last_note"]["pitch"] = action
            self.state["last_note"]["interval"] = self.state["current_interval"]
            self.state["playing"] = True

        # If a sustain action is detected, nothing needs to be updated
        elif action == self.SUSTAIN:
            pass
        
        # If a rest action is detected, just update the playing variable
        elif action == self.REST:
            self.state["playing"] = False
            pass
        
        self.state["current_interval"] += 1

        return self.state, reward
    
    def update_piano_roll(self, piano_roll, action, interval_count):
        """
        Updates the piano roll for the current episode based on the chosen action.

        Parameters:
            piano_roll (np.ndarray): The piano roll for the current episode.
            action (int): The action chosen by the agent.
            interval_count (int): The current time interval of this episode.

        Returns:
            (np.ndarray): The updated piano roll.
        """
        # Update piano roll based on action:
        # If a play action is detected, put a 1 at the corresponding pitch
        if action in self.PLAY:
            piano_roll[interval_count][action] = 1

        # If a sustain action is detected, put a 2 at the corresponding pitch
        # only if there is a note before it to be sustained (not a 0)
        elif action == self.SUSTAIN:
            for i in range(self.num_actions - 2):
                if piano_roll[interval_count - 1][i] != 0:
                    piano_roll[interval_count][i] = 2
        
        # If a rest action is detected, the interval is already initialized
        # to 0, so do nothing
        elif action == self.REST:
            pass

        return piano_roll