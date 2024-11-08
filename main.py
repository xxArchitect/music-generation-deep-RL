# COMP 4010 Project
# Carleton University
# Members:
# Aidan MacGillivary (101181223)
# Fraser Rankin (101192297)
# Angus MacGillivary (101161464)
# Zhi Ou Yang (101161808)
# Vasily Inkovskiy (101084058)

import numpy as np
import copy
import os
from env import Env
from agent import Agent
from helpers import piano_roll_to_midi
from helpers import midi_to_audio

def train(env, agent, max_episodes, episode_length, note_range, episodes_to_store):
    """
    Trains the agent in the environment for a specified number of episodes using
    an action-reward feedback loop.

    Parameters:
        env (Env): Environment object.
        agent (Agent): Agent object.
        max_episodes (int): The number of episodes the agent will be trained for.
        episode_length (int): The length of each episode, measured in intervals of time.
        note_range (int): The range of notes the agent can play as an action.
        episodes_to_store ([int]): The index of each episode that should be saved for later.

    Returns:
        [(int, np.ndarray)]: A list of tuples containing each episode that was stored.
        The tuples contain:
            - episode (int): The episode number.
            - piano_roll (np.ndarray): The piano roll for that episode.
    """
    # Initialize results
    results = []

    # Training loop
    for episode in range(max_episodes):

        # Initialize environment
        state = env.reset()
        interval_count = 0
        piano_roll = np.zeros((episode_length, note_range), int)

        while interval_count < episode_length:
            # 1. Agent chooses action based on state
            action = agent.choose_action(state)

            # Store state as a copy instead of a reference to prevent it from updating
            # before the agent updates its q-table
            state = copy.deepcopy(state)

            # 2. Environment updates based on action, returning the next state and reward
            next_state, reward = env.step(action)

            # 3. Agent updates q-table based on the reward
            agent.update(state, action, reward, next_state)

            # 4. Update other variables
            state = next_state
            piano_roll[interval_count] = state[-1]
            interval_count += 1

            # Repeat for each interval of time for each episode

        # Store current episode for final results later
        if episode in episodes_to_store:
            results.append((episode, piano_roll))

    return results
            

def show_results(results, tempo, rhythmic_precision, note_range):
    """
    Converts the results received from the agent's training to terminal output and audio 
    files.

    Parameters:
        results ([(int, np.ndarray)]): A list of tuples containing each episode that was stored.
        The tuples contain:
            - episode (int): The episode number.
            - piano_roll (np.ndarray): The piano roll for that episode.
        tempo (int): The speed at which to play notes, measured in beats per minute.
        rhythmic_precision (int): The number of intervals of time per beat.
        note_range (int): The range of notes the agent can play as an action.

    Returns:
        None
    """
    # TO-DO (optional): Create UI to play audio files more easily

    # Center the pitch range around middle C (MIDI note number 60)
    lowest_note = 60 - (note_range // 2) 

    # Infinite length/width when printing arrays; prevents shortened piano rolls
    np.set_printoptions(threshold=np.inf)

    # Note labels for terminal output
    notes = ["C", "#", "D", "#", "E", "F", "#", "G", "#", "A", "#", "B",]
    piano_roll_header = "  "
    for note in range(note_range):
        piano_roll_header += notes[(note - note_range // 2) % 12]
        piano_roll_header += " "

    soundfont_path = os.path.join("soundfonts", "Arachno SoundFont - Version 1.0.sf2")

    for result in results:
        episode_num = result[0] + 1
        episode_data = result[1]

        # Terminal output
        print(f"Episode {episode_num}:")
        print(piano_roll_header)
        print(episode_data)

        # Piano roll to audio file conversion
        midi = piano_roll_to_midi(episode_data, f"episode{episode_num}", tempo, 
                                  lowest_note, rhythmic_precision)
        audio = midi_to_audio(midi, f"episode{episode_num}", 
                              soundfont_path)

def main():
    """
    The entry point for the program. Runs the training loop to generate results then
    converts them to text and audio form.

    Parameters:
        None

    Returns:
        None
    """

    gamma = 0.9
    step_size = 0.1
    epsilon = 0.1
    max_episodes = 1000
    episodes_to_store = [0, max_episodes // 3, 2 * max_episodes // 3, max_episodes - 1]
    beats_per_episode = 16
    tempo = 115 # beats per minute

    # Rhythmic precision is how many intervals of time there are per beat.
    rhythmic_precision = 4

    # The number of actions will be the number of notes the agent can play, plus a sustain
    # and rest action. The note/pitch range should be reduced down from a full piano range
    # of 88 notes for feasibility. Two octaves (25 notes) should be sufficient.
    num_actions = 27

    # Example piano roll with a single-octave pitch range (15 actions):
    # 0 - no note, 1 - note, 2 - sustain last note
    #
    #                                  Pop Goes the Weasel
    #                  
    #            | F# |  G | G# |  A | A# |  B |  C | C# |  D | D# |  E |  F | F# |
    # Intervals  __________________________________________________________________
    #     1      |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  1 |  0 |  0 |  0 |  0 |  0 |
    #     2      |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |
    #     3      |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #     4      |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #     5      |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #     6      |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #     7      |  0 |  0 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #     8      |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #     9      |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    10      |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    11      |  0 |  0 |  0 |  0 |  0 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    12      |  0 |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    13      |  0 |  0 |  0 |  0 |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    14      |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    15      |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    16      |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    17      |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    18      |  0 |  0 |  0 |  0 |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    19      |  1 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    20      |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    21      |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |
    #    22      |  2 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |  0 |

    # Defining the entire piano roll as the state produces a completely infeasible 
    # number of states. It would not be realistic to compute or store without using methods
    # that are outside the scope of this course. So instead, we'll define the state as the 
    # last four intervals for feasibility. Unfortunately, this greatly reduces the agent's
    # memory of what notes it played which could make it difficult to reward repetitive
    # note sequences.
    state_length = 4
    num_states = np.power(num_actions, state_length) 
    
    env = Env(state_length, num_actions, num_states)
    agent = Agent(gamma, step_size, epsilon, num_actions, num_states)

    results = train(env, agent, max_episodes, beats_per_episode * rhythmic_precision, 
                    num_actions - 2, episodes_to_store)
    
    show_results(results, tempo, rhythmic_precision, num_actions - 2)
    exit(0)

if __name__ == "__main__":
    main()