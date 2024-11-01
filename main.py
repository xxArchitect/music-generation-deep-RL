# COMP 4010 Project
# Carleton University
# Members:
# Aidan MacGillivary (101181223)
# Fraser Rankin (101192297)
# Angus MacGillivary (101161464)
# Zhi Ou Yang (101161808)
# Vasily Inkovskiy (101084058)
import numpy as np
from env import Env
from agent import Agent

def main():
    gamma = 0.9
    step_size = 0.1
    epsilon = 0.1
    max_episodes = 1000
    bars_per_episode = 4

    # Rhythmic precision is how many intervals of time there are per beat/bar.
    # Ex: With a precision of 16, there would be 16 spots to place a note in a bar
    # or 4 spots for a single beat, assuming a 4/4 time signature).
    rhythmic_precision = 16 

    # Restrict pitch range down from 88 notes for feasibility. We shouldn't need 
    # more than 2 octaves (25 notes) realistically.
    num_actions = 27 # x notes + sustain note (hold) + rest (no note)

    # Example piano roll with a single-octave pitch range (15 actions):
    # 0 - no note, 1 - note, 2 - sustain last note
    #                          Pop Goes the Weasel
    #  C | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  B | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # A# | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  A | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # G# | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  G | 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # F# | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  F | 0 0 0 0 0 0 0 0 0 0 1 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  E | 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0
    # D# | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  D | 0 0 0 0 0 0 1 2 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    # C# | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    #  C | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 2 2 2 0 0 0 0 0 0 0 0 0
    #     _______________________________________________________________
    #                               Intervals    
    #                  
    # Note: When printing a piano roll, it will be rotated 90 degrees clockwise
    # compared to this example

    # Defining the entire piano roll as the state produces a completely infeasible 
    # number of states. It would not be realistic to compute or store without using methods
    # that are outside the scope of this course. So instead, we'll define the state as the 
    # last four intervals for feasibility. Unfortunately, this greatly reduces the agent's
    # memory of what notes it played which could make it difficult to reward repetitive
    # note sequences.
    state_length = 4
    num_states = np.power(num_actions, state_length) # 6.25 million states
    
    env = Env(state_length, num_actions, num_states)
    agent = Agent(gamma, step_size, epsilon, num_actions, num_states)

    results = train(env, agent, max_episodes, bars_per_episode * rhythmic_precision, num_actions)
    
    tempo = 115 # beats per minute
    show_results(results, tempo)

def train(env, agent, max_episodes, episode_length, num_actions):
    # arbitrarily choose which episodes to store
    episodes_to_store = {0, int(max_episodes/3), int(2*max_episodes/3), max_episodes-1}
    results = []

    for episode in range(max_episodes):
        state = env.reset()
        interval_count = 0
        piano_roll = np.zeros((episode_length, num_actions - 2), int)

        while interval_count < episode_length:
            action = agent.choose_action(state)
            next_state, reward = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            piano_roll[interval_count] = state[len(state) - 1]
            interval_count += 1

        if episode in episodes_to_store:
            results.append((episode, piano_roll))

    return results
            

def show_results(results, tempo):
    # TO-DO
    # Create UI (or just generate audio files) with ability to playback 
    # piano roll from first and final episode, and maybe some in-between episodes
    #
    # I can code this part since it will require conversion of our 2D piano roll arrays
    # to MIDI files and VST usage, which I have some experience with. - Aidan

    # Terminal Output (placeholder)
    for result in results:
        print(f"Episode {result[0] + 1}:")
        print(result[1])
    return

if __name__ == "__main__":
    main()