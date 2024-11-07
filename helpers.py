# COMP 4010 Project
# Carleton University
# Members:
# Aidan MacGillivary (101181223)
# Fraser Rankin (101192297)
# Angus MacGillivary (101161464)
# Zhi Ou Yang (101161808)
# Vasily Inkovskiy (101084058)

import os
from midiutil import MIDIFile
from midi2audio import FluidSynth
import numpy as np

def piano_roll_to_midi(piano_roll, filename, tempo, lowest_note, rhythmic_precision):
    """
    Converts a piano roll to a MIDI (.mid) file.

    Parameters:
        piano_roll (np.ndarray): The piano roll to be converted.
        filename (string): The desired filename of the MIDI file.
        tempo (int): The speed at which to play notes, measured in beats per minute.
        lowest_note (int): The lowest MIDI note number the agent can play, based on 
        the number of actions set.
        rhythmic_precision (int): The number of intervals of time per beat.

    Returns:
        (string): The path to the new MIDI file.
    """

    # Make output folder for the MIDI file
    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", f"{filename}.mid")
    
    # Initialize MIDI file and tracking variables
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    playing = False
    curr_note_end_time = -1

    # Flag constants
    REST = 0
    PLAY = 1
    SUSTAIN = 2

    # Loop through each interval of time in the piano roll backwards.
    # (Iterating backwards simplifies the logic for sustained notes)
    for interval in range(len(piano_roll) - 1, 0, -1):
        
        for pitch in range(len(piano_roll[interval])):

            # If a note is being played, add a MIDI note event
            if piano_roll[interval][pitch] == PLAY:
                if playing == False:
                    playing = True
                    curr_note_end_time = interval + 1

                midi.addNote(track=0, channel=0, pitch=lowest_note + pitch,
                            time=interval / rhythmic_precision,
                            duration=(curr_note_end_time - interval) / rhythmic_precision,
                            volume=100)
                
                playing = False

            # If a note is being sustained, keep track of it until
            # the starting interval is known
            elif piano_roll[interval][pitch] == SUSTAIN:
                if playing == False:
                    playing = True
                    curr_note_end_time = interval + 1

            # If there is no note being played, do nothing
            elif piano_roll[interval][pitch] == REST:
                pass

    # Write the MIDI file
    with open(output_file, "wb") as output_midi_file:
        midi.writeFile(output_midi_file)

    return output_file


def midi_to_audio(midi_file, filename, soundfont):
    """
    Converts a MIDI file to an audio (.wav) file.

    Parameters:
        midi_file (string): The path to the MIDI file to be converted.
        filename (string): The desired filename of the audio file.
        soundfont (string): The path to the soundfont (.sf2) file.

    Returns:
        (string): The path to the new MIDI file.
    """

    # Make output folder (if not already existing) for the audio file
    os.makedirs("output", exist_ok=True)
    output_file = os.path.join("output", f"{filename}.wav")

    # Add fluidsynth.exe to environment variables for use
    fluidsynth_path = os.path.join("fluidsynth-2.4.0-win10-x64", "bin", "fluidsynth.exe")
    os.environ["FLUIDSYNTH"] = fluidsynth_path

    # Initialize FluidSynth with the soundfont
    fs = FluidSynth(soundfont)

    # Convert MIDI to audio
    fs.midi_to_audio(midi_file, output_file)

    return output_file





