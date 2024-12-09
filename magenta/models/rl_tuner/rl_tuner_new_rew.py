import numpy as np
from magenta.models.rl_tuner.rl_tuner import RLTuner
import tensorflow.compat.v1 as tf

NOTE_OFF = 0
NO_EVENT = 1


class ContourSmoothnessTuner(RLTuner):
    """RLTuner variant focusing on melodic contour smoothness."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_allowed_interval = 5
        self.smoothness_scaler = 1.0
        
    def reward_music_theory(self, action):
        reward = super().reward_music_theory(action)
        reward += self.reward_contour_smoothness(action)
        return reward
        
    def reward_contour_smoothness(self, action):
        if not self.composition:
            return 0.0
            
        prev_note = self.composition[-1]
        current_note = np.argmax(action)
        
        if current_note in (NOTE_OFF, NO_EVENT):
            return 0.0
            
        interval = abs(current_note - prev_note)
        return self.smoothness_scaler * (0.05 if interval <= self.max_allowed_interval else -0.1)


class DissonanceAwareTuner(RLTuner):
    """RLTuner variant that penalizes dissonant intervals."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.DISSONANT_INTERVALS = [1, 4, 6, 11]  # Minor second, Major third, Tritone, Major seventh
        self.dissonance_scaler = -0.2
        
    def reward_music_theory(self, action):
        reward = super().reward_music_theory(action)
        reward += self.reward_avoid_dissonant_intervals(action)
        return reward
        
    def reward_avoid_dissonant_intervals(self, action):
        if not self.composition:
            return 0.0
            
        prev_note = self.composition[-1]
        current_note = np.argmax(action)
        
        if current_note in (NOTE_OFF, NO_EVENT):
            return 0.0
            
        interval = abs(current_note - prev_note)
        return self.dissonance_scaler if interval in self.DISSONANT_INTERVALS else 0.0


class CadenceAwareTuner(RLTuner):
    """RLTuner variant that emphasizes proper cadences."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cadence_scaler = 2.0
        
    def reward_music_theory(self, action):
        reward = super().reward_music_theory(action)
        if self.beat == self.num_notes_in_melody - 1:
            reward += self.reward_cadence(action)
        return reward
        
    def reward_cadence(self, action):
        current_note = np.argmax(action)
        return self.cadence_scaler if current_note == self.note_rnn_hparams.tonic_note else 0.0


class PhraseSymmetryTuner(RLTuner):
    """RLTuner variant that rewards symmetrical phrases."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.symmetry_scaler = 1.5
        self.min_phrase_length = 4
        
    def reward_music_theory(self, action):
        reward = super().reward_music_theory(action)
        reward += self.reward_phrase_symmetry(action)
        return reward
        
    def reward_phrase_symmetry(self, action):
        if len(self.composition) < 2 * self.min_phrase_length:
            return 0.0
            
        last_phrase = self.composition[-self.min_phrase_length:]
        current_note = np.argmax(action)
        extended_phrase = last_phrase + [current_note]
        
        half = self.min_phrase_length // 2
        first_half = extended_phrase[:half]
        second_half = extended_phrase[-half:]
        
        return self.symmetry_scaler * 0.5 if first_half == second_half[::-1] else 0.0


class DynamicRangeTuner(RLTuner):
    """RLTuner variant that encourages use of full pitch range."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LOW_NOTE_THRESHOLD = 50
        self.HIGH_NOTE_THRESHOLD = 78
        self.range_scaler = 0.5
        
    def reward_music_theory(self, action):
        reward = super().reward_music_theory(action)
        reward += self.reward_dynamic_range(action)
        return reward
        
    def reward_dynamic_range(self, action):
        current_note = np.argmax(action)
        
        if current_note < self.LOW_NOTE_THRESHOLD:
            return self.range_scaler * 0.2
        if current_note > self.HIGH_NOTE_THRESHOLD:
            return self.range_scaler * 0.2
        return 0.0


class DynamicMotionTuner(RLTuner):
    """RLTuner variant that rewards melodic direction changes."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_scaler = 0.1
        
    def reward_music_theory(self, action):
        reward = super().reward_music_theory(action)
        reward += self.reward_dynamic_motion(action)
        return reward
        
    def reward_dynamic_motion(self, action):
        if len(self.composition) < 2:
            return 0.0
            
        prev_note = self.composition[-1]
        current_note = np.argmax(action)
        
        if current_note in (NOTE_OFF, NO_EVENT):
            return 0.0
            
        interval = current_note - prev_note
        prev_interval = self.composition[-2] - self.composition[-1]
        
        direction_change = (interval > 0 and prev_interval < 0) or (interval < 0 and prev_interval > 0)
        return self.motion_scaler * 0.2 if direction_change else 0.0