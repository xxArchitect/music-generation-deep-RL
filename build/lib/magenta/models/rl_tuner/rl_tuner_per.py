import numpy as np
from magenta.models.rl_tuner.rl_tuner import RLTuner
from collections import namedtuple
import tensorflow.compat.v1 as tf


class PrioritizedExperienceBuffer:
    """A prioritized experience replay buffer."""
    
    Experience = namedtuple('Experience', 
                          ['observation', 'state', 'action', 'reward', 
                           'newobservation', 'newstate', 'new_reward_state'])
    
    def __init__(self, max_size=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """Initialize prioritized experience replay buffer.
        
        Args:
            max_size: Maximum size of the buffer
            alpha: How much prioritization to use (0=none, 1=full)
            beta_start: Initial value of beta for importance sampling weights
            beta_frames: Number of frames over which to linearly anneal beta to 1.0
        """
        self.max_size = max_size
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
    def beta_by_frame(self, frame_idx):
        """Calculate beta value for importance sampling weights."""
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, *args):
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.max_size:
            self.memory.append(None)
        
        self.memory[self.position] = self.Experience(*args)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.max_size
    
    def sample(self, batch_size):
        """Sample a batch of experiences with importance sampling weights.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            tuple containing:
                batch of experiences
                indices of sampled experiences
                importance sampling weights
        """
        if len(self.memory) == 0:
            return None, None, None
            
        if len(self.memory) < self.max_size:
            priorities = self.priorities[:len(self.memory)]
        else:
            priorities = self.priorities
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # Calculate weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get selected experiences
        experiences = [self.memory[idx] for idx in indices]
        
        return experiences, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        """Return current size of memory."""
        return len(self.memory)
    

class RLTunerPER(RLTuner):
    """Extension of RLTuner with Prioritized Experience Replay."""
    
    def __init__(self, *args, **kwargs):
        # Add PER specific parameters
        self.per_alpha = kwargs.pop('per_alpha', 0.6)
        self.per_beta_start = kwargs.pop('per_beta_start', 0.4)
        self.per_beta_frames = kwargs.pop('per_beta_frames', 100000)
        self.per_epsilon = kwargs.pop('per_epsilon', 1e-6)
        
        super(RLTunerPER, self).__init__(*args, **kwargs)
        
        # Replace standard experience buffer with prioritized version
        self.experience = PrioritizedExperienceBuffer(
            max_size=self.dqn_hparams.max_experience,
            alpha=self.per_alpha,
            beta_start=self.per_beta_start,
            beta_frames=self.per_beta_frames
        )

    def store(self, observation, state, action, reward, newobservation, newstate, new_reward_state):
        """Stores an experience in the prioritized replay buffer.

        Args:
            observation: A one hot encoding of an observed note.
            state: The internal state of the q_network MelodyRNN LSTM model.
            action: A one hot encoding of action taken by network.
            reward: Reward received for taking the action.
            newobservation: The next observation that resulted from the action.
            newstate: The internal state of the q_network MelodyRNN that is
                observed after taking the action.
            new_reward_state: The internal state of the reward_rnn network that is
                observed after taking the action
        """
        if self.num_times_store_called % self.dqn_hparams.store_every_nth == 0:
            self.experience.push(observation, state, action, reward,
                               newobservation, newstate, new_reward_state)
        self.num_times_store_called += 1

    def build_graph(self):
        with self.graph.as_default():
            tf.logging.info('Adding reward computation portion of the graph')
            with tf.name_scope('reward_computation'):
                self.reward_scores = tf.identity(self.reward_rnn(), name='reward_scores')

            tf.logging.info('Adding taking action portion of graph')
            with tf.name_scope('taking_action'):
                self.action_scores = tf.identity(self.q_network(), name='action_scores')
                tf.summary.histogram('action_scores', self.action_scores)

                if self.algorithm == 'g':
                    self.g_action_scores = self.action_scores + self.reward_scores
                    self.action_softmax = tf.nn.softmax(self.g_action_scores, name='action_softmax')
                    self.predicted_actions = tf.one_hot(
                        tf.argmax(self.g_action_scores, dimension=1, name='predicted_actions'),
                        self.num_actions)
                else:
                    self.action_softmax = tf.nn.softmax(self.action_scores, name='action_softmax')
                    self.predicted_actions = tf.one_hot(
                        tf.argmax(self.action_scores, dimension=1, name='predicted_actions'),
                        self.num_actions)

            tf.logging.info('Add estimating future rewards portion of graph')
            with tf.name_scope('estimating_future_rewards'):
                self.next_action_scores = tf.stop_gradient(self.target_q_network())
                tf.summary.histogram('target_action_scores', self.next_action_scores)
                self.rewards = tf.placeholder(tf.float32, (None,), name='rewards')

                if self.algorithm == 'psi':
                    self.target_vals = tf.reduce_logsumexp(self.next_action_scores, reduction_indices=[1,])
                elif self.algorithm == 'g':
                    self.g_normalizer = tf.reduce_logsumexp(self.reward_scores, reduction_indices=[1,])
                    self.g_normalizer = tf.reshape(self.g_normalizer, [-1, 1])
                    self.g_normalizer = tf.tile(self.g_normalizer, [1, self.num_actions])
                    self.g_action_scores = tf.subtract(
                        (self.next_action_scores + self.reward_scores), self.g_normalizer)
                    self.target_vals = tf.reduce_logsumexp(self.g_action_scores, reduction_indices=[1,])
                else:
                    self.target_vals = tf.reduce_max(self.next_action_scores, reduction_indices=[1,])

                self.future_rewards = self.rewards + self.discount_rate * self.target_vals

            tf.logging.info('Adding q value prediction portion of graph')
            with tf.name_scope('q_value_prediction'):
                # Add importance sampling weights placeholder
                self.importance_weights = tf.placeholder(
                    tf.float32, [None], name='importance_weights')
                
                self.action_mask = tf.placeholder(
                    tf.float32, (None, self.num_actions), name='action_mask')
                self.masked_action_scores = tf.reduce_sum(
                    self.action_scores * self.action_mask, reduction_indices=[1,])

                temp_diff = self.masked_action_scores - self.future_rewards
                
                # Individual TD errors for updating priorities
                self.td_errors = tf.square(temp_diff)
                
                # Weighted mean squared error
                self.prediction_error = tf.reduce_mean(
                    self.importance_weights * self.td_errors)

                self.params = tf.trainable_variables()
                self.gradients = self.optimizer.compute_gradients(self.prediction_error)

                for i, (grad, var) in enumerate(self.gradients):
                    if grad is not None:
                        self.gradients[i] = (tf.clip_by_norm(grad, 5), var)

                for grad, var in self.gradients:
                    tf.summary.histogram(var.name, var)
                    if grad is not None:
                        tf.summary.histogram(var.name + '/gradients', grad)

                self.train_op = self.optimizer.apply_gradients(self.gradients)

            tf.logging.info('Adding target network update portion of graph')
            with tf.name_scope('target_network_update'):
                self.target_network_update = []
                for v_source, v_target in zip(self.q_network.variables(),
                                            self.target_q_network.variables()):
                    update_op = v_target.assign_sub(
                        self.target_network_update_rate * (v_target - v_source))
                    self.target_network_update.append(update_op)
                self.target_network_update = tf.group(*self.target_network_update)

            tf.summary.scalar('prediction_error', self.prediction_error)

            self.summarize = tf.summary.merge_all()
            self.no_op1 = tf.no_op()

    def training_step(self):
        if self.num_times_train_called % self.dqn_hparams.train_every_nth == 0:
            if len(self.experience) < self.dqn_hparams.minibatch_size:
                return
            
            samples, indices, weights = self.experience.sample(self.dqn_hparams.minibatch_size)
            
            if samples is None:
                return
                
            # Prepare batch data
            states = np.empty((len(samples), self.q_network.cell.state_size))
            new_states = np.empty((len(samples), self.target_q_network.cell.state_size))
            reward_new_states = np.empty((len(samples), self.reward_rnn.cell.state_size))
            observations = np.empty((len(samples), self.input_size))
            new_observations = np.empty((len(samples), self.input_size))
            action_mask = np.zeros((len(samples), self.num_actions))
            rewards = np.empty((len(samples),))
            lengths = np.full(len(samples), 1, dtype=int)

            for i, exp in enumerate(samples):
                observations[i, :] = exp.observation
                new_observations[i, :] = exp.newobservation
                states[i, :] = exp.state
                new_states[i, :] = exp.newstate
                action_mask[i, :] = exp.action
                rewards[i] = exp.reward
                reward_new_states[i, :] = exp.new_reward_state

            observations = np.reshape(observations, (len(samples), 1, self.input_size))
            new_observations = np.reshape(new_observations, (len(samples), 1, self.input_size))

            # Create feed dictionary with importance sampling weights
            feed_dict = {
                self.q_network.melody_sequence: observations,
                self.q_network.initial_state: states,
                self.q_network.lengths: lengths,
                self.target_q_network.melody_sequence: new_observations,
                self.target_q_network.initial_state: new_states,
                self.target_q_network.lengths: lengths,
                self.action_mask: action_mask,
                self.rewards: rewards,
                self.importance_weights: weights
            }
            
            if self.algorithm == 'g':
                feed_dict.update({
                    self.reward_rnn.melody_sequence: new_observations,
                    self.reward_rnn.initial_state: reward_new_states,
                    self.reward_rnn.lengths: lengths,
                })
            
            # Calculate TD errors for updating priorities
            td_errors = self.session.run(self.td_errors, feed_dict)
            new_priorities = np.sqrt(td_errors + self.per_epsilon)
            self.experience.update_priorities(indices, new_priorities)

            calc_summaries = self.iteration % 100 == 0
            calc_summaries = calc_summaries and self.summary_writer is not None

            _, _, target_vals, summary_str = self.session.run([
                self.prediction_error,
                self.train_op,
                self.target_vals,
                self.summarize if calc_summaries else self.no_op1,
            ], feed_dict)

            total_logs = (self.iteration * self.dqn_hparams.train_every_nth)
            if total_logs % self.output_every_nth == 0:
                self.target_val_list.append(np.mean(target_vals))

            self.session.run(self.target_network_update)

            if calc_summaries:
                self.summary_writer.add_summary(summary_str, self.iteration)

            self.iteration += 1

        self.num_times_train_called += 1