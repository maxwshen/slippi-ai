import sonnet as snt
import tensorflow as tf

from policies import Policy
from policies import RLPolicy, RLQfunc

def to_time_major(t):
  permutation = list(range(len(t.shape)))
  permutation[0] = 1
  permutation[1] = 0
  return tf.transpose(t, permutation)

class Learner:
  """Imitation learning learner."""

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
  )

  def __init__(self,
      learning_rate: float,
      policy: Policy):
    self.policy = policy
    self.optimizer = snt.optimizers.Adam(learning_rate)
    self.compiled_step = tf.function(self.step)

  def step(self, batch, initial_states, train=True):
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.policy.initial_state(restarting.shape[0]),
        initial_states)

    # switch axes to time-major
    tm_gamestate = tf.nest.map_structure(to_time_major, bm_gamestate)

    with tf.GradientTape() as tape:
      loss, final_states, distances = self.policy.loss(
          tm_gamestate, initial_states)
      mean_loss = tf.reduce_mean(loss)
    stats = dict(loss=mean_loss, distances=distances)

    if train:
      params = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in self.policy.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(mean_loss, params)
      self.optimizer.apply(grads, params)
    return stats, final_states

class RLLearner:
  '''
    Off-policy reinforcement learning learner.
  '''

  DEFAULT_CONFIG = dict(
      learning_rate=1e-4,
  )

  def __init__(self,
      learning_rate: float,
      actorcritic: ActorCritic,
      ):
    self.actorcritic = actorcritic
    self.optimizer = snt.optimizers.Adam(learning_rate)
    self.compiled_step = tf.function(self.step)

  def step(self, batch, initial_states, phase, frozen_nets, train=True):
    '''
      initial_states: (initial_states_q, initial_states_policy)
    '''
    bm_gamestate, restarting = batch

    # reset initial_states where necessary
    restarting = tf.expand_dims(restarting, -1)
    initial_states[0] = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.actorcritic.q_net.initial_state(restarting.shape[0]),
        initial_states[0])
    initial_states[1] = tf.nest.map_structure(
        lambda x, y: tf.where(restarting, x, y),
        self.actorcritic.policy.initial_state(restarting.shape[0]),
        initial_states[1])

    # switch axes to time-major
    tm_gamestate = tf.nest.map_structure(to_time_major, bm_gamestate)

    # Move this into unified policy class, which takes in phase as an argument?
    # Or keep this logic here, and call the correct loss fn in policy class?
    if phase == 'policy evaluation':
      loss_func = self.actorcritic.policy_evaluation_loss
    elif phase == 'policy iteration':
      loss_func = self.actorcritic.policy_iteration_loss

    with tf.GradientTape() as tape:
      loss, final_states = loss_func(tm_gamestate, initial_states, frozen_nets)
      mean_loss = tf.reduce_mean(loss)
    stats = dict(loss=mean_loss, phase=phase)

    if train:
      params = tape.watched_variables()
      watched_names = [p.name for p in params]
      trainable_names = [v.name for v in phase_func.trainable_variables]
      assert set(watched_names) == set(trainable_names)
      grads = tape.gradient(mean_loss, params)
      self.optimizer.apply(grads, params)

    # final_states is (final_state_q, final_state_policy)
    return stats, final_states