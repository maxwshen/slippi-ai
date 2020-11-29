import datetime
import embed
import itertools
import os
import secrets

import numpy as np
import tensorflow as tf
import tree

import utils

def get_experiment_directory():
  # create directory for tf checkpoints and other experiment artifacts
  today = datetime.date.today()
  expt_tag = f'{today.year}-{today.month}-{today.day}_{secrets.token_hex(8)}'
  expt_dir = f'experiments/{expt_tag}'
  os.makedirs(expt_dir, exist_ok=True)
  return expt_dir

# necessary because our dataset has some mismatching types, which ultimately
# come from libmelee occasionally giving differently-typed data
# Won't be necessary if we re-generate the dataset.
embed_game = embed.make_game_embedding()

def sanitize_game(game):
  """Casts inputs to the right dtype and discard unused inputs."""
  gamestates, counts, rewards = game
  gamestates = embed_game.map(lambda e, a: a.astype(e.dtype), gamestates)
  return gamestates, counts, rewards

def sanitize_batch(batch):
  game, restarting = batch
  game = sanitize_game(game)
  return game, restarting

class TrainManager:
  """Imitation learning training manager."""

  DEFAULT_CONFIG = dict()

  def __init__(self, learner, data_source, step_kwargs={}):
    self.learner = learner
    self.data_source = data_source
    self.hidden_state = learner.policy.initial_state(data_source.batch_size)
    self.step_kwargs = step_kwargs

    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

  def step(self):
    with self.data_profiler:
      batch = sanitize_batch(next(self.data_source))
    with self.step_profiler:
      stats, self.hidden_state = self.learner.compiled_step(
          batch, self.hidden_state, **self.step_kwargs)
    return stats

class RLTrainManager():
  '''
    Off-policy reinforcement learning training manager.

    Handles alternating policy evaluation and policy iteration updates
      Policy evaluation: Update q-function
      Policy iteration: Update policy
    
    Hidden states for recurrent networks are reset at the beginning of each phase, which are not guaranteed to align with the beginning of games. Hidden states are maintained for up to (phase_steps * unroll_length) frames. Note that Learner resets hidden states at the beginning of games as well. 
  '''

  # TODO - Make RLTrainManager configurable with policy_eval_steps and policy_iter_steps
  DEFAULT_CONFIG = dict(
      policy_eval_steps=10,
      policy_iter_steps=10,
      q_freeze_interval=500,
      policy_freeze_interval=500,
  )

  def __init__(self, 
      learner, 
      data_source, 
      policy_eval_steps: int = 10,
      policy_iter_steps: int = 10,
      q_freeze_interval: int = 500,
      policy_freeze_interval: int = 500,
      step_kwargs={}
      ):
    self.learner = learner
    self.data_source = data_source
    self.step_kwargs = step_kwargs

    self.data_profiler = utils.Profiler()
    self.step_profiler = utils.Profiler()

    self.phase_idx_cycler = itertools.cycle((0, 1))
    self.phases = ('policy evaluation', 'policy iteration')
    self.num_steps = (policy_eval_steps, policy_iter_steps)
    self.curr_step = 0
    self.phase_idx = 0

    self.init_hidden_states = (
        learner.actorcritic.q_net.initial_state(data_source.batch_size),
        learner.actorcritic.policy.initial_state(data_source.batch_size))
    self.hidden_states = self.init_hidden_states

    self.freeze_intervals = (q_freeze_interval, policy_freeze_interval)
    self.steps_since_freeze = (0, 0)
    self.frozen_nets = (
        learner.actorcritic.q_net, 
        learner.actorcritic.policy)

  def freeze_net(self, phase):
    # TODO - Check this works. Verify this freezes controller head too
    if phase == 'policy evaluation':
      source = self.learner.actorcritic.q_net
      target = self.frozen_nets[0]
    elif phase == 'policy iteration':
      source = self.learner.actorcritic.policy
      target = self.frozen_nets[1]
    for s, t in zip(source.variables, target.variables):
      t.assign(s)
    return

  def step(self):
    p = self.phase_idx
    phase = self.phases[p]

    with self.data_profiler:
      batch = sanitize_batch(next(self.data_source))
    with self.step_profiler:
      stats, self.hidden_states = self.learner.compiled_step(
          batch, self.hidden_states, phase, self.frozen_nets, **self.step_kwargs)

    # Freeze models
    self.steps_since_freeze[p] += 1
    if self.steps_since_freeze[p] >= self.freeze_intervals[p]:
      # update self.frozen_nets
      # access through self.learner.
      pass
      self.freeze_net(phase)
      self.steps_since_freeze[p] = 0

    # Toggle between phases: policy evaluation and policy iteration
    self.curr_step += 1
    if self.curr_step >= self.num_steps[p]:
      self.phase_idx = next(self.phase_idx_cycler)
      self.curr_step = 0
      self.hidden_states = self.init_hidden_states

    return stats

def log_stats(ex, stats, step=None, sep='.'):
  def log(path, value):
    if isinstance(value, tf.Tensor):
      value = value.numpy()
    if isinstance(value, np.ndarray):
      value = value.mean()
    key = sep.join(map(str, path))
    ex.log_scalar(key, value, step=step)
  tree.map_structure_with_path(log, stats)
