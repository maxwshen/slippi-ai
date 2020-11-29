import sonnet as snt
import tensorflow as tf

from controller_heads import ControllerHead
import embed

def get_p1_controller(gamestate, action_repeat):
  p1_controller = gamestate['player'][1]['controller_state']
  return dict(
      controller=p1_controller,
      action_repeat=action_repeat)

def write_p1_controller(gamestate, sampled_controller_with_repeat):
  # Replace observed p1 controller with samples from policy
  unpacked_gamestate, action_repeat, rewards = gamestate
  sampled_controller = sampled_controller_with_repeat['controller']
  sampled_action_repeat = sampled_controller_with_repeat['action_repeat']

  written_gamestate = unpacked_gamestate
  written_gamestate['player'][1]['controller_state'] = sampled_controller

  return (written_gamestate, sampled_action_repeat, rewards)

class Policy(snt.Module):
  """Imitation learning policy."""

  def __init__(self, network, controller_head: ControllerHead):
    super().__init__(name='Policy')
    self.network = network
    self.controller_head = controller_head
    self.initial_state = self.network.initial_state

  def loss(self, gamestate, initial_state):
    gamestate, action_repeat, rewards = gamestate
    p1_controller = get_p1_controller(gamestate, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestate, p1_controller_embed, action_repeat)
    outputs, final_state = self.network.unroll(inputs, initial_state)

    prev_action = tf.nest.map_structure(lambda t: t[:-1], p1_controller)
    next_action = tf.nest.map_structure(lambda t: t[1:], p1_controller)

    distances = self.controller_head.distance(
        outputs[:-1], prev_action, next_action)
    loss = tf.add_n(tf.nest.flatten(distances))

    return loss, final_state, distances

  def sample(self, gamestate, initial_state):
    gamestate, action_repeat, rewards = gamestate
    p1_controller = get_p1_controller(gamestate, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestate, p1_controller_embed, action_repeat)
    output, final_state = self.network.step(inputs, initial_state)

    controller_sample = self.controller_head.sample(
        output, p1_controller)
    return controller_sample, final_state

class ActorCritic(snt.Module):
  '''
    Actor Critic for off-policy reinforcement learning.

    Maintains a policy (actor) = p(a|s) and a Q-function (critic).

    Learning alternates between:
      Policy evaluation: Updating the Q-function using a frozen Q-function and policy
      Policy iteration: Updating the policy using a frozen Q-function and policy
  '''

  def __init__(self, network, policy: Policy):
    super().__init__(name='ActorCritic')
    self.q_net = network
    self.q_head = snt.Linear(1)

    self.policy = policy

    # TODO - make configurable, consider implementing batched sampling
    self.num_policy_samples = 10

  '''
    Policy evaluation and Q-function methods
  '''
  def q_predict(self, gamestate, initial_state_q, net):
    gamestate, action_repeat, rewards = gamestate
    p1_controller = get_p1_controller(gamestate, action_repeat)

    p1_controller_embed = self.policy.controller_head.embed_controller(p1_controller)
    inputs = (gamestate, p1_controller_embed, action_repeat)
    outputs, final_state_q = net.unroll(inputs, initial_state_q)
    outputs = self.q_head(outputs)
    return outputs, final_state_q

  def policy_evaluation_loss(self, gamestate, initial_states, frozen_nets):
    '''
      One-step temporal difference loss

      TODO - Consider multistep TD loss with tree-backup importance weighting (Precup et al., 2000), or Retrace importance weighting (Munoz et al., 2016), though Retrace would require precalculating global summary statistics.
    '''
    unpacked_gamestate, action_repeat, rewards = gamestate
    initial_state_q, initial_state_policy = initial_states
    frozen_q_net, frozen_policy = frozen_nets

    prev_rewards = tf.nest.map_structure(lambda t: t[:-1], rewards)
    prev_gamestate = tf.nest.map_structure(lambda t: t[:-1], gamestate)
    next_gamestate = tf.nest.map_structure(lambda t: t[1:], gamestate)

    # TODO next: for loop over self.num_policy_samples
    # TODO - Consider implementing batch sampling
    sampled_controller_with_repeat, final_state_policy = self.policy.sample(
        gamestate, initial_state_policy)
    next_sampled_controller_with_repeat = tf.nest.map_structure(lambda t: t[1:],
        sampled_controller_with_repeat)

    # replace controller in next_gamestate with sampled controller actions from policy
    next_sampled_gamestate = write_p1_controller(next_gamestate,
        next_sampled_controller_with_repeat)

    # Run frozen q-function on policy_samples
    q_samples, _ = self.q_predict(next_sampled_gamestate,
        initial_state_q, frozen_q_net)
    # shape: (T, B, 1) with q_head
    import pdb; pdb.set_trace()
    
    # Aggregate expected_future_q over policy samples
    expected_future_q = tf.reduce_mean(q_samples, axis=-1)

    # prev_rewards: (T, B)
    targets = prev_rewards + expected_future_q
    tf.stop_gradient(targets)

    predicted, final_state_q = self.q_predict(prev_gamestate, initial_state_q, self.q_net)
    predicted = tf.squeeze(predicted, axis=-1)
    # predicted / targets: (T, B)
    loss = tf.reduce_mean(tf.square(predicted - targets))

    return loss, (final_state_q, final_state_policy)

  '''
    Policy iteration and Q-function methods
  '''
  def policy_iteration_loss(self, gamestate, initial_states, frozen_nets):
    unpacked_gamestate, action_repeat, rewards = gamestate
    initial_state_q, initial_state_policy = initial_states
    frozen_q_net, frozen_policy_net = frozen_nets

    return loss, (final_state_q, final_state_policy)

