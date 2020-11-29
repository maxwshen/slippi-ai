import sonnet as snt
import tensorflow as tf

from controller_heads import ControllerHead
import embed

def get_p1_controller(gamestate, action_repeat):
  p1_controller = gamestate['player'][1]['controller_state']
  return dict(
      controller=p1_controller,
      action_repeat=action_repeat)

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
    inputs = (gamestate, p1_controller_embed)
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
    inputs = (gamestate, p1_controller_embed)
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
    self.policy = policy

    # TODO - make configurable, consider implementing batched sampling
    self.num_policy_samples = 10

  '''
    Policy evaluation and Q-function methods
  '''
  def q_predict(self, gamestate, initial_state_q, net=self.q_net):
    gamestate, action_repeat, rewards = gamestate
    p1_controller = get_p1_controller(gamestate, action_repeat)

    p1_controller_embed = self.controller_head.embed_controller(p1_controller)
    inputs = (gamestate, p1_controller_embed, action_repeat)
    outputs, final_state_q = self.q_net.unroll(inputs, initial_state_q)
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

    # for loop over self.num_policy_samples
    # TODO - Consider implementing batch sampling
    sampled_controller, final_state_policy = self.policy.sample(next_gamestate, initial_state_policy)

    # replace controller in next_gamestate with sampled_controller -> policy_samples
    # TODO 
    sampled_gamestates = None

    # Run frozen q-function on policy_samples
    q_samples = self.q_predict(sampled_gamestates, initial_state_q, net=frozen_q_net)
    
    # Aggregate expected_future_q over samples
    expected_future_q = tf.reduce_mean(q_samples)

    targets = prev_rewards + expected_future_q
    tf.stop_gradient(targets)

    predicted, final_state_q = self.q_predict(prev_gamestate, initial_state_q)
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

