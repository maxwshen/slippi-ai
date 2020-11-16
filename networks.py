import sonnet as snt
import tensorflow as tf

DEFAULT_CONFIG = dict(
  name='mlp',
  mlp=dict(
    output_sizes=[256, 128],
    dropout_rate=0.0,
  ),
)

LSTM_CONFIG = dict(
  name='lstm',
  lstm=dict(hidden_size=196)
)

GRU_CONFIG = dict(
  name='gru',
  gru=dict(hidden_size=196)
)

'''
  Network class wrappers
'''
class MLP(snt.Module):
  def __init__(self, output_sizes, **params):
    super(MLP, self).__init__()
    self.network = snt.nets.MLP(output_sizes, activate_final=True, **params)

  def __call__(self, data, restarting = None):
    return self.network(data)


class LSTM(snt.Module):
  def __init__(self, hidden_size, **params):
    super(LSTM, self).__init__()
    # with self._enter_variable_scope():
    self.network = snt.LSTM(hidden_size, **params)
    self.hidden_size = hidden_size

  # @snt.once
  def _init_states(self, data):
    self.init_state = self.network.initial_state(batch_size = data.shape[1])
    self.state = self.init_state

  def partial_restart_state(self, restart_mask, keep_mask):
    from sonnet import LSTMState
    expand = lambda x: tf.expand_dims(x, axis = -1)
    restart_mask = expand(restart_mask)
    keep_mask = expand(keep_mask)
    return LSTMState(
      hidden = tf.multiply(self.state.hidden, restart_mask) + \
        tf.multiply(self.init_state.hidden, keep_mask),
      cell = tf.multiply(self.state.cell, restart_mask) + \
        tf.multiply(self.init_state.cell, keep_mask)
    )

  def __call__(self, data, restarting):
    '''
      Data: (num_timepoints, batch_len, x_dim)

      Stateless LSTM - state is not transferred between batches
      If we want to use a stateful LSTM, can try Keras LSTM.

      If restarting is guaranteed to never be True except at the beginning of a batch, for loop can be replaced with tf.nn.dynamic_rnn: https://stackoverflow.com/questions/56465346/lstm-timesteps-in-sonnet 
    '''
    self._init_states(data)
    rs_mat = tf.cast(restarting, tf.float32) 
    not_rs_mat = tf.cast(~restarting, tf.float32)

    outputs = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)
    for t in tf.range(data.shape[0]):
      if tf.reduce_any(restarting[t]):
        self.state = self.partial_restart_state(rs_mat[t], not_rs_mat[t])

      out_t, self.state = self.network(data[t], self.state)
      outputs = outputs.write(outputs.size(), out_t)

    outputs = outputs.stack()
    return outputs


class GRU(snt.Module):
  def __init__(self, hidden_size, **params):
    super(GRU, self).__init__()
    # with self._enter_variable_scope():
    self.network = snt.GRU(hidden_size, **params)
    self.hidden_size = hidden_size

  # @snt.once
  def _init_states(self, data):
    self.init_state = self.network.initial_state(batch_size = data.shape[1])
    self.state = self.init_state

  def partial_restart_state(self, restart_mask, keep_mask):
    expand = lambda x: tf.expand_dims(x, axis = -1)
    restart_mask = expand(restart_mask)
    keep_mask = expand(keep_mask)
    return tf.multiply(self.state, restart_mask) + \
        tf.multiply(self.init_state, keep_mask)

  def __call__(self, data, restarting):
    '''
      Data: (num_timepoints, batch_len, x_dim)

      Stateless - state is not transferred between batches

      If restarting is guaranteed to never be True except at the beginning of a batch, for loop can be replaced with tf.nn.dynamic_rnn: https://stackoverflow.com/questions/56465346/lstm-timesteps-in-sonnet 
    '''
    self._init_states(data)
    rs_mat = tf.cast(restarting, tf.float32) 
    not_rs_mat = tf.cast(~restarting, tf.float32)

    outputs = tf.TensorArray(dtype = tf.float32, size = 0, dynamic_size = True)
    for t in tf.range(data.shape[0]):
      if tf.reduce_any(restarting[t]):
        self.state = self.partial_restart_state(rs_mat[t], not_rs_mat[t])

      out_t, self.state = self.network(data[t], self.state)
      outputs = outputs.write(outputs.size(), out_t)

    outputs = outputs.stack()
    return outputs


CONSTRUCTORS = dict(
  mlp=MLP,
  lstm=LSTM,
  gru=GRU,
)

def construct_network(name, **config):
  return CONSTRUCTORS[name](**config[name])
