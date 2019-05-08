import tensorflow as tf
import numpy as np
from inverse_rl.models.tf_util import relu_layer, linear, elu_layer, softplus_layer


def make_relu_net(layers=2, dout=1, d_hidden=32):
    def relu_net(x, last_layer_bias=True):
        out = x
        for i in range(layers):
            out = relu_layer(out, dout=d_hidden, name='l%d' % i)
        out = linear(out, dout=dout, name='lfinal', bias=last_layer_bias)
        return out

    return relu_net


def conv_net_airl(x, dout=1, ff_layers=1, ff_d_hidden=16, env_spec=None):
    # undo reshaping based on env_spec
    obs_shape = env_spec.observation_space.shape
    dA = env_spec.action_space.flat_dim
    # x_obs = x[:, :-dA]
    # x_act = x[:, -dA:]
    x_obs = x
    if len(obs_shape) == 2:
        x_obs = tf.reshape(x_obs, (-1, ) + obs_shape + (1, ))
    else:
        x_obs = tf.reshape(x_obs, (-1, ) + obs_shape)

    out = x_obs

    out = tf.layers.conv2d(
        inputs=out,
        filters=2,
        kernel_size=[5, 5],
        strides=2,
        padding='valid',
        activation=tf.nn.relu,
        name='conv_l1')
    out = tf.layers.conv2d(
        inputs=out,
        filters=2,
        kernel_size=[5, 5],
        strides=2,
        padding='valid',
        activation=tf.nn.relu,
        name='conv_l2')
    out_size = np.prod([int(size) for size in out.shape[1:]])
    out_flat = tf.reshape(out, [-1, out_size])

    # concat action
    #out = tf.concat([out_flat, x_act], axis=1)
    out = out_flat
    for i in range(ff_layers):
        out = relu_layer(out, dout=ff_d_hidden, name='ff_l%d' % i)
    out = linear(out, dout=dout, name='ff_lfinal')
    return out


def conv_net_airl_softplus(x,
                           dout=1,
                           ff_layers=1,
                           ff_d_hidden=16,
                           env_spec=None):
    # undo reshaping based on env_spec
    obs_shape = env_spec.observation_space.shape
    dA = env_spec.action_space.flat_dim
    # x_obs = x[:, :-dA]
    # x_act = x[:, -dA:]
    x_obs = x
    if len(obs_shape) == 2:
        x_obs = tf.reshape(x_obs, (-1, ) + obs_shape + (1, ))
    else:
        x_obs = tf.reshape(x_obs, (-1, ) + obs_shape)

    out = x_obs

    out = tf.layers.conv2d(
        inputs=out,
        filters=2,
        kernel_size=[5, 5],
        strides=2,
        padding='valid',
        activation=tf.nn.relu,
        name='conv_l1')
    out = tf.layers.conv2d(
        inputs=out,
        filters=2,
        kernel_size=[5, 5],
        strides=2,
        padding='valid',
        activation=tf.nn.relu,
        name='conv_l2')
    out_size = np.prod([int(size) for size in out.shape[1:]])
    out_flat = tf.reshape(out, [-1, out_size])

    # concat action
    #out = tf.concat([out_flat, x_act], axis=1)
    out = out_flat
    for i in range(ff_layers):
        out = relu_layer(out, dout=ff_d_hidden, name='ff_l%d' % i)
    out = softplus_layer(out, dout=dout, name='lfinal')
    return out


def relu_softplus_net(x, layers=2, dout=1, d_hidden=32):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d' % i)
    out = softplus_layer(out, dout=dout, name='lfinal')
    return out


def relu_net(x, layers=2, dout=1, d_hidden=32):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d' % i)
    out = linear(out, dout=dout, name='lfinal')
    return out


def elu_net(x, layers=2, dout=1, d_hidden=32):
    out = x
    for i in range(layers):
        out = elu_layer(out, dout=d_hidden, name='l%d' % i)
    out = linear(out, dout=dout, name='lfinal')
    return out


def linear_net(x, dout=1):
    out = x
    out = linear(out, dout=dout, name='lfinal')
    return out


def feedforward_energy(obs_act, ff_arch=linear_net):
    # for trajectories, using feedforward nets rather than RNNs
    dimOU = int(obs_act.get_shape()[2])
    orig_shape = tf.shape(obs_act)

    obs_act = tf.reshape(obs_act, [-1, dimOU])
    outputs = ff_arch(obs_act)
    dOut = int(outputs.get_shape()[-1])

    new_shape = tf.stack([orig_shape[0], orig_shape[1], dOut])
    outputs = tf.reshape(outputs, new_shape)
    return outputs


def rnn_trajectory_energy(obs_act):
    """
    Operates on trajectories
    """
    # for trajectories
    dimOU = int(obs_act.get_shape()[2])

    cell = tf.contrib.rnn.GRUCell(num_units=dimOU)
    cell_out = tf.contrib.rnn.OutputProjectionWrapper(cell, 1)
    outputs, hidden = tf.nn.dynamic_rnn(
        cell_out, obs_act, time_major=False, dtype=tf.float32)
    return outputs
