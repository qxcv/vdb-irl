import tensorflow as tf
import numpy as np
from inverse_rl.models.tf_util import relu_layer, linear, elu_layer, softplus_layer

# Use the architectures in this files for the actual function approximators in AIRL
# For the general classes of basis function approximators or basis weight approximators
# Check out basis_airl_architectures.py

# Contents:

# relu_net, relu_softplus_net, relu_sigmoid_net, relu_clip_net
# Neural network with ReLU activations on hidden layers, plus configurable output nonlinearity

# distance_function_net
# Learns a distribution N(\mu, \Sigma), where \Sigma is diagonal, and defines f(x) = - \log p(x)

# independent_net, independent_sigmoid_net, independent_relu_net
# Each output goes through a separate neural network, so that outputs do not have any sharing


def vae_layer(x, is_train, name=None):
    if name is None:
        name = 'vae'
    else:
        name = name + '/vae'
    with tf.name_scope(name):
        in_dim = x.get_shape().as_list()[-1]
        assert in_dim is not None, "input tensor has variable last dim (?)"
        mean = linear(
            x,
            in_dim,
            name='mean',
            w_init=tf.initializers.random_uniform(-0.01, 0.01))
        logstd = linear(
            x,
            in_dim,
            name='logstd',
            w_init=tf.initializers.random_uniform(-0.01, 0.01))
        std = tf.exp(logstd, name='exp')
        noise = tf.random_normal(tf.shape(x), name='eps')
        noise_zeros = tf.zeros_like(noise)
        noise_cond = tf.case([(is_train, lambda: noise)],
                             default=lambda: noise_zeros)
        reparam = std * noise_cond + mean
        return reparam, mean, logstd


def relu_net(x, layers=2, dout=1, d_hidden=32, vae=False, is_train=None):
    out = x
    for i in range(layers):
        out = relu_layer(out, dout=d_hidden, name='l%d' % i)
    if vae:
        assert is_train is not None
        out, mean, logstd = vae_layer(out, is_train=is_train)
        out = linear(out, dout=dout, name='lfinal')
        return out, mean, logstd
    out = linear(out, dout=dout, name='lfinal')
    return out


def relu_softplus_net(x, layers=2, dout=1, d_hidden=32):
    return tf.nn.softplus(
        relu_net(x, layers=layers, dout=dout, d_hidden=d_hidden))


def relu_sigmoid_net(x, layers=2, dout=1, d_hidden=32, scale=5):
    return scale * tf.nn.sigmoid(
        relu_net(x, layers=layers, dout=dout, d_hidden=d_hidden))


def relu_clip_net(x, layers=2, dout=1, d_hidden=32, scale=5):
    return tf.clip_by_value(
        relu_net(x, layers=layers, dout=dout, d_hidden=d_hidden), 0, scale)


def distance_function_net(x, **kwargs):
    dX = int(x.get_shape()[-1])
    mu = tf.get_variable('mu', shape=(1, dX))
    logstd = tf.get_variable('logstd', shape=(1, dX))
    ll = ((obs - mean)**2 * tf.exp(-2 * logstd) + logstd + 6.28)
    ll = tf.reduce_sum(ll, axis=1, keep_dims=True)
    return -1 * ll


def independent_net(x,
                    base_class=relu_net,
                    layers=2,
                    dout=1,
                    d_hidden=32,
                    **kwargs):
    outputs = []
    for n in range(dout):
        with tf.variable_scope('column%d' % n):
            outputs.append(
                base_class(
                    x, layers=layers, dout=1, d_hidden=d_hidden, **kwargs))
    return tf.concat(outputs, 1)


def independent_sigmoid_net(x, layers=2, dout=1, d_hidden=32, scale=5):
    outputs = independent_net(
        x, base_class=relu_net, layers=layers, dout=dout, d_hidden=d_hidden)
    s = scale * tf.nn.sigmoid(outputs)
    return s


def independent_relu_net(x, layers=2, dout=1, d_hidden=32):
    return independent_net(
        x, base_class=relu_net, layers=layers, dout=dout, d_hidden=d_hidden)
