import tensorflow as tf
import numpy as np

REG_VARS = 'reg_vars'


def linear(X, dout, name, bias=True, w_init=None):
    with tf.variable_scope(name):
        dX = int(X.get_shape()[-1])

        W = tf.get_variable('W', shape=(dX, dout), initializer=w_init)
        tf.add_to_collection(REG_VARS, W)
        if bias:
            b = tf.get_variable(
                'b',
                initializer=tf.constant(np.zeros(dout).astype(np.float32)))
        else:
            b = 0
    return tf.matmul(X, W) + b


def batch_grad_penalty(ys, xs):
    """Compute ||\nabla dys/dxs||^2 across batch (assumed to be first
    dimension). Useful for zero-centred gradient penalties."""
    # for grad penalty computation purposes it's fine that tf.gradients() just
    # sums over ys, since each of the xs[b] contributes only to ys[b]
    grads, = tf.gradients(ys, xs)
    sq_grads = tf.square(grads)
    penalty = tf.reduce_sum(tf.reduce_mean(sq_grads, axis=0))
    return penalty


def discounted_reduce_sum(X, discount, axis=-1):
    if discount != 1.0:
        disc = tf.cumprod(discount * tf.ones_like(X), axis=axis)
    else:
        disc = 1.0
    return tf.reduce_sum(X * disc, axis=axis)


def assert_shape(tens, shape):
    assert tens.get_shape().is_compatible_with(shape)


def relu_layer(X, dout, name):
    return tf.nn.relu(linear(X, dout, name))


def elu_layer(X, dout, name):
    return tf.nn.elu(linear(X, dout, name))


def softplus_layer(X, dout, name):
    return tf.nn.softplus(linear(X, dout, name))


def tanh_layer(X, dout, name):
    return tf.nn.tanh(linear(X, dout, name))


def get_session_config():
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    #session_config.gpu_options.per_process_gpu_memory_fraction = 0.2
    return session_config


def load_prior_params(pkl_fname):
    import joblib
    with tf.Session(config=get_session_config()):
        params = joblib.load(pkl_fname)
    tf.reset_default_graph()
    #joblib.dump(params, file_name, compress=3)
    params = params['irl_params']
    #print(params)
    assert params is not None
    return params
