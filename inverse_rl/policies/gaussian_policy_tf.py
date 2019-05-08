import numpy as np

from inverse_rl.models.tf_util import linear, relu_layer
from sandbox.rocky.tf.spaces.box import Box
import tensorflow as tf

from rllab.core.serializable import Serializable
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from rllab.misc.overrides import overrides
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils
import tensorflow as tf


def gaussian_ff_arch(obs, env_spec):
    action_dim = env_spec.action_space.flat_dim

    features = tf.nn.relu(linear(obs, dout=64, name='pol_feats1'))
    features = tf.nn.relu(linear(features, dout=64, name='pol_feats2'))

    mean = 0.1 * linear(features, dout=action_dim, name='pol_mean')
    log_std = 0.1 * linear(features, dout=action_dim, name='pol_log_std') - 1
    return mean, log_std


def gaussian_conv_jnt_arch(obs,
                           env_spec,
                           ff_layers=2,
                           ff_d_hidden=64,
                           obs_img_shape=(64, 64, 3),
                           obs_jnt_dims=7):
    #obs_shape = env_spec.observation_space.shape
    action_dim = env_spec.action_space.flat_dim

    x_img = obs[:, :-obs_jnt_dims]
    x_jnt = obs[:, -obs_jnt_dims:]
    if len(obs_img_shape) == 2:
        x_img = tf.reshape(x_img, (-1, ) + obs_img_shape + (1, ))
    else:
        x_img = tf.reshape(x_img, (-1, ) + obs_img_shape)

    out = x_img
    out = tf.layers.conv2d(
        inputs=out,
        filters=2,
        kernel_size=[5, 5],
        strides=2,
        padding='valid',
        activation=tf.nn.relu,
        name='pol_conv_l1')
    out = tf.layers.conv2d(
        inputs=out,
        filters=2,
        kernel_size=[5, 5],
        strides=2,
        padding='valid',
        activation=tf.nn.relu,
        name='pol_conv_l2')
    out_size = np.prod([int(size) for size in out.shape[1:]])
    out_flat = tf.reshape(out, [-1, out_size])

    # concat action
    out = tf.concat([out_flat, x_jnt], axis=1)
    for i in range(ff_layers):
        out = relu_layer(out, dout=ff_d_hidden, name='pol_ff_l%d' % i)
    out = linear(out, dout=ff_d_hidden, name='pol_ff_lfinal')
    features = out

    mean = 0.1 * linear(features, dout=action_dim, name='pol_mean')
    log_std = 0.1 * linear(features, dout=action_dim, name='pol_log_std') - 1
    return mean, log_std


class GaussianTFPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            arch=gaussian_ff_arch,
            arch_args=None,
            min_std=1e-6,
    ):
        """
        :param env_spec:
        :param hidden_sizes: list of sizes for the fully-connected hidden layers
        :param learn_std: Is std trainable
        :param init_std: Initial std
        :param adaptive_std:
        :param std_share_network:
        :param std_hidden_sizes: list of sizes for the fully-connected layers for std
        :param min_std: whether to make sure that the std is at least some threshold value, to avoid numerical issues
        :param std_hidden_nonlinearity:
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :param output_nonlinearity: nonlinearity for the output layer
        :param mean_network: custom network for the output mean
        :param std_network: custom network for the output log std
        :param std_parametrization: how the std should be parametrized. There are a few options:
            - exp: the logarithm of the std will be stored, and applied a exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        :return:
        """
        Serializable.quick_init(self, locals())
        super(GaussianTFPolicy, self).__init__(env_spec)
        assert isinstance(env_spec.action_space, Box)

        if arch_args is None:
            arch_args = {}
        self.arch_args = arch_args
        self.name = name

        self.min_std_param = np.log(min_std)

        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        self._dist = DiagonalGaussian(action_dim)
        self.arch = arch

        with tf.variable_scope(self.name):
            self.sample_obs = tf.placeholder(tf.float32, [None, obs_dim])
            self.sample_mu, self.sample_log_std = self.arch(
                self.sample_obs, env_spec=env_spec, **self.arch_args)

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None):
        #mean_var, std_param_var = L.get_output([self._l_mean, self._l_std_param], obs_var)
        with tf.variable_scope(self.name, reuse=True):
            mean_var, std_param_var = self.arch(
                obs_var, env_spec=self.env_spec, **self.arch_args)

        if self.min_std_param is not None:
            std_param_var = tf.maximum(std_param_var, self.min_std_param)
        log_std_var = std_param_var
        return dict(mean=mean_var, log_std=log_std_var)

    @overrides
    def get_action(self, observation):
        """
        flat_obs = self.observation_space.flatten(observation)
        mean, log_std = [x[0] for x in self._f_dist([flat_obs])]
        rnd = np.random.normal(size=mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, dict(mean=mean, log_std=log_std)
        """
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        assert len(np.array(observations).shape) == 2
        flat_obs = self.observation_space.flatten_n(observations)

        means, log_stds = \
            tf.get_default_session().run([self.sample_mu, self.sample_log_std],
                                        feed_dict={self.sample_obs: flat_obs})
        rnd = np.random.normal(size=means.shape)
        actions = rnd * np.exp(log_stds) + means
        return actions, dict(mean=means, log_std=log_stds)

    def get_reparam_action_sym(self, obs_var, action_var, old_dist_info_vars):
        """
        Given observations, old actions, and distribution of old actions, return a symbolically reparameterized
        representation of the actions in terms of the policy parameters
        :param obs_var:
        :param action_var:
        :param old_dist_info_vars:
        :return:
        """
        new_dist_info_vars = self.dist_info_sym(obs_var, action_var)
        new_mean_var, new_log_std_var = new_dist_info_vars[
            "mean"], new_dist_info_vars["log_std"]
        old_mean_var, old_log_std_var = old_dist_info_vars[
            "mean"], old_dist_info_vars["log_std"]
        epsilon_var = (action_var - old_mean_var) / (
            tf.exp(old_log_std_var) + 1e-8)
        new_action_var = new_mean_var + epsilon_var * tf.exp(new_log_std_var)
        return new_action_var

    def log_diagnostics(self, paths):
        log_stds = np.vstack(
            [path["agent_infos"]["log_std"] for path in paths])
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))

    @property
    def distribution(self):
        return self._dist

    def get_params_internal(self, **tags):
        key = tf.GraphKeys.GLOBAL_VARIABLES
        if tags.get('trainable', False):
            key = tf.GraphKeys.TRAINABLE_VARIABLES
        return tf.get_collection(key, scope=self.name)
