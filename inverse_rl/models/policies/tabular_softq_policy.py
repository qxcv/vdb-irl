import tensorflow as tf
import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides
from rllab.spaces import Discrete
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.policies.base import StochasticPolicy


class CategoricalSoftQPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            env_spec,
            hardcoded_q=None,
            scope='policy',
            ent_wt=1.0,
    ):
        """
        :param env_spec: A spec for the env.
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        :return:
        """
        #self.graph = tf.get_default_graph()
        assert isinstance(env_spec.action_space, Discrete)
        Serializable.quick_init(self, locals())
        super(CategoricalSoftQPolicy, self).__init__(env_spec)
        obs_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        self.dist = Categorical(action_dim)
        self.ent_wt = ent_wt
        self.hardcoded_q = hardcoded_q

        with tf.variable_scope(scope) as vs:
            self.vs = vs

            self.q_func = tf.get_variable(
                'q_func', shape=(obs_dim, action_dim))

            self.q_func_plc = tf.placeholder(
                tf.float32, shape=(obs_dim, action_dim))
            self.q_func_assgn = tf.assign(self.q_func, self.q_func_plc)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        obs_var = tf.cast(obs_var, tf.float32)
        qvals = tf.matmul(obs_var, self.q_func)
        probs = tf.nn.softmax((1.0 / self.ent_wt) * qvals, dim=1)
        return dict(prob=probs)

    @property
    def vectorized(self):
        return True

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        if self.hardcoded_q is not None:
            q_func = self.hardcoded_q
        else:
            q_func = tf.get_default_session().run(self.q_func)
        q_vals = flat_obs.dot(q_func)

        # softmax
        qv = (1.0 / self.ent_wt) * q_vals
        qv = qv - np.max(qv, axis=1, keepdims=True)
        probs = np.exp(qv)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        actions = special.weighted_sample_n(probs,
                                            np.arange(self.action_space.n))
        agent_info = dict(prob=probs)
        return actions, agent_info

    def set_q_func(self, q_fn):
        tf.get_default_session().run(
            self.q_func_assgn, feed_dict={self.q_func_plc: q_fn})
        #new_q = tf.get_default_session().run(self.q_func)

    @property
    def distribution(self):
        return self.dist

    def get_params_internal(self, **tags):
        key = tf.GraphKeys.GLOBAL_VARIABLES
        if tags.get('trainable', False):
            key = tf.GraphKeys.TRAINABLE_VARIABLES
        return tf.get_collection(key, scope=self.vs.name)
