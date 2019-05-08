import tensorflow as tf
import numpy as np
from inverse_rl.models.tf_util import relu_layer, linear
from inverse_rl.utils.hyperparametrized import Hyperparametrized


def ff_relu_net(obs_act):
    l1 = relu_layer(obs_act, dout=32, name='l1')
    l2 = relu_layer(l1, dout=32, name='l2')
    return linear(l2, dout=1, name='lfinal')


class TimeInferenceModel(Hyperparametrized):
    def __init__(self,
                 env_spec,
                 T,
                 train_steps=50,
                 arch=ff_relu_net,
                 name='time_model'):
        self.dO = env_spec.observation_space.flat_dim
        self.dA = env_spec.action_space.flat_dim
        self.train_steps = train_steps

        with tf.variable_scope(name):
            self.obs_t = tf.placeholder(tf.float32, [None, T, self.dO])
            self.act_t = tf.placeholder(tf.float32, [None, T, self.dA])
            self.label_t = tf.placeholder(tf.int32, [None])

            obs_act = tf.concat([self.obs_t, self.act_t],
                                axis=2)  # None x T x (dO+dA)
            logits = arch(obs_act)  # None x T x 1

            self.probs_t = tf.nn.softmax(logits, axis=1)

            self.loss = tf.softmax_cross_entropy_with_logits(
                labels=self.label_t, logits=logits)

            self.train_step = tf.train.AdamOptimizer(lr=1e-3).minimize(
                self.loss)

    def fit(self, paths):
        obs = np.concatenate([path['observations'] for path in paths])
        act = np.concatenate([path['actions'] for path in paths])
        event_probs = np.concatenate([path['event_probs'] for path in paths])

        for t in range(self.train_steps):
            _, loss = tf.get_default_session().run(
                [self.train_step, self.loss],
                feed_dict={
                    self.obs_t: obs,
                    self.act_t: act,
                    self.label_t: event_probs
                })
        return loss

    def predict_probs(self, obs, acts):
        obs = np.expand_dims(obs, axis=0)
        acts = np.expand_dims(acts, axis=0)
        probs = tf.get_default_session().run(
            self.probs_t, feed_dict={
                self.obs_t: obs,
                self.act_t: act
            })
        return probs[0]
