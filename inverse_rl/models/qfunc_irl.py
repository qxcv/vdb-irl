import tensorflow as tf
import numpy as np
from sandbox.rocky.tf.spaces.box import Box

from inverse_rl.models.fusion_manager import RamFusionDistr
from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator


class AIRL(SingleTimestepIRL):
    """ 
    Fits advantage function based reward functions
    """

    def __init__(self,
                 env,
                 expert_trajs=None,
                 discrim_arch=relu_net,
                 discrim_arch_args={},
                 normalize_reward=False,
                 score_dtau=False,
                 init_itrs=None,
                 discount=1.0,
                 l2_reg=0,
                 state_only=False,
                 shaping_with_actions=False,
                 max_itrs=100,
                 fusion=False,
                 fusion_subsample=0.5,
                 action_penalty=0.0,
                 name='trajprior'):
        super(AIRL, self).__init__()
        env_spec = env.spec
        if fusion:
            self.fusion = RamFusionDistr(100, subsample_ratio=fusion_subsample)
        else:
            self.fusion = None
        self.dO = env_spec.observation_space.flat_dim
        self.dU = env_spec.action_space.flat_dim
        if isinstance(env.action_space, Box):
            self.continuous = True
        else:
            self.continuous = False
        self.normalize_reward = normalize_reward
        self.score_dtau = score_dtau
        self.init_itrs = init_itrs
        self.gamma = discount
        #assert fitted_value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.max_itrs = max_itrs

        # build energy model
        with tf.variable_scope(name) as _vs:
            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(
                tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(
                tf.float32, [None, self.dO], name='nobs')
            self.act_t = tf.placeholder(
                tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(
                tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(
                tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            #obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
            with tf.variable_scope('discrim') as dvs:
                if self.state_only:
                    with tf.variable_scope('energy') as vs:
                        # reward function (or q-function)
                        self.energy = discrim_arch(
                            self.obs_t, dout=1, **discrim_arch_args)
                        energy_vars = tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                else:
                    if self.continuous:
                        obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
                        with tf.variable_scope('energy') as vs:
                            # reward function (or q-function)
                            self.energy = discrim_arch(
                                obs_act, dout=1, **discrim_arch_args)
                            energy_vars = tf.get_collection(
                                tf.GraphKeys.TRAINABLE_VARIABLES,
                                scope=vs.name)
                    else:
                        raise ValueError()

                if shaping_with_actions:
                    nobs_act = tf.concat([self.nobs_t, self.nact_t], axis=1)
                    obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
                else:
                    nobs_act = self.nobs_t
                    obs_act = self.obs_t

                # with tf.variable_scope('vfn'):
                #     fitted_value_fn_n = fitted_value_fn_arch(nobs_act, dout=1)
                # with tf.variable_scope('vfn', reuse=True):
                #     self.value_fn = fitted_value_fn = fitted_value_fn_arch(obs_act, dout=1)

                self.value_fn = tf.zeros(shape=[])

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)

                if action_penalty > 0:
                    self.r = r = -self.energy + action_penalty * tf.reduce_sum(
                        tf.square(self.act_t), axis=1, keepdims=True)
                else:
                    self.r = r = -self.energy

                self.qfn = r  #+self.gamma*fitted_value_fn_n
                log_p_tau = r  #  + self.gamma*fitted_value_fn_n - fitted_value_fn
                discrim_vars = tf.get_collection('reg_vars', scope=dvs.name)

            log_q_tau = self.lprobs

            if l2_reg > 0:
                reg_loss = l2_reg * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)
            self.d_tau = tf.exp(log_p_tau - log_pq)
            cent_loss = -tf.reduce_mean(self.labels * (log_p_tau - log_pq) +
                                        (1 - self.labels) *
                                        (log_q_tau - log_pq))

            self.loss = cent_loss
            tot_loss = self.loss + reg_loss
            self.step = tf.train.AdamOptimizer(
                learning_rate=self.lr).minimize(tot_loss)
            self._make_param_ops(_vs)

    def fit(self,
            paths,
            policy=None,
            batch_size=32,
            logger=None,
            lr=1e-3,
            last_timestep_only=False,
            **kwargs):

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths + old_paths

        self._compute_path_probs(paths, insert=True)

        #self.eval_expert_probs(paths, policy, insert=True)
        for traj in self.expert_trajs:
            if 'agent_infos' in traj:
                #print('deleting agent_infos')
                del traj['agent_infos']
                del traj['a_logprobs']
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)

        obs, obs_next, acts, acts_next, path_probs = self.extract_paths(
            paths,
            keys=('observations', 'observations_next', 'actions',
                  'actions_next', 'a_logprobs'),
            last_timestep_only=last_timestep_only)
        expert_obs, expert_obs_next, expert_acts, expert_acts_next, expert_probs = self.extract_paths(
            self.expert_trajs,
            keys=('observations', 'observations_next', 'actions',
                  'actions_next', 'a_logprobs'),
            last_timestep_only=last_timestep_only)

        # Train discriminator
        for it in TrainingIterator(self.max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs, batch_size=batch_size)

            nexpert_obs_batch, expert_obs_batch, nexpert_act_batch, expert_act_batch, expert_lprobs_batch = \
                self.sample_batch(expert_obs_next, expert_obs, expert_acts_next, expert_acts, expert_probs, batch_size=batch_size)

            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate([nobs_batch, nexpert_obs_batch],
                                        axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate([nact_batch, nexpert_act_batch],
                                        axis=0)
            lprobs_batch = np.expand_dims(
                np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0),
                axis=1).astype(np.float32)

            feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr
            }
            loss, _ = tf.get_default_session().run([self.loss, self.step],
                                                   feed_dict=feed_dict)

            it.record('loss', loss)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            #obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            #logZ,
            energy, logZ, dtau = tf.get_default_session().run(
                [self.energy, self.value_fn, self.d_tau],
                feed_dict={
                    self.act_t: acts,
                    self.obs_t: obs,
                    self.nobs_t: obs_next,
                    self.nact_t: acts_next,
                    self.lprobs: np.expand_dims(path_probs, axis=1)
                })
            logger.record_tabular('GCLLogZ', np.mean(logZ))
            logger.record_tabular('GCLAverageEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageLogPtau', np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageLogQtau', np.mean(path_probs))
            logger.record_tabular('GCLMedianLogQtau', np.median(path_probs))
            logger.record_tabular('GCLAverageDtau', np.mean(dtau))

            #expert_obs_next = np.r_[expert_obs_next, np.expand_dims(expert_obs_next[-1], axis=0)]
            energy, logZ, dtau = tf.get_default_session().run(
                [self.energy, self.value_fn, self.d_tau],
                feed_dict={
                    self.act_t: expert_acts,
                    self.obs_t: expert_obs,
                    self.nobs_t: expert_obs_next,
                    self.nact_t: expert_acts_next,
                    self.lprobs: np.expand_dims(expert_probs, axis=1)
                })
            logger.record_tabular('GCLAverageExpertEnergy', np.mean(energy))
            logger.record_tabular('GCLAverageExpertLogPtau',
                                  np.mean(-energy - logZ))
            logger.record_tabular('GCLAverageExpertLogQtau',
                                  np.mean(expert_probs))
            logger.record_tabular('GCLMedianExpertLogQtau',
                                  np.median(expert_probs))
            logger.record_tabular('GCLAverageExpertDtau', np.mean(dtau))
        return mean_loss

    def eval(self, paths, gamma=1.0, **kwargs):
        """
        Return bonus
        """
        if self.score_dtau:
            self._compute_path_probs(paths, insert=True)
            obs, obs_next, acts, path_probs = self.extract_paths(
                paths,
                keys=('observations', 'observations_next', 'actions',
                      'a_logprobs'))
            path_probs = np.expand_dims(path_probs, axis=1)
            scores = tf.get_default_session().run(
                self.d_tau,
                feed_dict={
                    self.act_t: acts,
                    self.obs_t: obs,
                    self.nobs_t: obs_next,
                    self.lprobs: path_probs
                })
            score = np.log(scores) - np.log(1 - scores)
            score = score[:, 0]
        else:
            obs, acts = self.extract_paths(paths)
            energy = tf.get_default_session().run(
                self.energy, feed_dict={
                    self.act_t: acts,
                    self.obs_t: obs
                })
            score = (-energy)[:, 0]
        return self.unpack(score, paths)

    def eval_discrim(self, paths):
        self._compute_path_probs(paths, insert=True)
        obs, obs_next, acts, path_probs = self.extract_paths(
            paths,
            keys=('observations', 'observations_next', 'actions',
                  'a_logprobs'))
        path_probs = np.expand_dims(path_probs, axis=1)
        scores = tf.get_default_session().run(
            self.d_tau,
            feed_dict={
                self.act_t: acts,
                self.obs_t: obs,
                self.nobs_t: obs_next,
                self.lprobs: path_probs
            })
        score = (scores)
        score = score[:, 0]
        return self.unpack(score, paths)

    def eval_single(self, obs):
        energy = tf.get_default_session().run(
            self.energy, feed_dict={self.obs_t: obs})
        score = (-energy)[:, 0]
        return score

    def debug_eval(self, paths, **kwargs):
        obs, acts = self.extract_paths(paths)
        energy, v, qfn = tf.get_default_session().run(
            [self.energy, self.value_fn, self.qfn],
            feed_dict={
                self.act_t: acts,
                self.obs_t: obs
            })
        return {
            'reward': -energy,
            'value': v,
            'qfn': qfn,
        }
        return {}
