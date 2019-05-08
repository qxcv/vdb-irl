import tensorflow as tf
import numpy as np

from inverse_rl.models.imitation_learning import SingleTimestepIRL
from inverse_rl.models.tf_util import batch_grad_penalty

from sandbox.rocky.tf.spaces.box import Box
from multiple_irl.models.fusion_manager import RamFusionDistr
from multiple_irl.models.architectures import relu_net
from inverse_rl.utils import TrainingIterator, kl_loss


class AIRL(SingleTimestepIRL):
    """
    Fits advantage function based reward functions
    """

    def __init__(
            self,
            env,
            expert_trajs=None,
            discrim_arch=relu_net,
            discrim_arch_args={},
            fitted_value_fn_arch=relu_net,
            fitted_value_fn_arch_args={},
            normalize_reward=False,
            score_dtau=False,
            score_shaped=False,
            init_itrs=None,
            discount=1.0,
            l2_reg=0,
            state_only=True,
            time_in_state=False,
            shaping_with_actions=False,
            fusion=False,
            fusion_subsample=0.5,
            action_penalty=0.0,
            freeze=False,
            name='trajprior',
            gp_coeff=None,
            # VAIRL options
            vairl=False,
            # initial (or fixed) beta
            vairl_beta=1.0,
            # should VAIRL beta be updated over time? If so, at what
            # rate/toward what target KL?
            vairl_adaptive_beta=False,
            vairl_beta_step_size=1e-6,
            vairl_kl_target=0.5, ):

        super(AIRL, self).__init__()
        env_spec = env.spec
        if fusion:
            # This looks way too small. How many samples are we adding at each
            # step? What is the mean & median age of the distribution?
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
        self.score_shaped = score_shaped

        self.init_itrs = init_itrs
        self.gamma = discount

        # assert fitted_value_fn_arch is not None
        self.set_demos(expert_trajs)
        self.state_only = state_only
        self.frozen = freeze
        self.vairl = vairl

        # build energy model
        with tf.variable_scope(name) as _vs:
            self.is_train_t = tf.placeholder_with_default(
                False, shape=(), name='is_train')
            if vairl:
                self.vairl_adaptive_beta = vairl_adaptive_beta
                if self.vairl_adaptive_beta:
                    self.vairl_initial_beta = vairl_beta
                    self.vairl_beta_step_size = vairl_beta_step_size
                    self.vairl_kl_target = vairl_kl_target
                    # now update ops
                    self.vairl_beta = tf.get_variable(
                        'vairl_beta',
                        trainable=False,
                        shape=(),
                        dtype=tf.float32,
                        initializer=tf.initializers.constant(vairl_beta))
                    self.vairl_mean_kl = tf.placeholder(
                        name='vairl_kl_ph', shape=(), dtype=tf.float32)
                    # Make beta higher if mean_kl>target, or lower if
                    # mean_kl<target. Clip up to 0 if necessary.
                    beta_new_value = tf.maximum(
                        0.0, self.vairl_beta + self.vairl_beta_step_size *
                        (self.vairl_mean_kl - self.vairl_kl_target))
                    self.vairl_beta_update_op = tf.assign(
                        self.vairl_beta, beta_new_value)
                else:
                    self.vairl_beta = vairl_beta
                discrim_arch_args = dict(discrim_arch_args)
                fitted_value_fn_arch_args = dict(fitted_value_fn_arch_args)
                fitted_value_fn_arch_args.update({
                    'vae': True,
                    'is_train': self.is_train_t
                })
                discrim_arch_args.update({
                    'vae': True,
                    'is_train': self.is_train_t
                })
                kl_losses = []

            # Should be batch_size x T x dO/dU
            self.obs_t = tf.placeholder(
                tf.float32, [None, self.dO], name='obs')
            self.nobs_t = tf.placeholder(
                tf.float32, [None, self.dO], name='nobs')
            if time_in_state:
                self.obs_t = self.obs_t[:, :-1]
                self.nobs_t = self.nobs_t[:, :-1]

            self.act_t = tf.placeholder(
                tf.float32, [None, self.dU], name='act')
            self.nact_t = tf.placeholder(
                tf.float32, [None, self.dU], name='nact')
            self.labels = tf.placeholder(tf.float32, [None, 1], name='labels')
            self.lprobs = tf.placeholder(
                tf.float32, [None, 1], name='log_probs')
            self.lr = tf.placeholder(tf.float32, (), name='lr')

            # obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
            with tf.variable_scope('discrim') as dvs:
                if self.state_only:
                    with tf.variable_scope('energy') as vs:
                        # reward function (or q-function)
                        if vairl:
                            self.energy, mean, logstd = discrim_arch(
                                self.obs_t, dout=1, **discrim_arch_args)
                            kl_losses.append(kl_loss(mean, logstd))
                        else:
                            self.energy = discrim_arch(
                                self.obs_t, dout=1, **discrim_arch_args)
                        energy_vars = tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)
                    rew_input = self.obs_t
                else:
                    if self.continuous:
                        obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
                        with tf.variable_scope('energy') as vs:
                            # reward function (or q-function)
                            if vairl:
                                self.energy, mean, logstd = discrim_arch(
                                    obs_act, dout=1, **discrim_arch_args)
                                kl_losses.append(kl_loss(mean, logstd))
                            else:
                                self.energy = discrim_arch(
                                    obs_act, dout=1, **discrim_arch_args)
                            energy_vars = tf.get_collection(
                                tf.GraphKeys.TRAINABLE_VARIABLES,
                                scope=vs.name)
                        rew_input = obs_act
                    else:
                        raise ValueError()

                if shaping_with_actions:
                    nobs_act = tf.concat([self.nobs_t, self.nact_t], axis=1)
                    obs_act = tf.concat([self.obs_t, self.act_t], axis=1)
                else:
                    nobs_act = self.nobs_t
                    obs_act = self.obs_t

                with tf.variable_scope('vfn'):
                    if vairl:
                        fitted_value_fn_n, mean, logstd \
                            = fitted_value_fn_arch(
                                nobs_act, dout=1, **fitted_value_fn_arch_args)
                        kl_losses.append(kl_loss(mean, logstd))
                    else:
                        fitted_value_fn_n = fitted_value_fn_arch(
                            nobs_act, dout=1, **fitted_value_fn_arch_args)
                    vfn_n_input = nobs_act
                with tf.variable_scope('vfn', reuse=True):
                    if vairl:
                        self.value_fn, _, _ \
                            = fitted_value_fn, mean, logstd \
                            = fitted_value_fn_arch(
                                obs_act, dout=1, **fitted_value_fn_arch_args)
                        kl_losses.append(kl_loss(mean, logstd))
                    else:
                        self.value_fn = fitted_value_fn = fitted_value_fn_arch(
                            obs_act, dout=1, **fitted_value_fn_arch_args)
                    vfn_input = obs_act

                # self.value_fn = tf.zeros(shape=[])

                # Define log p_tau(a|s) = r + gamma * V(s') - V(s)

                if action_penalty > 0:
                    self.r = r = -self.energy + action_penalty * tf.reduce_sum(
                        tf.square(self.act_t), axis=1, keepdims=True)
                else:
                    self.r = r = -self.energy

                self.qfn = r + self.gamma * fitted_value_fn_n
                log_p_tau = self.qfn - fitted_value_fn
                self.shaped_r = log_p_tau

                discrim_vars = tf.get_collection('reg_vars', scope=dvs.name)

            log_q_tau = self.lprobs

            if l2_reg > 0:
                reg_loss = l2_reg * tf.reduce_sum(
                    [tf.reduce_sum(tf.square(var)) for var in discrim_vars])
            else:
                reg_loss = 0.0

            if gp_coeff:
                # gradient penalty applied only to real stuff (label=1)
                assert self.labels.shape.as_list()[1:] == [1]
                real_mask = tf.abs(self.labels[:, 0] - 1) < 1e-3
                # was using tf.boolean_mask(rew_input, real_mask), but not
                # using that anymore because slicing into input tensor makes it
                # impossible to use tf.gradients :(
                in_rew_all = rew_input
                out_rew_reals = tf.boolean_mask(self.energy, real_mask)
                # was using tf.boolean_mask(vfn_input, real_mask)
                in_vfn_all = vfn_input
                out_vfn_reals = tf.boolean_mask(fitted_value_fn, real_mask)
                # was using tf.boolean_mask(vfn_n_input, real_mask)
                in_vfn_n_all = vfn_n_input
                out_vfn_n_reals = tf.boolean_mask(fitted_value_fn_n, real_mask)
                # divide coefficient by 2 for comparability with past work
                self.gp_value = tf.reduce_sum([
                    batch_grad_penalty(out_rew_reals, in_rew_all),
                    batch_grad_penalty(out_vfn_reals, in_vfn_all),
                    batch_grad_penalty(out_vfn_n_reals, in_vfn_n_all),
                ])
                gp_loss = gp_coeff / 2.0 * self.gp_value
            else:
                self.gp_value = None
                gp_loss = tf.constant(0.0)

            if vairl:
                self.tot_kl_loss = sum(kl_losses)
            else:
                # wrap in tf.constant so that we can pass it to sess.run()
                self.tot_kl_loss = tf.constant(0.0, dtype=tf.float32)

            log_pq = tf.reduce_logsumexp([log_p_tau, log_q_tau], axis=0)

            # d_tau is exp(log(f/(f+pi))) = f/(f+pi); cent_loss uses logarithm
            # directly
            self.d_tau = tf.exp(log_p_tau - log_pq)
            self.cent_loss = -tf.reduce_mean(self.labels * (
                log_p_tau - log_pq) + (1 - self.labels) * (log_q_tau - log_pq))

            self.loss = self.cent_loss
            tot_loss = self.loss + reg_loss + gp_loss
            if vairl:
                tot_loss += self.vairl_beta * self.tot_kl_loss
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
            max_itrs=100,
            **kwargs):

        if self.frozen:
            return 0

        if self.fusion is not None:
            old_paths = self.fusion.sample_paths(n=len(paths))
            self.fusion.add_paths(paths)
            paths = paths + old_paths
            # log fusion stats
            fstats = self.fusion.compute_age_stats()
            logger.record_tabular('FusionAgeMean', fstats['mean'])
            logger.record_tabular('FusionAgeMed', fstats['med'])
            logger.record_tabular('FusionAgeStd', fstats['std'])
            logger.record_tabular('FusionAgeMax', fstats['max'])
            logger.record_tabular('FusionAgeMin', fstats['min'])
            logger.record_tabular('FusionAgePFresh', fstats['pfresh'])

        self._compute_path_probs(paths, insert=True)

        # self.eval_expert_probs(paths, policy, insert=True)
        for traj in self.expert_trajs:
            if 'agent_infos' in traj:
                # print('deleting agent_infos')
                del traj['agent_infos']
            if 'a_logprobs' in traj:
                del traj['a_logprobs']
        self.eval_expert_probs(self.expert_trajs, policy, insert=True)

        self._insert_next_state(paths)
        self._insert_next_state(self.expert_trajs)

        obs, obs_next, acts, acts_next, path_probs = self.extract_paths2(
            paths,
            keys=('observations', 'observations_next', 'actions',
                  'actions_next', 'a_logprobs'),
            last_timestep_only=last_timestep_only)
        (expert_obs, expert_obs_next, expert_acts, expert_acts_next,
         expert_probs) = self.extract_paths2(
             self.expert_trajs,
             keys=('observations', 'observations_next', 'actions',
                   'actions_next', 'a_logprobs'),
             last_timestep_only=last_timestep_only)

        # Train discriminator
        for it in TrainingIterator(max_itrs, heartbeat=5):
            nobs_batch, obs_batch, nact_batch, act_batch, lprobs_batch = \
                self.sample_batch(obs_next, obs, acts_next, acts, path_probs,
                                  batch_size=batch_size)

            (nexpert_obs_batch, expert_obs_batch, nexpert_act_batch,
             expert_act_batch, expert_lprobs_batch) = self.sample_batch(
                 expert_obs_next,
                 expert_obs,
                 expert_acts_next,
                 expert_acts,
                 expert_probs,
                 batch_size=batch_size)

            labels = np.zeros((batch_size * 2, 1))
            labels[batch_size:] = 1.0
            obs_batch = np.concatenate([obs_batch, expert_obs_batch], axis=0)
            nobs_batch = np.concatenate(
                [nobs_batch, nexpert_obs_batch], axis=0)
            act_batch = np.concatenate([act_batch, expert_act_batch], axis=0)
            nact_batch = np.concatenate(
                [nact_batch, nexpert_act_batch], axis=0)
            lprobs_batch = np.expand_dims(
                np.concatenate([lprobs_batch, expert_lprobs_batch], axis=0),
                axis=1).astype(np.float32)

            learn_step_feed_dict = {
                self.act_t: act_batch,
                self.obs_t: obs_batch,
                self.nobs_t: nobs_batch,
                self.nact_t: nact_batch,
                self.labels: labels,
                self.lprobs: lprobs_batch,
                self.lr: lr,
                # we only enable noise during training
                self.is_train_t: True,
            }
            sess = tf.get_default_session()
            loss, tot_kl, _ = sess.run(
                [self.loss, self.tot_kl_loss, self.step],
                feed_dict=learn_step_feed_dict)
            if self.vairl and self.vairl_adaptive_beta:
                beta, _ = sess.run(
                    [self.vairl_beta, self.vairl_beta_update_op],
                    feed_dict={self.vairl_mean_kl: tot_kl})

            it.record('loss', loss)
            it.record('tot_kl', tot_kl)
            if self.vairl and self.vairl_adaptive_beta:
                it.record('beta', beta)
            if it.heartbeat:
                print(it.itr_message())
                mean_loss = it.pop_mean('loss')
                print('\tLoss:%f' % mean_loss)
                mean_tot_kl = it.pop_mean('tot_kl')
                print('\tKL:%f' % mean_tot_kl)
                if self.vairl and self.vairl_adaptive_beta:
                    mean_beta = it.pop_mean('beta')
                    print('\tBeta:%f' % mean_beta)

        if logger:
            logger.record_tabular('GCLDiscrimLoss', mean_loss)
            # the 'DiscrimVAIRLKL' one is just retained so I don't break my
            # parsing scripts :)
            logger.record_tabular('GCLDiscrimVAIRLKL', mean_tot_kl)
            logger.record_tabular('GCLVAIRLKL', mean_tot_kl)
            if self.vairl and self.vairl_adaptive_beta:
                logger.record_tabular('GCLVAIRLBeta', mean_beta)
            # obs_next = np.r_[obs_next, np.expand_dims(obs_next[-1], axis=0)]
            # logZ,
            for is_train in [True, False]:
                # make sure to keep stats about test-mode configuration as well
                # as train-mode configuration, in case we have something like
                # dropout or VDB noise that affects discriminator results
                prefix = '' if is_train else 'NotIsTrain'
                fake_in_dict = {
                    'energy': self.energy,
                    'logZ': self.value_fn,
                    'dtau_fake': self.d_tau
                }
                real_in_dict = {
                    'energy': self.energy,
                    'logZ': self.value_fn,
                    'dtau_real': self.d_tau
                }
                if self.gp_value is not None:
                    fake_in_dict['gp_value'] = real_in_dict['gp_value'] = self.gp_value
                fake_out_dict = tf.get_default_session().run(
                    fake_in_dict,
                    feed_dict={
                        self.act_t: acts,
                        self.obs_t: obs,
                        self.nobs_t: obs_next,
                        self.nact_t: acts_next,
                        self.lprobs: np.expand_dims(path_probs, axis=1),
                        self.is_train_t: is_train,
                        self.labels: np.zeros((len(acts), 1)),
                    })
                energy = fake_out_dict['energy']
                logZ = fake_out_dict['logZ']
                dtau_fake = fake_out_dict['dtau_fake']
                logger.record_tabular(prefix + 'GCLLogZ', np.mean(logZ))
                logger.record_tabular(prefix + 'GCLAverageEnergy', np.mean(energy))
                logger.record_tabular(prefix + 'GCLAverageLogPtau', np.mean(-energy - logZ))
                logger.record_tabular(prefix + 'GCLAverageLogQtau', np.mean(path_probs))
                logger.record_tabular(prefix + 'GCLMedianLogQtau', np.median(path_probs))
                logger.record_tabular(prefix + 'GCLAverageDtau', np.mean(dtau_fake))

                # expert_obs_next = np.r_[expert_obs_next,
                # np.expand_dims(expert_obs_next[-1], axis=0)]
                real_out_dict = tf.get_default_session().run(
                    real_in_dict,
                    feed_dict={
                        self.act_t: expert_acts,
                        self.obs_t: expert_obs,
                        self.nobs_t: expert_obs_next,
                        self.nact_t: expert_acts_next,
                        self.lprobs: np.expand_dims(expert_probs, axis=1),
                        self.is_train_t: is_train,
                        self.labels: np.ones((len(expert_acts), 1)),
                    })
                energy = real_out_dict['energy']
                logZ = real_out_dict['logZ']
                dtau_real = real_out_dict['dtau_real']
                logger.record_tabular(prefix + 'GCLAverageExpertEnergy', np.mean(energy))
                logger.record_tabular(prefix + 'GCLAverageExpertLogPtau',
                                    np.mean(-energy - logZ))
                logger.record_tabular(prefix + 'GCLAverageExpertLogQtau',
                                    np.mean(expert_probs))
                logger.record_tabular(prefix + 'GCLMedianExpertLogQtau',
                                    np.median(expert_probs))
                logger.record_tabular(prefix + 'GCLAverageExpertDtau', np.mean(dtau_real))

                # 1 real, 0 fake
                disc_true_nfake = len(dtau_fake)
                disc_true_nreal = len(dtau_real)
                disc_true_pos = np.sum(dtau_real >= 0.5)
                disc_false_neg = disc_true_nreal - disc_true_pos
                assert disc_false_neg == np.sum(dtau_real < 0.5)
                disc_true_neg = np.sum(dtau_fake < 0.5)
                disc_false_pos = disc_true_nfake - disc_true_neg
                assert disc_false_pos == np.sum(dtau_fake >= 0.5)
                disc_total = disc_true_nfake + disc_true_nreal
                assert 0 <= disc_true_pos and 0 <= disc_false_neg \
                    and 0 <= disc_true_neg and 0 <= disc_false_pos
                assert disc_true_pos + disc_false_neg + disc_true_neg \
                    + disc_false_pos == disc_total
                # acc = (tp+tn)/(tp+fp+tn+fn)
                disc_acc = (disc_true_pos + disc_true_neg) / disc_total
                # precision = |relevant&retrieved|/|retrieved| = tp/(tp+fp)
                disc_prec = disc_true_pos / (disc_true_pos + disc_false_pos)
                # recall = |relevant&retrieved|/|relevant| = tp/(tp+fn)
                disc_recall = disc_true_pos / (disc_true_pos + disc_false_neg)
                # tpr = tp/(tp+fn) = recall
                disc_tpr = disc_true_pos / (disc_true_pos + disc_false_neg)
                assert disc_tpr == disc_recall
                # tnr = tn/(tn+fp) = recall
                disc_tnr = disc_true_neg / (disc_true_neg + disc_false_pos)
                assert 0 <= disc_prec <= 1 and 0 <= disc_prec <= 1 and \
                    0 <= disc_acc <= 1 and 0 <= disc_tpr <= 1 and \
                    0 <= disc_tnr <= 1
                disc_f1 \
                    = 2 * disc_prec * disc_recall / (disc_prec + disc_recall)
                assert 0 <= disc_f1 <= 1
                logger.record_tabular(prefix + 'GCLDiscAcc', disc_acc)
                logger.record_tabular(prefix + 'GCLDiscF1', disc_f1)
                # TPR is accuracy when predicting reals
                logger.record_tabular(prefix + 'GCLDiscTPR', disc_tpr)
                # TNR is accuracy when predicting fakes
                logger.record_tabular(prefix + 'GCLDiscTNR', disc_tnr)
                logger.record_tabular(prefix + 'GCLDiscNFake', disc_true_nfake)
                logger.record_tabular(prefix + 'GCLDiscNReal', disc_true_nreal)

                if self.gp_value is not None:
                    gp_value = 0.5 * (real_out_dict['gp_value']
                                      + fake_out_dict['gp_value'])
                    logger.record_tabular('GCLDiscGradPenaltyUnscaled', gp_value)

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

        elif self.score_shaped:
            self._insert_next_state(paths)
            obs, obs_next, acts = self.extract_paths(
                paths, keys=('observations', 'observations_next', 'actions', ))

            score = tf.get_default_session().run(
                self.shaped_r,
                feed_dict={
                    self.act_t: acts,
                    self.obs_t: obs,
                    self.nobs_t: obs_next
                })
            score = score[:, 0]

        else:
            obs, acts = self.extract_paths(paths)
            energy = tf.get_default_session().run(
                self.energy, feed_dict={self.act_t: acts,
                                        self.obs_t: obs})
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

    def debug_eval(self, obs, **kwargs):

        energy, v, = tf.get_default_session().run(
            [
                self.energy,
                self.value_fn,
            ], feed_dict={self.obs_t: obs})

        return {
            'reward': -energy,
            'value': v,
        }
        return {}

    @staticmethod
    def extract_paths2(*args, **kwargs):
        return AIRL.extract_paths(*args, **kwargs)
