import tensorflow as tf
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc.overrides import overrides
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.spaces import Box

KL_MODE_MC = 'mc'  # Monte-Carlo KL estimation
KL_MODE_VAR = 'var'  # Variational KL estimation
KL_MODE_GAUSS = 'gauss'  # Gaussian KL estimation


def sample_gmm(N, mu, log_std, cluster_wts):
    num_clusters = cluster_wts.shape[1]
    sel_log_std = []
    sel_mu = []
    for n in range(N):
        cluster = np.random.choice(np.arange(num_clusters), p=cluster_wts[n])
        sel_mu.append(mu[n, cluster])
        sel_log_std.append(log_std[n, cluster])
    sel_mu = np.array(sel_mu)
    sel_log_std = np.array(sel_log_std)

    rnd = np.random.normal(size=sel_mu.shape)
    actions = rnd * np.exp(sel_log_std) + sel_mu
    return actions


def dbg_print(input, data, message=None, summarize=10):
    #return tf.Print(input, data, message=message, summarize=summarize)
    return input


def gaussian_kl(mu, log_std, mu_, log_std_):
    old_std = tf.exp(log_std_)
    new_std = tf.exp(log_std)
    numerator = tf.square(mu_ - mu) + \
                tf.square(old_std) - tf.square(new_std)
    denominator = 2 * tf.square(new_std) + 1e-8
    numerator = dbg_print(numerator, [numerator], message='single_numer')
    denominator = dbg_print(denominator, [denominator], message='single_denom')

    quot = numerator / denominator
    quot = dbg_print(quot, [quot], message='single_quot')
    std_dist = log_std - log_std_
    std_dist = dbg_print(std_dist, [std_dist], message='single_std_dist')

    return tf.reduce_sum(quot + std_dist, axis=-1)


def batch_pairwise_diff(x, y):
    # x shape = [?, x, d]
    # y shape = [?, y, d]
    # output = [?, x, y, d]
    nx = int(x.get_shape()[1])
    ny = int(y.get_shape()[1])
    dim = int(x.get_shape()[2])
    x = tf.expand_dims(x, axis=2)
    y = tf.expand_dims(y, axis=1)
    dists = x - y
    assert dists.get_shape().is_compatible_with([None, nx, ny, dim])
    return dists


def gaussian_pairwise_kl(mu, log_std, mu_, log_std_):
    n_cluster_x = int(mu.get_shape()[1])
    n_cluster_y = int(mu_.get_shape()[1])

    new_std = tf.exp(log_std)
    old_std = tf.exp(log_std_)

    pairwise_mu_dists = tf.square(batch_pairwise_diff(mu, mu_))
    pairwise_mu_dists = dbg_print(
        pairwise_mu_dists, [pairwise_mu_dists],
        message='pair_mu_dist',
        summarize=20)

    pairwise_sqstd_dists = -batch_pairwise_diff(
        tf.square(new_std), tf.square(old_std))
    pairwise_std_dists = batch_pairwise_diff(
        log_std, log_std_
    )  # some bug here!!! needs to be positive or negative depending on shape

    numerator = pairwise_mu_dists + pairwise_sqstd_dists
    denominator = 2 * tf.square(new_std) + 1e-8
    denominator = tf.expand_dims(denominator, axis=2)

    numerator = dbg_print(
        numerator, [numerator], message='pair_numer', summarize=10)
    denominator = dbg_print(
        denominator, [denominator], message='pair_denom', summarize=10)

    quot = numerator / denominator
    quot = dbg_print(quot, [quot], message='pair_quot', summarize=10)

    pairwise_std_dists = dbg_print(
        pairwise_std_dists, [pairwise_std_dists],
        message='pair_std_dist',
        summarize=10)

    kl = tf.reduce_sum(quot + pairwise_std_dists, axis=-1)
    assert kl.get_shape().is_compatible_with([None, n_cluster_x, n_cluster_y])

    #tf.assert_greater_equal(kl, 0)
    return tf.nn.relu(kl)


class GMMDistribution(object):
    # Diagonal Gaussian GMM
    def __init__(self, dim=2, clusters=2, kl_mode=KL_MODE_GAUSS,
                 kl_samples=64):
        super(GMMDistribution, self).__init__()
        self.dim = dim
        self.clusters = clusters
        self.kl_samples = kl_samples
        self.kl_mode = kl_mode

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two distributions
        """
        # draw samples from new_dist_info_vars
        if False:  #self.clusters == 1:
            old_means = old_dist_info_vars["gmm_mu"][:, 0]
            old_log_stds = old_dist_info_vars["gmm_log_std"][:, 0]
            new_means = new_dist_info_vars["gmm_mu"][:, 0]
            new_log_stds = new_dist_info_vars["gmm_log_std"][:, 0]
            return gaussian_kl(new_means, new_log_stds, old_means,
                               old_log_stds)
        else:
            if self.kl_mode == KL_MODE_MC:
                gmm_mu, gmm_log_std, gmm_cluster_wts = new_dist_info_vars['gmm_mu'], new_dist_info_vars['gmm_log_std'], \
                                                       new_dist_info_vars['gmm_cluster_wts']
                batch_size = tf.shape(gmm_mu)[0]
                cluster_samples = tf.contrib.distributions.Categorical(
                    probs=gmm_cluster_wts)
                cluster_samples = tf.transpose(cluster_samples.sample())
                assert cluster_samples.get_shape().is_compatible_with([None])
                gauss_samples = tf.random_normal(
                    shape=(batch_size, self.dim), mean=0.0, stddev=1.0)

                elems = tf.range(batch_size)
                gmm_mu_sel = tf.map_fn(lambda idx: gmm_mu[tf.cast(idx, tf.int32),
                                                        cluster_samples[tf.cast(idx, tf.int32)],:],
                                            tf.cast(elems, tf.float32), parallel_iterations=1000)
                gmm_log_std_sel = tf.map_fn(lambda idx: gmm_log_std[tf.cast(idx, tf.int32),
                                                        cluster_samples[tf.cast(idx, tf.int32)],:],
                                            tf.cast(elems, tf.float32), parallel_iterations=1000)

                assert gmm_mu_sel.get_shape().is_compatible_with(
                    [None, self.dim])
                assert gmm_log_std_sel.get_shape().is_compatible_with(
                    [None, self.dim])
                dist_samples = gauss_samples * tf.exp(
                    gmm_log_std_sel) + gmm_mu_sel
                assert dist_samples.get_shape().is_compatible_with(
                    [None, self.dim])
                # compute
                new_prob = self.log_likelihood_sym(
                    dist_samples, new_dist_info_vars,
                    extra_dim=True)  # shape should be (?, kl_samples)
                old_prob = self.log_likelihood_sym(
                    dist_samples, old_dist_info_vars, extra_dim=True)
                kl_sym = new_prob - old_prob
            elif self.kl_mode == KL_MODE_VAR:
                gmm_mu, gmm_log_std, gmm_cluster_wts = new_dist_info_vars['gmm_mu'], new_dist_info_vars['gmm_log_std'], \
                                                       new_dist_info_vars['gmm_cluster_wts']
                ogmm_mu, ogmm_log_std, ogmm_cluster_wts = old_dist_info_vars['gmm_mu'], old_dist_info_vars['gmm_log_std'], \
                                                       old_dist_info_vars['gmm_cluster_wts']

                # Upper bound the KL (see Approximating the KL Divergence Between GMMs, Hershey & Olsen)
                fg_kl = gaussian_pairwise_kl(gmm_mu, gmm_log_std, ogmm_mu,
                                             ogmm_log_std)
                wa = tf.expand_dims(gmm_cluster_wts, axis=1)
                wb = tf.expand_dims(ogmm_cluster_wts, axis=2)
                wts = wa * wb * fg_kl
                kl_sym = tf.reduce_sum(wts, axis=[1, 2])
                kl_sym = tf.Print(
                    kl_sym, [kl_sym], message='kl_sym', summarize=32)
                kl_sym = tf.Print(
                    kl_sym, [tf.reduce_mean(kl_sym)], message='kl_sym_mean')

                old_means = old_dist_info_vars["gmm_mu"][:, 0]
                old_log_stds = old_dist_info_vars["gmm_log_std"][:, 0]
                new_means = new_dist_info_vars["gmm_mu"][:, 0]
                new_log_stds = new_dist_info_vars["gmm_log_std"][:, 0]
                kl_single = gaussian_kl(new_means, new_log_stds, old_means,
                                        old_log_stds)
                kl_single = tf.Print(
                    kl_single, [kl_single], message='kl_single', summarize=32)
                kl_single = tf.Print(
                    kl_single, [tf.reduce_mean(kl_single)],
                    message='kl_single_mean')

                diff = tf.reduce_sum(tf.abs(kl_sym - kl_single))
                #assert_op = tf.Assert( diff<=1e-4, [diff])
                #with tf.control_dependencies([assert_op]):
                #    kl_sym = 0.9999*kl_sym + 0.0001*kl_single
                kl_sym = tf.Print(kl_sym, [diff], message='kl_diff')
                #import pdb; pdb.set_trace()
            elif self.kl_mode == KL_MODE_GAUSS:
                gmm_mu, gmm_log_std, gmm_cluster_wts = new_dist_info_vars['gmm_mu'], new_dist_info_vars['gmm_log_std'], \
                                                       new_dist_info_vars['gmm_cluster_wts']
                ogmm_mu, ogmm_log_std, ogmm_cluster_wts = old_dist_info_vars['gmm_mu'], old_dist_info_vars['gmm_log_std'], \
                                                          old_dist_info_vars['gmm_cluster_wts']
                gmm_cluster_wts = tf.expand_dims(gmm_cluster_wts, axis=2)
                ogmm_cluster_wts = tf.expand_dims(ogmm_cluster_wts, axis=2)

                #mu_a = gmm_mu[:,0]
                #mu_b = ogmm_mu[:,0]
                #mu_a = tf.reduce_sum(gmm_mu, axis=1)
                #mu_b = tf.reduce_sum(ogmm_mu, axis=1)
                mu_a = tf.reduce_sum(gmm_cluster_wts * gmm_mu, axis=1)
                mu_b = tf.reduce_sum(ogmm_cluster_wts * ogmm_mu, axis=1)

                #mumu_a = tf.expand_dims(mu_a, axis=1) * tf.expand_dims(mu_a, axis=2)
                #mumu_a = tf.expand_dims(mumu_a, axis=1)
                #mumu_b = tf.expand_dims(mu_b, axis=1) * tf.expand_dims(mu_b, axis=2)
                #mumu_b = tf.expand_dims(mumu_b, axis=1)
                std_a = gmm_log_std[:, 0]
                std_b = ogmm_log_std[:, 0]
                #std_a = tf.log(tf.reduce_sum(gmm_cluster_wts * tf.exp(gmm_log_std), axis=1))
                #std_b = tf.log(tf.reduce_sum(ogmm_cluster_wts * tf.exp(ogmm_log_std), axis=1))
                kl_sym = gaussian_kl(mu_a, std_a, mu_b, std_b)
            else:
                raise NotImplementedError()
        #kl_sym = tf.Print(kl_sym, [kl_sym], summarize=20, message='kl_sym')
        assert kl_sym.get_shape().is_compatible_with([None])
        return kl_sym

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError()

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars,
                             new_dist_info_vars):
        old_loglik = self.log_likelihood_sym(x_var, old_dist_info_vars)
        new_loglik = self.log_likelihood_sym(x_var, new_dist_info_vars)
        return tf.exp(new_loglik - old_loglik)

    def entropy(self, dist_info):
        # do a crappy entropy approximation

        # average individual entropies... not sure how good this is
        log_stds, cluster_wts = dist_info['gmm_log_std'], dist_info[
            'gmm_cluster_wts']
        entropies = np.sum(
            log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)
        entropy_approx = np.sum(entropies * cluster_wts, axis=1)
        return entropy_approx

    def entropy_sym(self, dist_info_vars):
        # do a crappy entropy approximation
        # average individual entropies... not sure how good this is
        log_stds, cluster_wts = dist_info_vars['gmm_log_std'], dist_info_vars[
            'gmm_cluster_wts']
        entropies = tf.reduce_sum(
            log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)
        entropy_approx = tf.reduce_sum(entropies * cluster_wts, axis=1)
        return entropy_approx

    def log_likelihood_sym(self, x_var, dist_info_vars, extra_dim=False):
        mu, log_std, cluster_wts = dist_info_vars['gmm_mu'], dist_info_vars[
            'gmm_log_std'], dist_info_vars['gmm_cluster_wts']
        x_expand = tf.expand_dims(x_var, axis=1)

        diff = x_expand - mu
        diff = tf.reshape(diff, [-1, self.clusters, self.dim])
        assert diff.get_shape().is_compatible_with(
            [None, self.clusters, self.dim])
        log_var = 2 * log_std
        assert log_var.get_shape().is_compatible_with(
            [None, self.clusters, self.dim])

        ll = (-0.5 * self.dim * np.log(np.pi*2)) + \
             (-0.5*tf.reduce_sum(log_var, axis=2)) + \
             (-0.5*tf.reduce_sum(tf.square(diff)/tf.exp(log_var), axis=2))
        assert ll.get_shape().is_compatible_with([None, self.clusters])
        ll = tf.reduce_sum(ll * cluster_wts, axis=1)

        return ll

    def log_likelihood(self, xs, dist_info):
        raise NotImplementedError

    @property
    def dist_info_specs(self):
        return [("gmm_mu", (self.clusters, self.dim)),
                ("gmm_log_std", (self.clusters, self.dim)),
                ("gmm_cluster_wts", (self.clusters, ))]

    @property
    def dist_info_keys(self):
        return ["gmm_mu", "gmm_log_std", "gmm_cluster_wts"]


def test_ff_relu_arch(obs,
                      clusters=2,
                      dout=1,
                      name='ff_relu',
                      reuse=False,
                      cluster_hack=True):
    from inverse_rl.models.tf_util import linear
    with tf.variable_scope(name, reuse=reuse):
        features = linear(obs, dout=32, name='features')
        features = tf.nn.relu(features)

        cluster_wts = tf.nn.softmax(
            linear(features, dout=clusters, name='cluster_wts'))
        mu_flat = linear(features, dout=clusters * dout, name='mu_flat') * 1e-1
        log_std_flat = linear(
            features, dout=clusters * dout, name='log_std_flat') * 1e-1
    mu = tf.reshape(mu_flat, [-1, clusters, dout])
    log_std = tf.reshape(log_std_flat, [-1, clusters, dout])
    return mu, log_std, cluster_wts


class GMMPolicy(StochasticPolicy, Serializable):
    def __init__(self,
                 name,
                 env_spec,
                 num_clusters=2,
                 network_arch=test_ff_relu_arch):
        Serializable.quick_init(self, locals())
        self.obs_dim = obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = action_dim = env_spec.action_space.flat_dim
        #self.graph = tf.get_default_graph()
        self.name = name
        self.num_clusters = num_clusters
        assert isinstance(env_spec.action_space, Box)
        super(GMMPolicy, self).__init__(env_spec)

        self.arch = network_arch
        self.dist = GMMDistribution(dim=action_dim, clusters=num_clusters)

        self.sample_obs = tf.placeholder(tf.float32, [None, obs_dim])
        self.sample_mu, self.sample_log_std, self.sample_cluster_wts = \
            network_arch(self.sample_obs, dout=self.action_dim, name=name, clusters=self.num_clusters)

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars):
        mu, log_std, cluster_wts = self.arch(
            obs_var,
            dout=self.action_dim,
            clusters=self.num_clusters,
            name=self.name,
            reuse=True)
        return {
            'gmm_mu': mu,
            'gmm_log_std': log_std,
            'gmm_cluster_wts': cluster_wts
        }

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
        assert len(np.array(observations).shape) == 2
        mu, log_std, cluster_wts = \
            tf.get_default_session().run([self.sample_mu, self.sample_log_std, self.sample_cluster_wts],
                                        feed_dict={self.sample_obs: np.array(observations)})
        """
        sel_log_std = []
        sel_mu = []
        for n in range(observations.shape[0]):
            cluster = np.random.choice(np.arange(self.num_clusters), p=cluster_wts[n])
            sel_mu.append(mu[cluster])
            sel_log_std.append(log_std[cluster])
        sel_mu = np.array(sel_mu)
        sel_log_std = np.array(sel_log_std)

        rnd = np.random.normal(size=mu.shape)
        actions = rnd*np.exp(sel_log_std) + sel_mu
        """
        actions = sample_gmm(len(observations), mu, log_std, cluster_wts)
        assert actions.shape == (len(observations), self.action_dim)

        return actions, dict(
            gmm_mu=mu, gmm_log_std=log_std, gmm_cluster_wts=cluster_wts)

    def log_diagnostics(self, paths):
        cluster_wts = np.concatenate(
            [traj['agent_infos']['gmm_cluster_wts'] for traj in paths])
        gmm_mu = np.concatenate(
            [traj['agent_infos']['gmm_mu'] for traj in paths])
        gmm_std = np.concatenate(
            [traj['agent_infos']['gmm_log_std'] for traj in paths])

        logger.record_tabular('AverageClusterWts', np.mean(
            cluster_wts, axis=0))
        logger.record_tabular(
            'AverageClusterEnt', -np.mean(
                np.sum(cluster_wts * np.log(cluster_wts + 1e-6), axis=1),
                axis=0))
        for cl in range(self.num_clusters):
            logger.record_tabular('AverageMu_Cluster%.2d' % cl,
                                  np.mean(gmm_mu[:, cl], axis=0))
            logger.record_tabular('AverageLogStd_Cluster%.2d' % cl,
                                  np.mean(gmm_std[:, cl], axis=0))

    @property
    @overrides
    def recurrent(self):
        return False

    @property
    def distribution(self):
        return self.dist

    @property
    def state_info_specs(self):
        return []

    def get_params_internal(self, **tags):
        #layers = L.get_all_layers(self._output_layers, treat_as_input=self._input_layers)
        #params = itertools.chain.from_iterable(l.get_params(**tags) for l in layers)
        #return L.unique(params)
        key = tf.GraphKeys.GLOBAL_VARIABLES
        if tags.get('trainable', False):
            key = tf.GraphKeys.TRAINABLE_VARIABLES
        return tf.get_collection(key, scope=self.name)


if __name__ == "__main__":
    # test kl
    dim = 2

    mu = tf.placeholder(tf.float32, [None, 2, dim])
    log_std = tf.placeholder(tf.float32, [None, 2, dim])
    mu2 = tf.placeholder(tf.float32, [None, 3, dim])
    log_std2 = tf.placeholder(tf.float32, [None, 3, dim])

    pairwise = gaussian_pairwise_kl(mu, log_std, mu2, log_std2)

    single_kl = gaussian_kl(mu[:, 0], log_std[:, 0], mu2[:, 0], log_std2[:, 0])
    single_kl2 = gaussian_kl(mu[:, 1], log_std[:, 1], mu2[:, 0],
                             log_std2[:, 0])
    #single_kl2 = gaussian_kl(mu[:,1], log_std[:,1], mu[:,0], log_std[:,0])

    my_mu = np.array([[[0, 0], [0, 1]]])
    my_mu2 = np.array([[[1, 1], [2, 1], [1, 1]]])
    my_std = np.array([[[0, 0], [0, 1]]])
    my_std2 = np.array([[[-1, 0], [0, -2], [-1, -1]]])
    with tf.Session().as_default():
        kls = tf.get_default_session().run(
            pairwise,
            feed_dict={
                mu: my_mu,
                log_std: my_std,
                mu2: my_mu2,
                log_std2: my_std2
            })
        print(kls)
        kls = tf.get_default_session().run([single_kl, single_kl2],
                                           feed_dict={
                                               mu: my_mu,
                                               log_std: my_std,
                                               mu2: my_mu2,
                                               log_std2: my_std2
                                           })
        print(kls)
