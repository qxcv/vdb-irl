import itertools

from inverse_rl.envs.tabular import q_iteration
from inverse_rl.models.policies.tabular_softq_policy import CategoricalSoftQPolicy
from inverse_rl.utils import log_utils
from inverse_rl.utils.math_utils import np_seed
from matplotlib import cm

import numpy as np
import matplotlib.pyplot as plt
from inverse_rl.envs.utils import flat_to_one_hot
from matplotlib.patches import Rectangle
from rllab.envs.base import Env
from rllab.misc import logger
from rllab.spaces import Box
from rllab.spaces import Discrete


class DiscreteEnv(Env):
    def __init__(self,
                 transition_matrix,
                 reward,
                 init_state,
                 terminate_on_reward=False):
        super(DiscreteEnv, self).__init__()
        dX, dA, dXX = transition_matrix.shape
        self.nstates = dX
        self.nactions = dA
        self.transitions = transition_matrix
        self.init_state = init_state
        self.reward = reward
        self.terminate_on_reward = terminate_on_reward

        self.__observation_space = Box(0, 1, shape=(self.nstates, ))
        #max_A = 0
        #for trans in self.transitions:
        #    max_A = max(max_A, len(self.transitions[trans]))
        self.__action_space = Discrete(dA)

    def reset(self):
        self.cur_state = self.init_state
        obs = flat_to_one_hot(self.cur_state, ndim=self.nstates)
        return obs

    def step(self, a):
        transition_probs = self.transitions[self.cur_state, a]
        next_state = np.random.choice(
            np.arange(self.nstates), p=transition_probs)
        r = self.reward[self.cur_state, a, next_state]
        self.cur_state = next_state
        obs = flat_to_one_hot(self.cur_state, ndim=self.nstates)

        done = False
        if self.terminate_on_reward and r > 0:
            done = True
        return obs, r, done, {}

    def tabular_trans_distr(self, s, a):
        return self.transitions[s, a]

    def reward_fn(self, s, a):
        return self.reward[s, a]

    def log_diagnostics(self, paths):
        #Ntraj = len(paths)
        #acts = np.array([traj['actions'] for traj in paths])
        obs = np.array(
            [np.sum(traj['observations'], axis=0) for traj in paths])

        state_count = np.sum(obs, axis=0)
        #state_count = np.mean(state_count, axis=0)
        state_freq = state_count / float(np.sum(state_count))
        for state in range(self.nstates):
            logger.record_tabular('AvgStateFreq%d' % state, state_freq[state])

    def get_optimal_policy(self, ent_wt=0.1, gamma=0.9):
        q_fn = q_iteration.q_iteration(self, ent_wt=ent_wt, gamma=gamma, K=100)
        policy = CategoricalSoftQPolicy(
            env_spec=self.spec, ent_wt=ent_wt, hardcoded_q=q_fn)
        policy.set_q_func(q_fn)

        #q_fn2 = 1.0/ent_wt * q_fn
        #probs = np.exp(q_fn2)/np.sum(np.exp(q_fn2), axis=1, keepdims=True)
        #import pdb; pdb.set_trace()

        return policy

    def plot_data(self, data, dirname=None, itr=0, fname='trajs_itr%d'):
        plt.figure()
        ax = plt.gca()
        normalized_values = data
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values / np.max(normalized_values)
        norm_data = normalized_values

        cmap = cm.RdYlBu

        if len(data.shape) == 1:
            ydim = 0
            for x in range(data.shape[0]):
                ax.text(x, 0, '%.1f' % data[x], size='x-small')
                color = cmap(norm_data[x])
                ax.add_patch(Rectangle([x, 0], 1, 1, color=color))
        elif len(data.shape) == 2:
            ydim = data.shape[1]
            for x, y in itertools.product(
                    range(data.shape[0]), range(data.shape[1])):
                iy = data.shape[1] - y - 1
                ax.text(x, iy, '%.1f' % data[x, y], size='x-small')
                color = cmap(norm_data[x, y])
                ax.add_patch(Rectangle([x, iy], 1, 1, color=color))

        ax.set_xticks(np.arange(-1, data.shape[0] + 1, 1))
        ax.set_yticks(np.arange(-1, ydim + 1, 1))
        plt.grid()

        if dirname is not None:
            log_utils.record_fig(fname % itr, subdir=dirname, rllabdir=True)
        else:
            plt.show()

    def plot_trajs(self, paths, dirname=None, itr=0):
        obs = np.array(
            [np.sum(traj['observations'], axis=0) for traj in paths])
        #state_count = np.sum(obs, axis=1)
        state_count = np.sum(obs, axis=0)
        state_freq = state_count / float(np.sum(state_count))
        self.plot_data(state_freq, dirname=dirname, itr=itr)

    def plot_costs(self,
                   paths,
                   cost_fn,
                   dirname=None,
                   itr=0,
                   policy=None,
                   use_traj_paths=False):
        if not use_traj_paths:
            # iterate through states, and each action - makes sense for non-rnn costs
            obses = []
            acts = []
            for (x, a) in itertools.product(
                    range(self.nstates), range(self.nactions)):
                obs = flat_to_one_hot(x, ndim=self.nstates)
                act = flat_to_one_hot(a, ndim=self.nactions)
                obses.append(obs)
                acts.append(act)
            path = {'observations': np.array(obses), 'actions': np.array(acts)}
            if policy is not None:
                if hasattr(policy, 'set_env_infos'):
                    policy.set_env_infos(path.get('env_infos', {}))
                actions, agent_infos = policy.get_actions(path['observations'])
                path['agent_infos'] = agent_infos
            paths = [path]

        plots = cost_fn.debug_eval(paths, policy=policy)
        for plot in plots:
            plots[plot] = plots[plot].squeeze()

        for plot in plots:
            data = plots[plot]
            data = np.reshape(data, (self.nstates, self.nactions))
            self.plot_data(
                data, dirname=dirname, fname=plot + '_itr%d', itr=itr)

    @property
    def transition_matrix(self):
        return self.transitions

    @property
    def rew_matrix(self):
        return self.reward

    @property
    def initial_state_distribution(self):
        return flat_to_one_hot(self.init_state, ndim=self.nstates)

    @property
    def action_space(self):
        return self.__action_space

    @property
    def observation_space(self):
        return self.__observation_space


def random_env(Nstates, Nact, seed=None, terminate=False, t_sparsity=0.75):
    assert Nstates >= 2
    if seed is None:
        seed = 0
    reward_state = 0
    start_state = 1
    with np_seed(seed):
        transition_matrix = np.random.rand(Nstates, Nact, Nstates)
        transition_matrix = np.exp(transition_matrix)
        for s in range(Nstates):
            for a in range(Nact):
                zero_idxs = np.random.randint(
                    0, Nstates, size=int(Nstates * t_sparsity))
                transition_matrix[s, a, zero_idxs] = 0.0

        transition_matrix = transition_matrix / np.sum(
            transition_matrix, axis=2, keepdims=True)
        reward = np.zeros((Nstates, Nact))
        reward[reward_state, :] = 1.0
        #reward = np.random.randn(Nstates,1 ) + reward

        stable_action = seed % Nact  #np.random.randint(0, Nact)
        transition_matrix[reward_state, stable_action] = np.zeros(Nstates)
        transition_matrix[reward_state, stable_action, reward_state] = 1
    return DiscreteEnv(
        transition_matrix,
        reward=reward,
        init_state=start_state,
        terminate_on_reward=terminate)


if __name__ == '__main__':
    env = random_env(5, 2, seed=0)
    print(env.transitions)
    print(env.transitions[0, 0])
    print(env.transitions[0, 1])
    env.reset()
    for _ in range(100):
        print(env.step(env.action_space.sample()))
