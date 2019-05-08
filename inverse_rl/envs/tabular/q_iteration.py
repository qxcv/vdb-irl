"""
Use q-iteration to solve for an optimal policy

Usage: q_iteration(env, gamma=discount factor, ent_wt= entropy bonus)
"""
import numpy as np
from scipy.misc import logsumexp as sp_lse
from rllab.misc import logger


def softmax(q, alpha=1.0):
    q = (1.0 / alpha) * q
    q = q - np.max(q)
    probs = np.exp(q)
    probs = probs / np.sum(probs)
    return probs


def logsumexp(q, alpha=1.0, axis=1):
    return alpha * sp_lse((1.0 / alpha) * q, axis=axis)


def get_policy(q_fn, ent_wt=1.0):
    v_rew = logsumexp(q_fn, alpha=ent_wt)
    adv_rew = q_fn - np.expand_dims(v_rew, axis=1)
    pol_probs = np.exp((1.0 / ent_wt) * adv_rew)
    assert np.all(np.isclose(np.sum(pol_probs, axis=1), 1.0)), str(pol_probs)
    return pol_probs


def q_iteration(env,
                reward_matrix=None,
                K=50,
                gamma=0.99,
                ent_wt=0.1,
                warmstart_q=None,
                policy=None):
    """
    Perform tabular soft Q-iteration
    """
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    if reward_matrix is None:
        reward_matrix = env.rew_matrix
    if warmstart_q is None:
        q_fn = np.zeros((dim_obs, dim_act))
    else:
        q_fn = warmstart_q

    t_matrix = env.transition_matrix
    for k in range(K):
        if policy is None:
            v_fn = logsumexp(q_fn, alpha=ent_wt)
        else:
            v_fn = np.sum(
                (q_fn - np.log(policy)) * policy, axis=1)  # this is wrong!!
        new_q = reward_matrix + gamma * t_matrix.dot(v_fn)
        q_fn = new_q
    return q_fn


def q_nonsoft_iteration(env,
                        reward_matrix=None,
                        K=50,
                        gamma=0.99,
                        ent_wt=0.1,
                        warmstart_q=None,
                        policy=None):
    """
    Perform tabular soft Q-iteration
    """
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    if reward_matrix is None:
        reward_matrix = env.rew_matrix
    if warmstart_q is None:
        q_fn = np.zeros((dim_obs, dim_act))
    else:
        q_fn = warmstart_q

    t_matrix = env.transition_matrix
    for k in range(K):
        if policy is None:
            v_fn = np.max(q_fn, axis=1)
        else:
            v_fn = np.sum(q_fn * policy, axis=1)  # this is wrong!!
        new_q = reward_matrix + gamma * t_matrix.dot(v_fn)
        q_fn = new_q
    return q_fn


def q_iteration_learning_curve(env,
                               reward_matrix=None,
                               K=50,
                               gamma=0.99,
                               ent_wt=0.1,
                               warmstart_q=None,
                               policy=None,
                               eval_reward_matrix=None):
    """
    Perform tabular soft Q-iteration
    """
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    if reward_matrix is None:
        reward_matrix = env.rew_matrix
    if warmstart_q is None:
        q_fn = np.zeros((dim_obs, dim_act))
    else:
        q_fn = warmstart_q
    learning_curve = []

    t_matrix = env.transition_matrix
    for k in range(K):
        returns = compute_returns(
            env,
            get_policy(q_fn),
            reward_matrix=eval_reward_matrix,
            ent_wt=ent_wt,
            gamma=gamma)
        learning_curve.append(returns)

        if policy is None:
            v_fn = logsumexp(q_fn, alpha=ent_wt)
        else:
            v_fn = np.sum(
                (q_fn - np.log(policy)) * policy, axis=1)  # this is wrong!!
        new_q = reward_matrix + gamma * t_matrix.dot(v_fn)
        q_fn = new_q

    return q_fn, learning_curve


def compute_returns(env, policy, reward_matrix=None, ent_wt=0.1, gamma=0.99):
    """
    Perform tabular soft Q-iteration
    """
    dim_obs = env.observation_space.flat_dim
    dim_act = env.action_space.flat_dim
    if reward_matrix is None:
        reward_matrix = env.rew_matrix
    start_state = env.init_state
    #t_matrix = env.transition_matrix
    q_pi = q_nonsoft_iteration(
        env,
        reward_matrix=reward_matrix,
        policy=policy,
        gamma=gamma,
        ent_wt=ent_wt,
        K=150)
    #v_pi = np.sum((q_pi - np.log(policy))*policy, axis=1)  # this is wrong!!
    v_pi = np.sum(q_pi * policy, axis=1)  # this is wrong!!
    #print('vpi', v_pi)
    ret = v_pi[start_state]
    return ret
