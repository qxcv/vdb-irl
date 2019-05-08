#!/usr/bin/env python3
"""Simulate a given policy to evaluate it against another (different) reward
function. Written in response to this suggestion from Sergey (2018-09-25):

> perhaps what you could do is get a good policy for transfer case with GT
> reward, and evaluate whether it's better or worse w.r.t. your learned reward;
> if it's better, you know it's an exploration issue
"""
import os
import argparse
import joblib
import json
import tensorflow as tf
import tqdm
import numpy as np

from sandbox.rocky.tf.envs.base import TfEnv
from rllab.sampler.utils import rollout
from multiple_irl.envs.env_utils import CustomGymEnv

from multiple_irl.models.shaped_airl import AIRL
from inverse_rl.models.imitation_learning import GAIL
from inverse_rl.utils.log_utils import load_latest_experts_walky
from inverse_rl.utils.sane_hyperparams import min_layers_hidsize_for, \
    irltrpo_params_for
from inverse_rl.models.tf_util import get_session_config
from inverse_rl.envs.env_utils import get_inner_env


def get_name(irl_pkl):
    with tf.Session(config=get_session_config(), graph=tf.Graph()):
        irl_pkl_data = joblib.load(irl_pkl)
        env_name = get_inner_env(irl_pkl_data['env']).env_name
        del irl_pkl_data
    tf.reset_default_graph()
    return env_name


def main(
        rundir='data',
        irl_pkl='',
        pol_pkl=None,
        method=None,
        hid_size=None,
        hid_layers=None,
        switch_env=None, ):
    print('irl_pkl =', irl_pkl, 'and pol_pkl =', pol_pkl)
    orig_env_name = get_name(irl_pkl)
    if switch_env is not None:
        this_env_name = switch_env
    else:
        this_env_name = orig_env_name
    print("Running on environment '%s'" % this_env_name)
    env = TfEnv(
        CustomGymEnv(this_env_name, record_video=False, record_log=False))

    if hid_size is None or hid_layers is None:
        # we want hidden size & layer count for the *original* environment,
        # since that's what the IRL model that we're trying to reconstruct was
        # trained on
        assert hid_size is None and hid_layers is None, \
            "must specify both size & layers, not one or the other"
        hid_layers, hid_size = min_layers_hidsize_for(orig_env_name)
    # we want trajectory length for the new environment rather than the
    # original environment, though
    traj_length = irltrpo_params_for(this_env_name,
                                     'retrain')['max_path_length']
    print('Horizon is', traj_length)

    expert_dir = os.path.join(rundir, 'env_%s/' % orig_env_name.lower())
    experts = load_latest_experts_walky(expert_dir, n=1)

    with tf.Session(config=get_session_config(), graph=tf.Graph()):
        irl_pkl_data = joblib.load(irl_pkl)

        disc_net_kwargs = {
            'layers': hid_layers,
            'd_hidden': hid_size,
        }
        if method in {'airl', 'vairl'}:
            irl_model = AIRL(
                env=env,
                expert_trajs=experts,
                state_only=True,
                freeze=True,
                vairl=method == 'vairl',
                vairl_beta=1e-4,
                discrim_arch_args=disc_net_kwargs,
                fitted_value_fn_arch_args=disc_net_kwargs)
        elif method in {'gail', 'vail'}:
            irl_model = GAIL(
                env,
                expert_trajs=experts,
                discrim_arch_args=disc_net_kwargs,
                name=method,
                freeze=True,
                vail=method == 'vail')
        else:
            raise NotImplementedError("Don't know how to handle method '%s'" %
                                      method)
        irl_model.set_params(irl_pkl_data['irl_params'])

        if pol_pkl is not None:
            with tf.variable_scope('please-work'):
                pol_pkl_data = joblib.load(pol_pkl)
                policy = pol_pkl_data['policy']
                print('Using policy loaded from %s' % pol_pkl)
        else:
            print('Using original IRL policy')
            policy = irl_pkl_data['policy']

        # do a few rollouts with given policy on given reward
        # report both the IRL reward AND the mean reward for the policy
        n_rollouts = 30
        irl_rets = np.zeros((n_rollouts, ))
        env_rets = np.zeros((n_rollouts, ))
        for i in tqdm.trange(n_rollouts):
            # how do I get final return? Hmm
            path = rollout(env, policy, max_path_length=traj_length)
            env_rets[i] = np.sum(path['rewards'])
            irl_rew = irl_model.eval([path])
            irl_rets[i] = np.sum(irl_rew)

        print('Env mean %.2f (std %.2f)' %
              (np.mean(env_rets), np.std(env_rets)))
        print('IRL mean %.2f (std %.2f)' %
              (np.mean(irl_rets), np.std(irl_rets)))


def infer_method(pkl_path):
    pkl_dir = os.path.dirname(os.path.abspath(pkl_path))
    param_path = os.path.join(pkl_dir, 'params.json')
    with open(param_path, 'r') as fp:
        json_data = json.load(fp)
    irl_cls = json_data['irl_model']['__clsname__']
    if irl_cls == 'AIRL':
        if json_data['irl_model']['vairl']:
            return 'vairl'
        return 'airl'
    elif irl_cls == 'GAIL':
        if json_data['irl_model']['vail']:
            return 'vail'
        return 'gail'
    raise NotImplementedError(
        "I don't know how to deal with IRL class %s for %s :(" %
        (irl_cls, pkl_path))


parser = argparse.ArgumentParser()
parser.add_argument(
    'irl_pkl_path',
    metavar='irl-pkl-path',
    help='saved reward function parameter pickle (environment is taken from '
    'this pickle)')
parser.add_argument(
    'pol_pkl_path',
    metavar='pol-pkl-path',
    nargs='?',
    help='saved policy parameter pickle (will use policy from irl-pkl-path '
    'if this is not given)')
parser.add_argument(
    '--switch-env',
    default=None,
    help='use this environment, not the one from irl-pkl-path')
parser.add_argument(
    '--rundir',
    default='data/',
    help='directory to store run data in (will be created if necessary)')
parser.add_argument(
    '--hid-layers',
    default=None,
    type=int,
    help='number of hidden layers in discriminator (or policy)')
parser.add_argument(
    '--hid-size',
    default=None,
    type=int,
    help='size of each hidden layer in discriminator (or policy)')
parser.add_argument(
    '--method',
    choices=('airl', 'vairl', 'gail', 'vail'),
    default=None,
    help='IRL method to  use')

if __name__ == "__main__":
    args = parser.parse_args()
    assert isinstance(args.irl_pkl_path, str)
    assert isinstance(args.pol_pkl_path, (type(None), str))
    if args.method is None:
        args.method = infer_method(args.irl_pkl_path)
        print('Assuming method %s' % args.method)
    main(**{
        'irl_pkl': args.irl_pkl_path,
        'pol_pkl': args.pol_pkl_path,
        'method': args.method,
        'rundir': args.rundir,
        'hid_size': args.hid_size,
        'hid_layers': args.hid_layers,
        'switch_env': args.switch_env,
    })
