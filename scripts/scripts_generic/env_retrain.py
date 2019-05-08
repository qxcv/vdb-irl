#!/usr/bin/env python3
"""Train a new policy using a previously-learnt reward function."""
import os
import argparse
import joblib
import json
import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from multiple_irl.envs.env_utils import CustomGymEnv

from inverse_rl.algos.irl_trpo import IRLTRPO
from multiple_irl.models.shaped_airl import AIRL
from inverse_rl.models.imitation_learning import GAIL
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts_walky
from inverse_rl.utils.hyper_sweep import run_sweep_serial
from inverse_rl.utils.sane_hyperparams import irltrpo_params_for, \
    min_layers_hidsize_polstd_for
from inverse_rl.models.tf_util import load_prior_params, get_session_config
from inverse_rl.envs.env_utils import get_inner_env


def get_name(irl_pkl):
    with tf.Session(config=get_session_config()):
        irl_pkl_data = joblib.load(irl_pkl)
        env_name = get_inner_env(irl_pkl_data['env']).env_name
        del irl_pkl_data
    tf.reset_default_graph()
    return env_name


def main(
        exp_name,
        rundir='data',
        irl_pkl='',
        ent_wt=1.0,
        trpo_anneal_steps=None,
        trpo_anneal_init_ent=None,
        trpo_step=0.01,
        init_pol_std=1.0,
        method=None,
        hid_size=None,
        hid_layers=None,
        switch_env=None, ):
    orig_env_name = get_name(irl_pkl)
    if switch_env is not None:
        this_env_name = switch_env
    else:
        this_env_name = orig_env_name
    print("Running on environment '%s'" % this_env_name)
    env = TfEnv(
        CustomGymEnv(this_env_name, record_video=False, record_log=False))

    if hid_size is None or hid_layers is None:
        assert hid_size is None and hid_layers is None, \
            "must specify both size & layers, not one or the other"
        hid_layers, hid_size, init_pol_std \
            = min_layers_hidsize_polstd_for(orig_env_name)
    env_trpo_params = irltrpo_params_for(orig_env_name, 'retrain')

    folder = os.path.dirname(irl_pkl)

    prior_params = load_prior_params(irl_pkl)
    expert_dir = os.path.join(rundir, 'env_%s/' % orig_env_name.lower())
    experts = load_latest_experts_walky(expert_dir, n=5)

    # For some reason IRLTRPO is responsible for setting weights in this code.
    # It would equally be possible to run global_variables_initializer()
    # ourselves and then do irl_model.set_params(prior_params) if we just
    # wanted to query energy, reward, etc. from the trained AIRL model without
    # using IRLTRPO.
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

    pol_hid_sizes = (hid_size, ) * hid_layers
    policy = GaussianMLPPolicy(
        name='policy',
        env_spec=env.spec,
        hidden_sizes=pol_hid_sizes,
        init_std=init_pol_std)
    irltrpo_kwargs = dict(
        env=env,
        policy=policy,
        irl_model=irl_model,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=ent_wt,  # should be 1.0 but 0.1 seems to work better
        step_size=trpo_step,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        init_irl_params=prior_params,
        force_batch_sampler=True,
        entropy_anneal_init_weight=trpo_anneal_init_ent,
        entropy_anneal_steps=trpo_anneal_steps,
        retraining=True)
    irltrpo_kwargs.update(env_trpo_params)
    algo = IRLTRPO(**irltrpo_kwargs)
    folder_suffix = ''
    if switch_env is not None:
        # append lower case environment name to retrain folder path
        folder_suffix = '_%s' % switch_env.lower()
    with rllab_logdir(
            algo=algo, dirname='%s/retrain%s' % (folder, folder_suffix)):
        with tf.Session():
            algo.train()


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
    'pkl_path',
    metavar='pkl-path',
    help='saved AIRL parameter pickle', )
parser.add_argument(
    '--switch-env',
    default=None,
    help='use this environment, not the unpickled one')
parser.add_argument(
    '--trpo-step', default=0.01, type=float, help='step size for TRPO policy')
parser.add_argument(
    '--trpo-ent', default=0.1, type=float, help='entropy weight for TRPO')
parser.add_argument(
    '--trpo-anneal-init-ent',
    type=float,
    default=None,
    help='initial entropy weight to use for TRPO; this will be annealed '
    'linearly until it reaches --trpo-ent (if --trpo-anneal-init-ent is not '
    'given then annealing will not be used)')
parser.add_argument(
    '--trpo-anneal-steps',
    type=int,
    default=None,
    help='number of iterations over which to anneal TRPO entropy bonus')
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
    if args.method is None:
        args.method = infer_method(args.pkl_path)
        print('Assuming method %s' % args.method)
    assert (args.trpo_anneal_init_ent is None) \
        == (args.trpo_anneal_steps is None), \
        "must supply both of --trpo-anneal-{init-ent,steps} or neither"
    params_dict = {
        'irl_pkl': [args.pkl_path],
        'ent_wt': [args.trpo_ent],
        'trpo_step': [args.trpo_step],
        'trpo_anneal_steps': [args.trpo_anneal_steps],
        'trpo_anneal_init_ent': [args.trpo_anneal_init_ent],
        'method': [args.method],
        'rundir': [args.rundir],
        'hid_size': [args.hid_size],
        'hid_layers': [args.hid_layers],
        'switch_env': [args.switch_env],
    }
    # run_sweep_parallel(main, params_dict, repeat=3)
    run_sweep_serial(main, params_dict, repeat=1)
