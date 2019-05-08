#!/usr/bin/env python3
"""Run AIRL or VAIRL on a given environment, then re-optimise the learnt reward
to make sure it still works."""
import argparse
import os
import subprocess
import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from inverse_rl.utils.sane_hyperparams import irltrpo_params_for, \
    min_layers_hidsize_polstd_for
from multiple_irl.envs.env_utils import CustomGymEnv

from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import GAIL

import multiple_irl.models.shaped_airl as shaped_airl
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts_walky
from inverse_rl.utils.hyper_sweep import run_sweep_parallel


def main(exp_name,
         rundir,
         ent_wt=1.0,
         env_name='Shaped_PM_MazeRoom_Small-v0',
         method="airl",
         beta=1e-2,
         disc_step=1e-3,
         disc_iters=100,
         disc_batch_size=32,
         disc_gp=None,
         trpo_step=1e-2,
         init_pol_std=1.0,
         adaptive_beta=False,
         target_kl=0.5,
         beta_step=1e-6,
         hid_size=None,
         hid_layers=None,
         max_traj=None):
    os.makedirs(rundir, exist_ok=True)

    if hid_size is None or hid_layers is None:
        assert hid_size is None and hid_layers is None, \
            "must specify both size & layers, not one or the other"
        hid_layers, hid_size, init_pol_std \
            = min_layers_hidsize_polstd_for(env_name)
    env_trpo_params = irltrpo_params_for(env_name, 'irl')

    env = TfEnv(CustomGymEnv(env_name, record_video=False, record_log=False))

    expert_dir = os.path.join(rundir, 'env_%s/' % env_name.lower())
    experts = load_latest_experts_walky(expert_dir, n=5, max_traj=max_traj)

    disc_net_kwargs = {
        'layers': hid_layers,
        'd_hidden': hid_size,
    }
    if method in {'airl', 'vairl'}:
        is_vairl = method == 'vairl'
        irl_model = shaped_airl.AIRL(
            env=env,
            expert_trajs=experts,
            state_only=True,
            fusion=True,
            discrim_arch_args=disc_net_kwargs,
            fitted_value_fn_arch_args=disc_net_kwargs,
            gp_coeff=disc_gp,
            # vairl flag
            vairl=is_vairl,
            # vairl fixed beta settings
            vairl_beta=beta,
            # vairl adaptive beta settings
            vairl_adaptive_beta=adaptive_beta,
            vairl_beta_step_size=beta_step,
            vairl_kl_target=target_kl)
    elif method in {'gail', 'vail'}:
        is_vail = method == 'vail'
        assert gp_coeff is None, "no GAIL/VAIL support for GP coeff"
        irl_model = GAIL(
            env,
            expert_trajs=experts,
            discrim_arch_args=disc_net_kwargs,
            name=method,
            # vail stuff (only adaptive beta for VAIL, no fixed beta like
            # VAIRL)
            vail=is_vail,
            # initial beta
            vail_init_beta=beta,
            vail_beta_step_size=beta_step,
            vail_kl_target=target_kl)
    else:
        raise NotImplementedError("don't know how to handle method '%s'" %
                                  (method, ))

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
        discrim_train_itrs=disc_iters,
        irl_model_wt=1.0,
        irl_lr=disc_step,
        irl_batch_size=disc_batch_size,
        step_size=trpo_step,
        # entropy_weight should be 1.0 but 0.1 seems to work better
        entropy_weight=ent_wt,
        force_batch_sampler=True,
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec), )
    irltrpo_kwargs.update(env_trpo_params)
    n_itr = irltrpo_kwargs['n_itr']
    print('irltrpo_kwargs:', irltrpo_kwargs)
    algo = IRLTRPO(**irltrpo_kwargs)

    run_name = 'env_{env_name}_{method}'.format(
        env_name=env_name.lower(), method=method)
    exp_folder = os.path.join(rundir, '%s/%s' % (run_name, exp_name))
    with rllab_logdir(algo=algo, dirname=exp_folder):
        with tf.Session():
            algo.train()
    this_dir = os.path.dirname(__file__)
    maze_retrain_path = os.path.join(this_dir, 'env_retrain.py')
    latest_irl_snap = '%s/itr_%d.pkl' % (exp_folder, n_itr - 1)
    subproc_cmd = [
        # script
        'python',
        maze_retrain_path,
        # experiment info
        latest_irl_snap,
        '--rundir',
        rundir,
        # TRPO params
        '--trpo-step',
        '%f' % trpo_step,
        '--trpo-ent',
        '%f' % ent_wt,
        # network params
        '--hid-layers',
        '%d' % hid_layers,
        '--hid-size',
        '%d' % hid_size,
        # we don't care about precise args relevant to given method because
        # we're just reloading a frozen model
        '--method',
        method,
    ]
    subprocess.run(subproc_cmd, check=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--repeat',
    type=int,
    default=5,
    help='number of times to repeat experiment')
parser.add_argument(
    '--env-name',
    default='Shaped_PM_MazeRoom_Small-v0',
    help='environment to use (default: maze env)')
parser.add_argument(
    '--disc-step',
    default=5e-5,
    type=float,
    help='step size for discriminator')
parser.add_argument(
    '--disc-batch-size',
    default=32,
    type=int,
    help='batch size for discriminator (Adam)')
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
    '--max-traj',
    default=None,
    type=int,
    help='maximum number of demo trajectories to load')
# Sam: I don't think I can use this in conjunction with run_sweep_parallel
# because of restrictions on nested multiprocessing (IIRC it deadlocked or
# something). Still, it might be worth changing the MP model if I need to get
# these experiments done _REALLY_ fast!
# parser.add_argument(
#     '--sampler-procs',
#     type=int,
#     default=8,
#     help='num procs to use for experience sampler')
parser.add_argument(
    '--trpo-step', default=0.01, type=float, help='step size for TRPO policy')
parser.add_argument(
    '--trpo-ent', default=0.1, type=float, help='entropy weight for TRPO')
parser.add_argument(
    '--disc-iters',
    type=int,
    default=100,
    help='# batches of discriminator training in each iteration '
    '(lower to balance training)')
parser.add_argument(
    '--dreg-gp',
    type=float,
    default=None,
    help='coefficient for zero-centred discriminator gradient penalty '
    '(default: 0)')
parser.add_argument(
    '--rundir',
    default='data/',
    help='directory to store run data in (will be created if necessary)')

subparsers = parser.add_subparsers(dest="method", title="IRL methods")

airl_parser = subparsers.add_parser("airl", help="use plain AIRL")
# AIRL has no extra options, beyond those listed above

vail_parser = subparsers.add_parser(
    "vail", help="use VIB version of GAIL (VAIL)")
vail_parser.add_argument(
    '--beta',
    default=0.005,
    type=float,
    help='initial coefficient for KL loss in discriminator')
vail_parser.add_argument(
    '--adaptive-beta-target-kl',
    type=float,
    default=0.5,
    help='target KL for beta')
vail_parser.add_argument(
    '--adaptive-beta-step',
    default=1e-5,
    type=float,
    help='step size for beta', )

gail_parser = subparsers.add_parser("gail", help="use GAIL")

vairl_parser = subparsers.add_parser(
    "vairl", help="use Variational AIRL (VAIRL)")
vairl_parser.add_argument(
    '--beta',
    default=0.005,
    type=float,
    help='coefficient for KL loss in discriminator')
vairl_parser.add_argument(
    '--adaptive-beta',
    action='store_true',
    default=False,
    help='enable adaptive beta')
vairl_parser.add_argument(
    '--adaptive-beta-target-kl',
    type=float,
    default=0.5,
    help='target KL for --adaptive-beta')
vairl_parser.add_argument(
    '--adaptive-beta-step',
    default=1e-5,
    type=float,
    help='step size for --adaptive-beta', )

if __name__ == "__main__":
    args = parser.parse_args()
    env_name = args.env_name
    print('Args:', args)
    params_dict = {
        'method': [args.method],
        'rundir': [args.rundir],
        'env_name': [env_name],
        'disc_step': [args.disc_step],
        'trpo_step': [args.trpo_step],
        'ent_wt': [args.trpo_ent],
        'hid_size': [args.hid_size],
        'hid_layers': [args.hid_layers],
        'disc_iters': [args.disc_iters],
        'disc_batch_size': [args.disc_batch_size],
        'disc_gp': [args.dreg_gp],
        'max_traj': [args.max_traj],
        # VAIR/VAIL params
        'beta': [getattr(args, 'beta', None)],
        'adaptive_beta': [getattr(args, 'adaptive_beta', None)],
        'target_kl': [getattr(args, 'adaptive_beta_target_kl', None)],
        'beta_step': [getattr(args, 'adaptive_beta_step', None)],
    }
    run_sweep_parallel(main, params_dict, repeat=args.repeat)
    # run_sweep_serial(main, params_dict, repeat=1)
