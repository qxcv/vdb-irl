#!/usr/bin/env python3
"""Use TRPO to create an expert demonstration on a gym environment."""
import argparse
import os
import warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from inverse_rl.algos.trpo import TRPO  # noqa: E402
from inverse_rl.utils.log_utils import rllab_logdir  # noqa: E402
from inverse_rl.utils.hyper_sweep import run_sweep_serial, \
    run_sweep_parallel  # noqa: E402
from inverse_rl.utils.sane_hyperparams import irltrpo_params_for, \
    min_layers_hidsize_polstd_for  # noqa: E402
from multiple_irl.envs.env_utils import CustomGymEnv  # noqa: E402

from rllab.baselines.linear_feature_baseline \
    import LinearFeatureBaseline  # noqa: E402
from sandbox.rocky.tf.envs.base import TfEnv  # noqa: E402
from sandbox.rocky.tf.policies.gaussian_mlp_policy \
    import GaussianMLPPolicy  # noqa: E402
from rllab.sampler import parallel_sampler  # noqa: E402


def main(exp_name,
         ent_wt=1.0,
         trpo_step=0.01,
         env_name='Shaped_PM_MazeRoom_Small-v0',
         rundir='data',
         init_pol_std=1.0,
         many_runs=False,
         hid_size=None,
         hid_layers=None):
    os.makedirs(rundir, exist_ok=True)

    env = TfEnv(CustomGymEnv(env_name, record_video=False, record_log=False))
    if hid_size is None or hid_layers is None:
        assert hid_size is None and hid_layers is None, \
            "must specify both size & layers, not one or the other"
        hid_layers, hid_size, init_pol_std \
            = min_layers_hidsize_polstd_for(env_name)
        hidden_sizes = (hid_size, ) * hid_layers
    env_trpo_params = irltrpo_params_for(env_name, 'expert')
    policy = GaussianMLPPolicy(
        name='policy',
        env_spec=env.spec,
        hidden_sizes=hidden_sizes,
        init_std=init_pol_std)
    algo = TRPO(
        env=env,
        policy=policy,
        entropy_weight=ent_wt,
        step_size=trpo_step,
        discount=0.99,
        store_paths=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        force_batch_sampler=True,
        # extra environment-specific params like batch size, max path length,
        # etc.
        **env_trpo_params, )

    if many_runs:
        logdir = os.path.join(rundir, 'env_%s' % (env_name.lower()), exp_name)
    else:
        logdir = os.path.join(rundir, 'env_%s' % (env_name.lower()))
    with rllab_logdir(algo=algo, dirname=logdir):
        algo.train()


parser = argparse.ArgumentParser()
parser.add_argument(
    'env_name',
    metavar='env-name',
    nargs='?',
    help='environment to use (default: maze env)',
    default='Shaped_PM_MazeRoom_Small-v0')
parser.add_argument(
    # XXX maybe use 0.05 for some mujoco envs? IDK
    '--trpo-step',
    default=0.01,
    type=float,
    help='step size for TRPO policy')
parser.add_argument(
    # XXX not clear what ent should be, but was doing experiments with zero
    # entropy before
    '--trpo-ent',
    default=0.1,
    type=float,
    help='entropy weight for TRPO')
parser.add_argument(
    '--hid-layers',
    default=None,
    type=int,
    help='number of hidden layers in policy (default: auto)')
parser.add_argument(
    '--hid-size',
    default=None,
    type=int,
    help='size of each hidden layer in policy (default: auto)')
parser.add_argument(
    '--repeat',
    default=1,
    type=int,
    help='number of times to repeat data collection (default: 1)')
parser.add_argument(
    '--rundir',
    default='data/',
    help='directory to store run data in (will be created if necessary)')


def _init(args):
    env_name = args.env_name
    print('Using environment %s' % env_name)
    params_dict = {
        'env_name': [env_name],
        'rundir': [args.rundir],
        'ent_wt': [args.trpo_ent],
        'trpo_step': [args.trpo_step],
        'hid_size': [args.hid_size],
        'hid_layers': [args.hid_layers],
        'many_runs': [args.repeat > 1]
    }
    if args.repeat > 1:
        # stacked parallel thing doesn't work, bleh
        warnings.warn(
            "You're trying to use --repeat N for N > 1, but that "
            "disables parallel sampling. This is probably going to be "
            "heinously slow or something, use at own risk.")
        # parallel_sampler.initialize(n_parallel=1)
        # parallel_sampler.set_seed(1)
        run_sweep_parallel(main, params_dict, repeat=args.repeat)
    else:
        parallel_sampler.initialize(n_parallel=8)
        parallel_sampler.set_seed(1)
        run_sweep_serial(main, params_dict, repeat=1)


if __name__ == "__main__":
    _init(parser.parse_args())
