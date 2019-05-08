#!/usr/bin/env python3
"""Run hyperparameter optimiser on VAIRL code.

Things that I want to optimise over:

- Beta (important! Maybe try an adaptive beta & optimise the relevant
  hyperparam there?).
- Discriminator step size.
- TRPO step size.
- Entropy penalty of *AIRL* policy. At the moment it's at like 0.1, but perhaps
  should be at 1.0 (?).
- Entropy penalty of demonstrator policy (is there a better way to optimise
  this? Can we just push it up as far as it will go while still consistently
  reaching the goal in 200 epochs?).
- Whether to use M trajectories from latest N demonstrator snapshots, or M*N
  trajectories from just the most recent demonstrator snapshot.
- Possibly # of AIRL training epochs (although I think this is probably high
  enough already; also not sure how to incorporate a penalty in the objective
  which would push this down).

Each optimisation run will look something like this:

Unsolved problems:

- How do I stop from running out of disk space? Maybe delete all but the last
  itr_*.pkl snapshot? Actually yeah, that's a pretty good idea."""

import argparse
import contextlib
import datetime
import glob
import os
import re
import subprocess
import sys
import time

import hyperopt as hpe
import hyperopt.hp as hp
import hyperopt.tpe as tpe
import hyperopt.mongoexp as mongoexp
import numpy as np
import pandas as pd

# basically keep going forever
MAX_EVALS = 100000
# where this script and all the other scripts are
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERT_SCRIPT = os.path.join(THIS_DIR, 'env_data_collect.py')
COMPARE_SCRIPT = os.path.join(THIS_DIR, 'env_compare.py')


@contextlib.contextmanager
def mongo_workers(mongo_url, n_workers=None):
    """Start running some hyperopt-mongo-workers that will execute (in
    parallel) jobs that the master process selects."""
    if n_workers is None:
        # we divide core count because each of our trials will be running a few
        # things in parallel
        n_workers = max(1, int(os.cpu_count() / 5.0))

    strip_pfx = 'mongo://'
    assert mongo_url.startswith(strip_pfx)
    mongo_url = mongo_url[len(strip_pfx):]
    strip_sfx = '/jobs'
    assert mongo_url.endswith(strip_sfx)
    mongo_url = mongo_url[:-len(strip_sfx)]

    # spool up workers
    cmdline = [
        'hyperopt-mongo-worker', '--mongo=%s' % mongo_url,
        '--poll-interval=0.1', '--workdir', 'hyperopt-scratch'
    ]
    # try augmenting PYTHONPATH so that workers can find this module
    new_pythonpath = THIS_DIR
    if new_pythonpath not in os.environ.get('PYTHONPATH', ''):
        if os.environ.get('PYTHONPATH', None) is not None:
            new_pythonpath += ':' + os.environ['PYTHONPATH']
        os.environ['PYTHONPATH'] = new_pythonpath

    procs = []
    try:
        for _ in range(n_workers):
            proc = subprocess.Popen(cmdline)
            procs.append(proc)
        # check that all procs started right
        time.sleep(0.5)
        for proc in procs:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    "hyperopt worker died w/ code %d! cmdline: %s\nstdout:\n"
                    "%s\n\nstderr:\n%s" %
                    (code, cmdline, proc.stdout, proc.stderr))

        # let the user go about their business
        # TODO: try spooling up a thread to monitor the processes periodically;
        # if they die then we should restart them etc (overkill for now)
        yield

    # shut down workers at the end
    finally:
        exc = None
        for proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception as ex:
                exc = ex
        if exc is not None:
            raise exc


def clean_pickles(pickle_dir, to_keep=5):
    """Remove all but the most recent `to_keep` files in given dir. Assumes all
    pickles are named `itr_<num>.pkl` and sorts by num to get most recent."""
    name_re = re.compile(r'^itr_(\d+)\.pkl$')
    nums_and_pickles = []
    for fn in os.listdir(pickle_dir):
        m = name_re.match(fn)
        if m is None:
            continue
        fpath = os.path.join(pickle_dir, fn)
        num = int(m.groups()[0])
        nums_and_pickles.append((num, fpath))
    nums_and_pickles = sorted(nums_and_pickles)
    # this should catch bugs arising from pointing clean_pickles to the wrong
    # directory
    assert len(nums_and_pickles) >= to_keep, \
        "should be at least %d pickles in '%s' but found %d" \
        % (to_keep, pickle_dir, len(nums_and_pickles))
    assert to_keep >= 0
    if to_keep == 0:
        to_remove = nums_and_pickles
    else:
        to_remove = nums_and_pickles[:-to_keep]
    for _, pickle_path in to_remove:
        os.unlink(pickle_path)


def run_trial(kwargs):
    # get a unique run dir
    str_kwargs = ['%s=%s' % (u, v) for u, v in sorted(kwargs.items())]
    time_str = datetime.datetime.now().isoformat()
    run_id = '_'.join(str_kwargs) + '_' + time_str
    run_dir = os.path.join('data', run_id)
    os.makedirs(run_dir, exist_ok=True)

    # these args are taken by all progs
    common_args = [
        '--rundir', run_dir, '--trpo-step', '%f' % kwargs['trpo_step']
    ]
    subprocess.run(
        [
            # script & env
            'python3',
            EXPERT_SCRIPT,
            kwargs['env_name'],
            # entropy bonus for policy
            '--trpo-ent',
            '%f' % kwargs['trpo_ent_expert'],
            # everything else
            *common_args
        ],
        check=True)
    subprocess.run(
        [
            # script & env
            'python3',
            COMPARE_SCRIPT,
            kwargs['env_name'],
            # KL penalty
            '--beta',
            '%f' % kwargs['beta'],
            # discriminator learning rate
            '--disc-step',
            '%f' % kwargs['disc_step'],
            # entropy bonus for adversarially-learnt policy
            '--trpo-ent',
            '%f' % kwargs['trpo_ent_vairl'],
            # always do VAIRL
            '--vairl',
            # everything else
            *common_args
        ],
        check=True)

    # now clean up most pickles so that we save a bit of space but can still
    # debug later
    expert_pickle_dir = os.path.join(run_dir,
                                     'env_%s' % kwargs['env_name'].lower())
    clean_pickles(expert_pickle_dir, to_keep=5)
    pat = os.path.join(run_dir, 'env_*vairl*/*_*/')
    vairl_run_dirs = glob.glob(pat)
    assert len(vairl_run_dirs) > 0, "no dirs for pattern '%s'" % pat
    all_rets = []
    for vairl_run_dir in vairl_run_dirs:
        vairl_retrain_dir = os.path.join(vairl_run_dir, 'retrain')
        clean_pickles(vairl_retrain_dir, to_keep=1)
        clean_pickles(vairl_run_dir, to_keep=1)

        # also read out stats
        retrain_csv = os.path.join(vairl_retrain_dir, 'progress.csv')
        retrain_data = pd.read_csv(retrain_csv)
        ret_col = retrain_data['OriginalTaskAverageReturn']
        last_ret = ret_col.tolist()[-1]
        all_rets.append(last_ret)

    # could subtract std to trade off between mean performance and consistency,
    # but not going to bother for now
    # std_coeff = 1.0
    final_ret_std = np.std(all_rets)
    final_ret_mean = np.mean(all_rets)
    max_objective_val = final_ret_mean  # - std_coeff * final_ret_std

    return {
        # everything is always alright
        'status': 'ok',
        # negate the thing we want to maximise
        'loss': -max_objective_val,
        'loss_variance': final_ret_std**2,
        # return side information
        'ret_info': {
            'ret_mean': final_ret_mean,
            'ret_std': final_ret_std,
            'ret_all': all_rets,
        },
    }


def main(args):
    # we need to re-impot this module so we can give it to hyperopt mongo
    # workers
    sys.path.append(THIS_DIR)
    import hype  # noqa: E402

    opt_space = {
        # coefficient on discriminator KL loss term
        'beta':
        hp.loguniform('beta', np.log(1e-4), np.log(1e-1)),
        # step size for discriminator
        'disc_step':
        hp.loguniform('disc_step', np.log(1e-5), np.log(1e-4)),
        # step size for TRPO (for all uses of TRPO, including expert demos and
        # VAIRL inner loop)
        'trpo_step':
        hp.loguniform('trpo_step', np.log(1e-3), np.log(1e-1)),
        # entropy weight for TRPO in expert
        'trpo_ent_expert':
        hp.loguniform('trpo_ent_expert', np.log(1e-2), np.log(10)),
        # entropy weight for TRPO in VAIRL
        'trpo_ent_vairl':
        hp.loguniform('trpo_ent_vairl', np.log(1e-3), np.log(1)),
        # this doesn't change; just needs to be passed in
        'env_name':
        args.env_name
    }
    mongo_url = 'mongo://localhost:27017/vairl-hype/jobs'
    print('Creating MongoTrials')
    exp_key = 'run-pid-%d-date-%s' % (os.getpid(), datetime.datetime.now().isoformat())
    trials = mongoexp.MongoTrials(mongo_url, exp_key=exp_key)
    with open('to_run_mongo.txt', 'a') as fp:
        # so that we can read later
        fp.write('mongoexp.MongoTrials(%r, exp_key=%r)\n' % (mongo_url, exp_key))
    print('Starting Mongo workers')
    with mongo_workers(mongo_url):
        print('Running fmin() (this should use heaps of CPU!)')
        hpe.fmin(
            hype.run_trial,
            opt_space,
            trials=trials,
            algo=tpe.suggest,
            max_evals=MAX_EVALS)


parser = argparse.ArgumentParser()
parser.add_argument(
    'env_name',
    metavar='env-name',
    nargs='?',
    help='environment to use (default: double maze env)',
    default='Shaped_PM_MazeRoom_Small-v0')

if __name__ == '__main__':
    main(parser.parse_args())
