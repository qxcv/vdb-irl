import os
import random
import re
from sklearn.externals import joblib
import json
import contextlib

import rllab.misc.logger as rllablogger
import tensorflow as tf
import numpy as np

from inverse_rl.utils.hyperparametrized import extract_hyperparams

import pickle

import os.path as osp


def load(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


@contextlib.contextmanager
def rllab_logdir(algo=None, dirname=None, mode=None):
    if dirname:
        rllablogger.set_snapshot_dir(dirname)
    if mode is not None:
        rllablogger.set_snapshot_mode(mode)
    dirname = rllablogger.get_snapshot_dir()
    rllablogger.add_tabular_output(os.path.join(dirname, 'progress.csv'))
    if algo:
        with open(os.path.join(dirname, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            json.dump(params, f)
    yield dirname
    rllablogger.remove_tabular_output(os.path.join(dirname, 'progress.csv'))


def save_itr_params_pickle(itr, params):
    from rllab.misc.logger import _snapshot_dir, _snapshot_mode

    if _snapshot_dir:
        if _snapshot_mode == 'all':
            file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
            #joblib.dump(params, file_name, compress=3)
            with open(file_name, 'wb') as f:
                pickle.dump(params, f)
        elif _snapshot_mode == 'last':
            # override previous params
            file_name = osp.join(_snapshot_dir, 'params.pkl')
            #joblib.dump(params, file_name, compress=3)
            with open(file_name, 'wb') as f:
                pickle.dump(params, f)
        elif _snapshot_mode == "gap":
            if itr % _snapshot_gap == 0:
                file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % itr)
                #joblib.dump(params, file_name, compress=3)
                with open(file_name, 'wb') as f:
                    pickle.dump(params, f)
        elif _snapshot_mode == 'none':
            pass
        else:
            raise NotImplementedError


def prune_old_snapshots(itr, keep_every=25, keep_latest=5):
    """Starting at iteration `itr`, prune all old snapshots so that only keep
    every `keep_every`th snapshot, plus the latest `keep_latest` snapshots."""
    from rllab.misc.logger import _snapshot_dir
    up_to_now = list(range(0, itr + 1))
    to_keep = set(up_to_now[-keep_latest:] + up_to_now[::keep_every])
    to_remove = sorted(set(up_to_now) - set(to_keep))
    for it in to_remove:
        # remove all the junk files
        file_name = osp.join(_snapshot_dir, 'itr_%d.pkl' % it)
        try:
            os.unlink(file_name)
        except FileNotFoundError:
            # doing this & swallowing exception for ALL files is probably
            # inefficient, but at least it's simple…
            pass


def get_expert_fnames(log_dir, n=5):
    print('Looking for paths')
    import re
    itr_reg = re.compile(r"itr_(?P<itr_count>[0-9]+)\.pkl")

    itr_files = []
    for i, filename in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('itr_count')
            itr_files.append((itr_count, filename))

    itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)[:n]
    for itr_file_and_count in itr_files:
        fname = os.path.join(log_dir, itr_file_and_count[1])
        print('Loading %s' % fname)
        yield fname


def load_experts(fname, max_files=float('inf'), min_return=None,
                 max_traj=None):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if hasattr(fname, '__iter__'):
        paths = []
        for fname_ in fname:
            tf.reset_default_graph()
            with tf.Session(config=config):

                snapshot_dict = joblib.load(fname_)
                # with open(fname_, 'rb') as f:
                # snapshot_dict = pickle.load(f)
                # snapshot_dict = load(fname)
                # import IPython; IPython.embed()
                # snapshot_dict = list(snapshot_dict)[0]

            # import IPython; IPython.embed()
            paths.extend(snapshot_dict['paths'])
    else:
        with tf.Session(config=config):
            snapshot_dict = joblib.load(fname)
        paths = snapshot_dict['paths']
    tf.reset_default_graph()

    trajs = []
    for path in paths:
        obses = path['observations']
        actions = path['actions']
        returns = path['returns']
        total_return = np.sum(returns)
        if (min_return is None) or (total_return >= min_return):
            traj = {'observations': obses, 'actions': actions}
            trajs.append(traj)
    random.shuffle(trajs)
    if max_traj is not None:
        trajs = trajs[:max_traj]
    print('Loaded %d trajectories' % len(trajs))
    return trajs


def load_latest_experts(logdir, n=5, min_return=None, max_traj=None):
    return load_experts(
        get_expert_fnames(logdir, n=n), min_return=min_return,
        max_traj=max_traj)


def load_latest_experts_multiple_runs(logdir, n=5):
    paths = []
    for i, dirname in enumerate(os.listdir(logdir)):
        dirname = os.path.join(logdir, dirname)
        if os.path.isdir(dirname):
            print('Loading experts from %s' % dirname)
            paths.extend(load_latest_experts(dirname, n=n))
    return paths


def load_latest_experts_walky(logdir, n=5, *, min_return=None, max_traj=None):
    """Scan for directories & subdirectories containing itr_*.pkl files, then
    load the latest `n` pickles from each.

    MMmm pickles"""
    itr_re = re.compile(r'^itr_(\d+)\.pkl$')
    logdir = os.path.abspath(logdir)
    all_ckpts = []
    for dir_path, _, filenames in os.walk(logdir):
        matches = map(itr_re.match, filenames)
        ckpts_fns = [(int(m.groups()[0]), fn)
                     for fn, m in zip(filenames, matches) if m is not None]
        latest_ckpts_fns = sorted(ckpts_fns, reverse=True)[:n]
        latest_fps = [os.path.join(dir_path, fn) for _, fn in latest_ckpts_fns]
        all_ckpts.extend(latest_fps)
    return load_experts(all_ckpts, min_return=min_return, max_traj=max_traj)


def load_policy(fname):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config):
        snapshot_dict = joblib.load(fname)
    pol_params = snapshot_dict['policy_params']
    tf.reset_default_graph()
    return pol_params
