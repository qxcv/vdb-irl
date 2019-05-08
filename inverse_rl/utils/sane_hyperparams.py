"""Return some sane TRPO parameters for solving a given environment. Meant to
help me avoid mistakes arising from running TRPO with crap parameters."""

import re

MAZE_PARAMS = {
    'n_itr': 200,
    'discrim_train_itrs': 50,
    'max_path_length': 100,
    'batch_size': 10000,
}
S_MAZE_EXTRA_PARAMS_TRAIN = {
    # need bigger batches, more training, bigger steps, etc.
    'batch_size': 50000,
    'n_itr': 200,
    'step_size': 0.01
}
S_MAZE_EXTRA_PARAMS_TEST = {
    # no idea whether these are actually necessary
    'batch_size': 50000,
    'n_itr': 400,
    'step_size': 0.001
}
MUJOCO_COMMON_PARAMS = {
    # lots of papers claim awesome results in 250 epochs, but I haven't
    # observed that with the RLLab impl + my chosen hyperparams; hopefully 1500
    # iters will fix that
    'n_itr': 1500,
    # not sure what the most appropriate value is for ALL things, but 1000
    # seems safe
    'max_path_length': 1000,
    # this is what Justin used for CustomAnt; I assume it's a good default for
    # other tasks
    'discrim_train_itrs': 10,
}
MUJOCO_SPECIFIC_PARAMS = {
    # check
    # https://github.com/joschu/modular_rl/blob/master/experiments/battery-trpo.yaml
    # for some possibly-good params
    'InvertedPendulum': {
        'batch_size': 5000
    },
    'Reacher': {
        'batch_size': 15000
    },
    'InvertedDoublePendulum': {
        'batch_size': 15000
    },
    'HalfCheetah': {
        'batch_size': 25000,
    },
    'Hopper': {
        'batch_size': 25000
    },
    'Swimmer': {
        'batch_size': 25000
    },
    'Walker2d': {
        'batch_size': 25000
    },
    'CustomAnt': {
        # For compat with Justin's experiments, I'm using the most "lenient"
        # parameters he has. (he uses 0.1 entropy weight) (also no, I have no
        # idea why his impl takes so many epochs to solve Ant; that's like 30M
        # interactions!)
        'n_itr': 1500,
        'batch_size': 20000,
        'max_path_length': 500,
    },
    'DisabledAnt': {
        'n_itr': 1500,
        'batch_size': 20000,
        'max_path_length': 500,
    },
    'AntGoto': {
        # this is surprisingly easy to optimise for w/ batch size of 50k; even
        # 500 iterations would be enough to get pretty smart behaviour
        'n_itr': 1000,
        'batch_size': 50000,
        'max_path_length': 1000,
    },
    'DisabledAntGoto': {
        'n_itr': 1000,
        'batch_size': 50000,
        'max_path_length': 1000,
    },
    'Ant': {
        'batch_size': 50000
    },
    # humanoid also needs two-layer net with 64,64 hidden sizes!
    'Humanoid': {
        'batch_size': 50000
    },
}
REQUIRED_KEYS = {'n_itr', 'batch_size', "max_path_length"}


def _is_maze_room(env_name):
    # Justin rooms
    justin_match = re.match(r'(TwoD|Point)Maze(Left|Right)(-v\d+)?', env_name)
    if justin_match is not None:
        return True
    # dibya rooms
    match = re.match(r'^(Shaped_)?PM_(Empty|Wall|Maze|Two)Room(_Flip)?_'
                     '(Small|Med|Large)(-v\d+)?$', env_name)
    return match is not None


def _remove_version_roboschool(env_name):
    m = re.match(r'(?:Roboschool)?(.+)-v\d+', env_name)
    assert m is not None, "no version suffix in '%s'" % env_name
    prefix, = m.groups()
    return prefix


def irltrpo_params_for(env_name, phase):
    assert phase in {'expert', 'irl', 'retrain'}
    env_prefix = _remove_version_roboschool(env_name)
    if env_prefix in MUJOCO_SPECIFIC_PARAMS:
        rv = dict(MUJOCO_COMMON_PARAMS)
        rv.update(MUJOCO_SPECIFIC_PARAMS[env_prefix])
        if phase != 'expert':
            rv['irl_lr'] = 1e-3
            if env_prefix in {'CustomAnt', 'DisabledAnt'}:
                rv.update({
                    'batch_size': 10000,
                    'n_itr': 1000,
                })
    elif _is_maze_room(env_name):
        rv = dict(MAZE_PARAMS)
        if 'MazeRoom' in env_name:
            # specific hacks for S-maze
            if phase == 'irl':
                rv.update(S_MAZE_EXTRA_PARAMS_TRAIN)
            elif phase == 'retrain':
                rv.update(S_MAZE_EXTRA_PARAMS_TEST)
    else:
        raise NotImplementedError("I don't know how to handle '%s' yet :(" %
                                  env_prefix)
    missing_keys = REQUIRED_KEYS - rv.keys()
    assert len(missing_keys) == 0, \
        "missing the following keys: %s" % ", ".join(sorted(missing_keys))
    # make copy just in case it gets manipulated
    return dict(rv)


def min_layers_hidsize_polstd_for(env_name):
    """Minimum layer count, hidden size, and init policy std for given
    environment."""
    env_prefix = _remove_version_roboschool(env_name)
    default_polstd = 1.0
    if env_prefix in MUJOCO_SPECIFIC_PARAMS:
        # (64, 64) is meant to work for all simple mujoco envs according to
        # https://github.com/joschu/modular_rl/blob/master/experiments/battery-trpo.yaml
        # however I think it's probably unnecessarily high-dimensional for the
        # simpler ones. Also, Justin uses (32,32) hid sizes for _everything_
        # AFAICT.
        if 'humanoid' in env_name.lower():
            return 2, 64, default_polstd
        return 2, 32, default_polstd
    elif _is_maze_room(env_name):
        if 'MazeRoom' in env_name:
            # S-maze has higher exploration
            return 2, 32, 2.0
        return 2, 32, default_polstd
    raise NotImplementedError("I don't know how to handle '%s' yet :(" %
                              env_prefix)
