# VDB IRL experiments

Implementation of Adversarial IRL (AIRL) with information bottleneck. Used in
the [Variational Discriminator Bottleneck (VDB) paper at
ICLR](https://openreview.net/pdf?id=HyxPx3R9tm).

## Getting Set Up

- Install `rllab` if not already. Try
  [qxcv/rllab](https://github.com/qxcv/rllab/tree/minor-fixes) on the
  `minor-fixes` branch (which adds some missing hooks & addresses some bugs in
  the original RLLab).
- Add folders `multiple_irl`, `inverse_rl`, and `scripts` to python path (to double check that this works, just try importing `multiple_irl.envs` from a python shell). The easiest way to do this is using the `setup.py` file in this directory, with `pip install -e .`.

When running scripts, you ought to run then directly from the root folder of the git repository.

## Developing

The core algorithm is in `multiple_irl/models/shaped_irl.py.

All the environments are in `multiple_irl/envs`: you can also find a comprehensive list of environments below in the README.

All the scripts are in `scripts/scripts_generic`. The scripts do the following:

- data_collect.py: This script collects expert trajectories.
- env_compare.py: This script trains an AIRL reward function on a single task.
- env_retrain.py: Takes a trained AIRL reward function and uses it to train a
  new policy from scratch in an environment.
