import numpy as np
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides


class ConstantBaseline(Baseline):
    def __init__(self, env_spec):
        self._mean = 0

    @overrides
    def get_param_values(self, **kwargs):
        return self._mean

    @overrides
    def set_param_values(self, val, **kwargs):
        self._mean = val

    @overrides
    def fit(self, paths):
        rewards = np.concatenate([path["rewards"] for path in paths])
        self._mean = np.mean(rewards)

    @overrides
    def predict(self, path):
        return path["rewards"] - self._mean
