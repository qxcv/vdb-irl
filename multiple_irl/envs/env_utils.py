# need this to make sure we get roboschool envs
import roboschool  # noqa: F401

import inverse_rl.envs.env_utils as irl_env_utils
from rllab.core.serializable import Serializable
import multiple_irl.envs


class CustomGymEnv(irl_env_utils.CustomGymEnv, Serializable):
    def __init__(self, *args, **kwargs):
        Serializable.quick_init(self, locals())
        assert 'register_fn' not in kwargs
        multiple_irl.envs.register_multitask_envs()
        super(CustomGymEnv, self).__init__(*args, **kwargs)
