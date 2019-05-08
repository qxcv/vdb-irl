import logging

from gym.envs import register

from inverse_rl.envs import register_custom_envs
register_irl_envs = register_custom_envs

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_multitask_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")

    register_irl_envs()

    pmroom_env_defs = {
        'PM_EmptyRoom_Small-v0': dict(
            room_type='empty', length=0.6, width=0.6),
        'PM_EmptyRoom_Med-v0': dict(room_type='empty', length=1.2, width=1.2),
        'PM_EmptyRoom_Large-v0': dict(
            room_type='empty', length=2.4, width=2.4),
        'PM_WallRoom_Small-v0': dict(room_type='wall', length=0.6, width=0.6),
        'PM_WallRoom_Med-v0': dict(room_type='wall', length=1.2, width=1.2),
        'PM_WallRoom_Large-v0': dict(room_type='wall', length=2.4, width=2.4),
        'PM_MazeRoom_Small-v0': dict(room_type='maze', length=0.6, n_walls=2),
        'PM_MazeRoom_Med-v0': dict(room_type='maze', length=1.2, n_walls=2),
        'PM_MazeRoom_Large-v0': dict(room_type='maze', length=1.2, n_walls=4),
        'PM_MazeRoom_Flip_Small-v0': dict(room_type='maze', length=0.6,
                                          n_walls=2, flip=True),
        'PM_MazeRoom_Flip_Med-v0': dict(room_type='maze', length=1.2,
                                        n_walls=2, flip=True),
        'PM_MazeRoom_Flip_Large-v0': dict(room_type='maze', length=1.2,
                                          n_walls=4, flip=True),
        'PM_TwoRoom_Small-v0': dict(room_type='two', length=0.6),
        'PM_TwoRoom_Med-v0': dict(room_type='two', length=1.2),
        'PM_TwoRoom_Large-v0': dict(room_type='two', length=2.4),
    }

    all_env_defs = {
        'multiple_irl.envs.pm_room_env:PointMassEnv': pmroom_env_defs,
    }

    for class_name, env_defs in all_env_defs.items():

        for env_name, kwarg_dict in env_defs.items():

            sparse_kwargs = kwarg_dict.copy()
            sparse_kwargs.update(shaped_reward=False, sparse_reward=True)

            shaped_kwargs = kwarg_dict.copy()
            shaped_kwargs.update(shaped_reward=True, sparse_reward=False)

            any_kwargs = kwarg_dict.copy()
            any_kwargs.update(
                shaped_reward=False, sparse_reward=True, start_anywhere=True)

            normal_kwargs = kwarg_dict.copy()
            normal_kwargs.update(shaped_reward=False, sparse_reward=False)

            stop_kwargs = kwarg_dict.copy()
            stop_kwargs.update(
                shaped_reward=False, sparse_reward=True, stop_early=True)

            tv_kwargs = kwarg_dict.copy()
            tv_kwargs.update(
                shaped_reward=False, sparse_reward=True, time_varying=True)

            tvany_kwargs = kwarg_dict.copy()
            tvany_kwargs.update(
                shaped_reward=False,
                sparse_reward=True,
                start_anywhere=True,
                time_varying=True)

            register(
                id='Sparse_%s' % env_name,
                entry_point=class_name,
                kwargs=sparse_kwargs)

            register(
                id='Shaped_%s' % env_name,
                entry_point=class_name,
                kwargs=shaped_kwargs)

            register(
                id='Any_%s' % env_name,
                entry_point=class_name,
                kwargs=any_kwargs)

            register(
                id='Normal_%s' % env_name,
                entry_point=class_name,
                kwargs=normal_kwargs)

            register(
                id='Stop_%s' % env_name,
                entry_point=class_name,
                kwargs=stop_kwargs)

            register(
                id='TV_%s' % env_name,
                entry_point=class_name,
                kwargs=tv_kwargs)

            register(
                id='TVAny_%s' % env_name,
                entry_point=class_name,
                kwargs=tvany_kwargs)
