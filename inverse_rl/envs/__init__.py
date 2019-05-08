import logging

from gym.envs import register

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering custom gym environments")
    register(
        id='TwoDMaze-v0', entry_point='inverse_rl.envs.twod_maze:TwoDMaze')
    register(
        id='PointMazeRight-v0',
        entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
        kwargs={
            'sparse_reward': False,
            'direction': 1
        })
    register(
        id='PointMazeLeft-v0',
        entry_point='inverse_rl.envs.point_maze_env:PointMazeEnv',
        kwargs={
            'sparse_reward': False,
            'direction': 0
        })

    # A modified ant which flips over less and learns faster via TRPO
    register(
        id='CustomAnt-v0',
        entry_point='inverse_rl.envs.ant_env:CustomAntEnv',
        kwargs={
            'gear': 30,
            'disabled': False
        })
    register(
        id='DisabledAnt-v0',
        entry_point='inverse_rl.envs.ant_env:CustomAntEnv',
        kwargs={
            'gear': 30,
            'disabled': True
        })


    register(
        id='AntGoto-v0',
        entry_point='inverse_rl.envs.ant_env_goto:CustomAntGotoEnv',
        kwargs={
            'gear': 30,
            'sparse_reward': False,
            'disabled': False,
        })
    register(
        id='DisabledAntGoto-v0',
        entry_point='inverse_rl.envs.ant_env_goto:CustomAntGotoEnv',
        kwargs={
            'gear': 30,
            'sparse_reward': False,
            'disabled': True,
        })
