import matplotlib.pyplot as plt
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from rllab.misc import logger

from inverse_rl.envs.dynamic_mjc.mjc_models import point_mass_maze
from multiple_irl.envs.rooms import draw_wall, draw_borders, draw_start_goal


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 direction=1,
                 maze_length=0.6,
                 sparse_reward=False,
                 no_reward=False,
                 episode_length=100):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.direction = direction
        self.length = maze_length

        self.episode_length = 0

        model = point_mass_maze(direction=self.direction, length=self.length)

        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        return self._step(a)

    def _step(self, a):
        vec_dist = self.get_body_com("particle") - self.get_body_com("target")
        vec_dist = np.linalg.norm(vec_dist)

        reward_dist = -vec_dist  # particle to target
        reward_ctrl = -np.square(a).sum()
        # Sam: this line was uncommented in Dibya's copy, but not in Justin's
        # Github code. Going to comment it out so we use dense reward by
        # default.
        # self.sparse_reward = True

        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            if vec_dist <= 0.1:
                reward = 0.0
            elif vec_dist <= 0.2:
                reward = -9.0
            else:
                reward = -99.0
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, dict(
            reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_body_com("particle"),
            #self.get_body_com("target"),
        ])

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        rew_dist = np.array(
            [traj['env_infos']['reward_dist'] for traj in paths])
        rew_ctrl = np.array(
            [traj['env_infos']['reward_ctrl'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', -np.mean(rew_dist.mean()))
        logger.record_tabular('AvgControlCost', -np.mean(rew_ctrl.mean()))
        logger.record_tabular('AvgMinToGoalDist',
                              np.mean(np.min(-rew_dist, axis=1)))

    def draw(self, ax=None):
        if ax is None:
            ax = plt.gca()

        draw_borders(ax, (-0.1, -0.1), (self.length, self.length))
        if self.direction == 0:
            draw_wall(ax, (-0.1, self.length / 2),
                      (2 * self.length / 3, self.length / 2))
        else:
            draw_wall(ax, (self.length / 3, self.length / 2),
                      (self.length, self.length / 2))
        draw_start_goal(
            ax,
            # particle starts at (L/2, 0)
            # (could also get COM for 'particle' but that's not so
            # great after first step)
            (self.length / 2, 0),
            # IDK where target is; only Mujoco knows where target is
            self.get_body_com('target')[:2])

    def _XY(self, n=20):
        X = np.linspace(-0.1, self.length, n)
        Y = np.linspace(-0.1, self.length, n)
        return np.meshgrid(X, Y)

    def draw_reward(self, reward=None, ax=None):
        if ax is None:
            ax = plt.gca()

        if reward is None:

            def reward(x, y):
                # just return negative distance (no shaping here)
                obj_pos = self.get_body_com("target")[:2]
                return -np.linalg.norm(obj_pos - np.array([x, y]))

        X, Y = self._XY()
        H = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                H[i, j] = reward(X[i, j], Y[i, j])
        return ax.contourf(X, Y, H, 30)
