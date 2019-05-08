import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from rllab.misc import logger

import multiple_irl.envs.rooms as rooms


class PointMassEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 room_type='empty',
                 shaped_reward=True,
                 sparse_reward=False,
                 episode_length=200,
                 start_anywhere=False,
                 stop_early=False,
                 time_varying=False,
                 **kwargs):
        utils.EzPickle.__init__(self, room_type, sparse_reward, episode_length,
                                start_anywhere, stop_early, time_varying,
                                **kwargs)

        self.shaped_reward = shaped_reward
        self.sparse_reward = sparse_reward
        self.start_anywhere = start_anywhere
        self.stop_early = stop_early
        self.time_varying = time_varying

        self.max_episode_length = episode_length

        self.episode_length = 0

        assert room_type in rooms.available_rooms
        self.room = rooms.available_rooms[room_type](**kwargs)

        model = self.room.get_mjcmodel()
        self.goal = self.room.get_target()
        self.start = self.room.get_start()
        self.pX, self.pY = self.room.XY()

        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):

        curr_position = self.get_body_com("particle")[:2].copy()
        vec_dist = np.linalg.norm(curr_position - self.goal)
        reward_dist = -vec_dist  # particle to target

        if self.sparse_reward or self.shaped_reward:
            if vec_dist <= 0.1:
                reward = 2.0
            elif vec_dist <= 0.2:
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = reward_dist

        self.do_simulation(a, self.frame_skip)

        if self.shaped_reward:

            new_position = self.get_body_com("particle")[:2].copy()
            vs = -1 * self.room.get_shaped_distance(curr_position)
            v_sn = -1 * self.room.get_shaped_distance(new_position)
            reward += v_sn - vs

        ob = self._get_obs()
        self.episode_length += 1

        done = self.episode_length >= self.max_episode_length
        if self.stop_early and vec_dist < 0.2:
            done = True
            reward += (self.max_episode_length - self.episode_length)

        return ob, reward, done, dict(distance=vec_dist)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 4.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -90.0

    def reset_model(self):
        qpos = self.init_qpos.copy()

        if self.start_anywhere:
            i = np.random.choice(self.pX.shape[0])
            j = np.random.choice(self.pY.shape[0])

            dX = self.pX[i, j] - self.start[0]
            dY = self.pY[i, j] - self.start[1]

            qpos[0] += dX
            qpos[1] += dY

        self.episode_length = 0

        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01)

        self.set_state(qpos, qvel)
        self.episode_length = 0
        return self._get_obs()

    def _get_obs(self):
        if self.time_varying:
            return np.concatenate([
                self.get_body_com("particle"),
                [self.episode_length / self.max_episode_length]
            ])

        return np.concatenate([
            self.get_body_com("particle"),
        ])

    def plot_trajs(self, *args, **kwargs):
        pass

    def log_diagnostics(self, paths):
        if any([
                len(path['rewards']) != len(paths[0]['rewards'])
                for path in paths
        ]):
            mean_dists = np.array(
                [np.mean(traj['env_infos']['distance']) for traj in paths])
            min_dists = np.array(
                [np.min(traj['env_infos']['distance']) for traj in paths])
            final_dists = np.array(
                [traj['env_infos']['distance'][-1] for traj in paths])

            logger.record_tabular('AvgObjectToGoalDist', np.mean(mean_dists))
            logger.record_tabular('AvgMinToGoalDist', np.mean(min_dists))
            logger.record_tabular('MinMinToGoalDist', np.min(min_dists))
            logger.record_tabular('AvgFinalToGoalDist', np.mean(final_dists))
            logger.record_tabular('MinFinalToGoalDist', np.min(final_dists))
            return

        dist = np.array([traj['env_infos']['distance'] for traj in paths])
        logger.record_tabular('AvgObjectToGoalDist', np.mean(dist.mean()))
        logger.record_tabular('AvgMinToGoalDist', np.mean(
            np.min(dist, axis=1)))
        logger.record_tabular('MinMinToGoalDist', np.min(np.min(dist, axis=1)))
        logger.record_tabular('AvgFinalToGoalDist', np.mean(dist[:, -1]))
        logger.record_tabular('MinFinalToGoalDist', np.min(dist[:, -1]))


def main():
    env = PointMassEnv(
        room_type='target', length=1.2, width=1.2, target=rooms.Target(0))
    obs = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, r, d, e = env.step(action)
        env.render()
        import time
        time.sleep(.05)
    env.close()


if __name__ == "__main__":
    main()
