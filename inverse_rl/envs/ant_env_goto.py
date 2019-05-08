import numpy as np
import time
from gym import utils
from gym.envs.mujoco import mujoco_env
from inverse_rl.envs.dynamic_mjc.model_builder import MJCModel
from rllab.misc import logger


def ant_env(gear=150, eyes=True):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(
        inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(
        data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",
        name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(
        builtin="gradient",
        height="100",
        rgb1="1 1 1",
        rgb2="0 0 0",
        type="skybox",
        width="100")
    asset.texture(
        builtin="flat",
        height="1278",
        mark="cross",
        markrgb="1 1 1",
        name="texgeom",
        random="0.01",
        rgb1="0.8 0.6 0.4",
        rgb2="0.8 0.6 0.4",
        type="cube",
        width="127")
    asset.texture(
        builtin="checker",
        height="100",
        name="texplane",
        rgb1="0 0 0",
        rgb2="0.8 0.8 0.8",
        type="2d",
        width="100")
    asset.material(
        name="MatPlane",
        reflectance="0.5",
        shininess="1",
        specular="1",
        texrepeat="60 60",
        texture="texplane")
    asset.material(name="geom", texture="texgeom", texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(
        friction=[1.5, 0.5, 0.5],
        density=5.0,
        margin=0.01,
        condim=3,
        conaffinity=0,
        rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()
    worldbody.light(
        cutoff="100",
        diffuse=[.8, .8, .8],
        dir="-0 0 -1.3",
        directional="true",
        exponent="1",
        pos="0 0 1.3",
        specular=".1 .1 .1")
    worldbody.geom(
        conaffinity=1,
        condim=3,
        material="MatPlane",
        name="floor",
        pos="0 0 0",
        rgba="0.8 0.9 0.8 1",
        size="40 40 40",
        type="plane")
    worldbody.geom(
        conaffinity=0,
        name="target",
        pos="0 0 0.6",
        rgba="1 0 0 1",
        size="0.2",
        type="sphere")

    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(
        armature="0",
        damping="0",
        limited="false",
        margin="0.01",
        name="root",
        pos=[0, 0, 0],
        type="free")

    if eyes:
        eye_z = 0.1
        eye_y = -.21
        eye_x_offset = 0.07
        # eyes
        ant.geom(
            fromto=[eye_x_offset, 0, eye_z, eye_x_offset, eye_y, eye_z],
            name='eye1',
            size='0.03',
            type='capsule',
            rgba=[1, 1, 1, 1])
        ant.geom(
            fromto=[eye_x_offset, 0, eye_z, eye_x_offset, eye_y - 0.02, eye_z],
            name='eye1_',
            size='0.02',
            type='capsule',
            rgba=[0, 0, 0, 1])
        ant.geom(
            fromto=[-eye_x_offset, 0, eye_z, -eye_x_offset, eye_y, eye_z],
            name='eye2',
            size='0.03',
            type='capsule',
            rgba=[1, 1, 1, 1])
        ant.geom(
            fromto=[
                -eye_x_offset, 0, eye_z, -eye_x_offset, eye_y - 0.02, eye_z
            ],
            name='eye2_',
            size='0.02',
            type='capsule',
            rgba=[0, 0, 0, 1])
        # eyebrows
        ant.geom(
            fromto=[
                eye_x_offset - 0.03, eye_y, eye_z + 0.07, eye_x_offset + 0.03,
                eye_y, eye_z + 0.1
            ],
            name='brow1',
            size='0.02',
            type='capsule',
            rgba=[0, 0, 0, 1])
        ant.geom(
            fromto=[
                -eye_x_offset + 0.03, eye_y, eye_z + 0.07,
                -eye_x_offset - 0.03, eye_y, eye_z + 0.1
            ],
            name='brow2',
            size='0.02',
            type='capsule',
            rgba=[0, 0, 0, 1])

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="aux_1_geom",
        size="0.08",
        type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(
        axis=[0, 0, 1],
        name="hip_1",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="left_leg_geom",
        size="0.08",
        type="capsule")
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(
        axis=[-1, 1, 0],
        name="ankle_1",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge")
    ankle_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
        name="left_ankle_geom",
        size="0.08",
        type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="aux_2_geom",
        size="0.08",
        type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(
        axis=[0, 0, 1],
        name="hip_2",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="right_leg_geom",
        size="0.08",
        type="capsule")
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(
        axis=[1, 1, 0],
        name="ankle_2",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge")
    ankle_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0],
        name="right_ankle_geom",
        size="0.08",
        type="capsule")

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="aux_3_geom",
        size="0.08",
        type="capsule")
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(
        axis=[0, 0, 1],
        name="hip_3",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_3.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="backleft_leg_geom",
        size="0.08",
        type="capsule")
    ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
    ankle_3.joint(
        axis=[-1, 1, 0],
        name="ankle_3",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge")
    ankle_3.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0],
        name="backleft_ankle_geom",
        size="0.08",
        type="capsule")

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="aux_4_geom",
        size="0.08",
        type="capsule")
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(
        axis=[0, 0, 1],
        name="hip_4",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_4.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="backright_leg_geom",
        size="0.08",
        type="capsule")
    ankle_4 = aux_4.body(pos=[0.2, -0.2, 0])
    ankle_4.joint(
        axis=[1, 1, 0],
        name="ankle_4",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge")
    ankle_4.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, -0.4, 0.0],
        name="backright_ankle_geom",
        size="0.08",
        type="capsule")

    actuator = mjcmodel.root.actuator()
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear=gear)
    return mjcmodel


def angry_ant_crippled(gear=150):
    mjcmodel = MJCModel('ant_maze')
    mjcmodel.root.compiler(
        inertiafromgeom="true", angle="degree", coordinate="local")
    mjcmodel.root.option(timestep="0.01", integrator="RK4")
    mjcmodel.root.custom().numeric(
        data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 0.0 1.0 0.0 -1.0 0.0 -1.0 0.0 1.0",
        name="init_qpos")
    asset = mjcmodel.root.asset()
    asset.texture(
        builtin="gradient",
        height="100",
        rgb1="1 1 1",
        rgb2="0 0 0",
        type="skybox",
        width="100")
    asset.texture(
        builtin="flat",
        height="1278",
        mark="cross",
        markrgb="1 1 1",
        name="texgeom",
        random="0.01",
        rgb1="0.8 0.6 0.4",
        rgb2="0.8 0.6 0.4",
        type="cube",
        width="127")
    asset.texture(
        builtin="checker",
        height="100",
        name="texplane",
        rgb1="0 0 0",
        rgb2="0.8 0.8 0.8",
        type="2d",
        width="100")
    asset.material(
        name="MatPlane",
        reflectance="0.5",
        shininess="1",
        specular="1",
        texrepeat="60 60",
        texture="texplane")
    asset.material(name="geom", texture="texgeom", texuniform="true")

    default = mjcmodel.root.default()
    default.joint(armature=1, damping=1, limited='true')
    default.geom(
        friction=[1.5, 0.5, 0.5],
        density=5.0,
        margin=0.01,
        condim=3,
        conaffinity=0,
        rgba="0.8 0.6 0.4 1")

    worldbody = mjcmodel.root.worldbody()

    worldbody.geom(
        conaffinity=1,
        condim=3,
        material="MatPlane",
        name="floor",
        pos="0 0 0",
        rgba="0.8 0.9 0.8 1",
        size="40 40 40",
        type="plane")
    worldbody.light(
        cutoff="100",
        diffuse=[.8, .8, .8],
        dir="-0 0 -1.3",
        directional="true",
        exponent="1",
        pos="0 0 1.3",
        specular=".1 .1 .1")
    worldbody.geom(
        conaffinity=0,
        name="target",
        pos="0 0 0.6",
        rgba="1 0 0 1",
        size="0.2",
        type="sphere")

    ant = worldbody.body(name='torso', pos=[0, 0, 0.75])
    ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
    ant.joint(
        armature="0",
        damping="0",
        limited="false",
        margin="0.01",
        name="root",
        pos=[0, 0, 0],
        type="free")

    eye_z = 0.1
    eye_y = -.21
    eye_x_offset = 0.07
    # eyes
    ant.geom(
        fromto=[eye_x_offset, 0, eye_z, eye_x_offset, eye_y, eye_z],
        name='eye1',
        size='0.03',
        type='capsule',
        rgba=[1, 1, 1, 1])
    ant.geom(
        fromto=[eye_x_offset, 0, eye_z, eye_x_offset, eye_y - 0.02, eye_z],
        name='eye1_',
        size='0.02',
        type='capsule',
        rgba=[0, 0, 0, 1])
    ant.geom(
        fromto=[-eye_x_offset, 0, eye_z, -eye_x_offset, eye_y, eye_z],
        name='eye2',
        size='0.03',
        type='capsule',
        rgba=[1, 1, 1, 1])
    ant.geom(
        fromto=[-eye_x_offset, 0, eye_z, -eye_x_offset, eye_y - 0.02, eye_z],
        name='eye2_',
        size='0.02',
        type='capsule',
        rgba=[0, 0, 0, 1])
    # eyebrows
    ant.geom(
        fromto=[
            eye_x_offset - 0.03, eye_y, eye_z + 0.07, eye_x_offset + 0.03,
            eye_y, eye_z + 0.1
        ],
        name='brow1',
        size='0.02',
        type='capsule',
        rgba=[0, 0, 0, 1])
    ant.geom(
        fromto=[
            -eye_x_offset + 0.03, eye_y, eye_z + 0.07, -eye_x_offset - 0.03,
            eye_y, eye_z + 0.1
        ],
        name='brow2',
        size='0.02',
        type='capsule',
        rgba=[0, 0, 0, 1])

    front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
    front_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="aux_1_geom",
        size="0.08",
        type="capsule")
    aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
    aux_1.joint(
        axis=[0, 0, 1],
        name="hip_1",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
        name="left_leg_geom",
        size="0.08",
        type="capsule")
    ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
    ankle_1.joint(
        axis=[-1, 1, 0],
        name="ankle_1",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge")
    ankle_1.geom(
        fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0],
        name="left_ankle_geom",
        size="0.08",
        type="capsule")

    front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
    front_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="aux_2_geom",
        size="0.08",
        type="capsule")
    aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
    aux_2.joint(
        axis=[0, 0, 1],
        name="hip_2",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0],
        name="right_leg_geom",
        size="0.08",
        type="capsule")
    ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
    ankle_2.joint(
        axis=[1, 1, 0],
        name="ankle_2",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge")
    ankle_2.geom(
        fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0],
        name="right_ankle_geom",
        size="0.08",
        type="capsule")

    # Back left leg is crippled
    thigh_length = 0.1  #0.2
    ankle_length = 0.2  #0.4
    dark_red = [0.8, 0.3, 0.3, 1.0]

    back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
    back_left_leg.geom(
        fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0],
        name="aux_3_geom",
        size="0.08",
        type="capsule",
        rgba=dark_red)
    aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
    aux_3.joint(
        axis=[0, 0, 1],
        name="hip_3",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_3.geom(
        fromto=[0.0, 0.0, 0.0, -thigh_length, -thigh_length, 0.0],
        name="backleft_leg_geom",
        size="0.08",
        type="capsule",
        rgba=dark_red)
    ankle_3 = aux_3.body(pos=[-thigh_length, -thigh_length, 0])
    ankle_3.joint(
        axis=[-1, 1, 0],
        name="ankle_3",
        pos=[0.0, 0.0, 0.0],
        range=[-70, -30],
        type="hinge")
    ankle_3.geom(
        fromto=[0.0, 0.0, 0.0, -ankle_length, -ankle_length, 0.0],
        name="backleft_ankle_geom",
        size="0.08",
        type="capsule",
        rgba=dark_red)

    back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
    back_right_leg.geom(
        fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0],
        name="aux_4_geom",
        size="0.08",
        type="capsule",
        rgba=dark_red)
    aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
    aux_4.joint(
        axis=[0, 0, 1],
        name="hip_4",
        pos=[0.0, 0.0, 0.0],
        range=[-30, 30],
        type="hinge")
    aux_4.geom(
        fromto=[0.0, 0.0, 0.0, thigh_length, -thigh_length, 0.0],
        name="backright_leg_geom",
        size="0.08",
        type="capsule",
        rgba=dark_red)
    ankle_4 = aux_4.body(pos=[thigh_length, -thigh_length, 0])
    ankle_4.joint(
        axis=[1, 1, 0],
        name="ankle_4",
        pos=[0.0, 0.0, 0.0],
        range=[30, 70],
        type="hinge")
    ankle_4.geom(
        fromto=[0.0, 0.0, 0.0, ankle_length, -ankle_length, 0.0],
        name="backright_ankle_geom",
        size="0.08",
        type="capsule",
        rgba=dark_red)

    actuator = mjcmodel.root.actuator()
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear=gear)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3",
        gear=1)  # cripple the joints
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3",
        gear=1)  # cripple the joints
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear=1)
    actuator.motor(
        ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear=1)
    return mjcmodel


class CustomAntGotoEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    A modified ant env with lower joint gear ratios so it flips less often and learns faster.
    """

    def __init__(self,
                 angle_range=(-np.pi, np.pi),
                 max_timesteps=1000,
                 disabled=False,
                 gear=30,
                 sparse_reward=False):
        #mujoco_env.MujocoEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

        self.timesteps = 0
        self.max_timesteps = max_timesteps

        self._set_goal([0, 0])
        self.angle_range = angle_range
        self.sparse_reward = sparse_reward

        if disabled:
            model = angry_ant_crippled(gear=gear)
        else:
            model = ant_env(gear=gear)

        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

        # sync goal with mujoco visualisation
        self._set_goal(self.goal)


    def step(self, a):
        return self._step(a)

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(a))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        state = self._state()
        compos = self.get_body_com('torso')
        forward_reward = -np.linalg.norm(
            (compos[:2] - self.goal)) / (np.linalg.norm(self.goal) + 1e-5)
        survive_reward = 1

        if self.sparse_reward:
            if np.linalg.norm((compos[:2] - self.goal)) < 1.0:
                reward = 1
            else:
                reward = 0
        else:
            reward = forward_reward - ctrl_cost - contact_cost + survive_reward

        #notdone = np.isfinite(state).all() \
        #    and not self.touching(b'torso_geom',b'floor')

        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0

        done = not notdone

        #import IPython; IPython.embed()

        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            distance=np.linalg.norm((compos[:2] - self.goal)))

    # def touching(self, geom1_name, geom2_name):
    #     self.geom_names_to_indices = {name:index for index,name in enumerate(self.model.geom_names)}
    #     idx1 = self.geom_names_to_indices[geom1_name]
    #     idx2 = self.geom_names_to_indices[geom2_name]
    #     import IPython; IPython.embed()

    #     for c in self.sim.data.contact:
    #         if (c.geom1 == idx1 and c.geom2 == idx2) or (c.geom1 == idx2 and c.geom2 == idx1):
    #             return True
    #     return False

    def _get_obs(self):
        current_position = self.get_body_com('torso')
        return np.concatenate([
            self.sim.data.qpos.flat, self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.sim.data.get_body_xmat('torso').flat, self.goal,
            current_position, current_position[:2] - self.goal
        ]).reshape(-1)

    def _state(self):
        return np.concatenate(
            [self.sim.data.qpos.flat, self.sim.data.qvel.flat])

    def _set_goal(self, new_goal):
        new_goal = np.asarray(new_goal)
        assert new_goal.shape == (2, ), \
            "new goal should be (2,) but is %s" % (new_goal.shape, )
        # update internal thing which lets us figure out where the goal is
        self.goal = new_goal

        # also update Mujoco
        if hasattr(self, 'model'):
            goal_geom_idx = self.model.geom_names.index('target')
            assert goal_geom_idx >= 0
            new_geom_pos = self.model.geom_pos.copy()
            self.model.geom_pos[goal_geom_idx] = np.concatenate([new_goal, [0.6]])
            # new_geom_pos[goal_geom_idx] = np.concatenate([new_goal, [0.6]])
            # self.model.geom_pos = new_geom_pos

    def reset_model(self):
        self.timesteps = 0

        angle = self.angle_range[0] + (np.random.rand() * (
            self.angle_range[1] - self.angle_range[0]))
        magnitude = 3
        self._set_goal(np.array(
            [magnitude * np.cos(angle), magnitude * np.sin(angle)]))
        qpos = self.init_qpos.copy().reshape(-1)
        qvel = self.init_qvel.copy().reshape(-1) + np.random.uniform(
            low=-0.005, high=0.005, size=self.model.nv)
        qvel[9:12] = 0
        qpos[-7:-5] = self.goal
        self.set_state(qpos.reshape(-1), qvel)
        # self.current_com = self.sim.data.get_body_com('torso')
        # self.dcom = np.zeros_like(self.current_com)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        forward_rew = np.array(
            [np.mean(traj['env_infos']['reward_forward']) for traj in paths])
        reward_ctrl = np.array(
            [np.mean(traj['env_infos']['reward_ctrl']) for traj in paths])
        reward_cont = np.array(
            [np.mean(traj['env_infos']['reward_contact']) for traj in paths])
        #reward_flip = np.array([np.mean(traj['env_infos']['reward_flipped']) for traj in paths])

        logger.record_tabular('AvgRewardFwd', np.mean(forward_rew))
        logger.record_tabular('AvgRewardCtrl', np.mean(reward_ctrl))
        logger.record_tabular('AvgRewardContact', np.mean(reward_cont))
        #logger.record_tabular('AvgRewardFlipped', np.mean(reward_flip))

        progs = [np.min(path["env_infos"]['distance']) for path in paths]

        progsFinal = [path["env_infos"]['distance'][-1] for path in paths]
        logger.record_tabular(prefix + 'AverageMinDistanceToGoal',
                              np.mean(progs))
        logger.record_tabular(prefix + 'AverageMinDistanceToGoal',
                              np.min(progs))

        logger.record_tabular(prefix + 'AverageFinalDistanceToGoal',
                              np.mean(progsFinal))
        logger.record_tabular(prefix + 'MinFinalDistanceToGoal',
                              np.min(progsFinal))


if __name__ == "__main__":
    env = CustomAntGotoEnv(disabled=True, gear=30)
    env.reset()

    for frame in range(1000):
        env.render()
        env.step(env.action_space.sample())
        if not (frame + 1) % 50:
            # reset periodically to show target moving around
            print('Resetting')
            env.reset()
        time.sleep(1 / 30.0)
