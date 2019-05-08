from inverse_rl.envs.dynamic_mjc.model_builder import MJCModel
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def add_wall(worldbody, start, end, name='wall', color="0.9 0.4 0.6 1"):
    worldbody.geom(
        conaffinity=1,
        fromto=[*start, .01, *end, .01],
        name=name,
        rgba=color,
        size=".02",
        type="capsule")


def make_borders(worldbody,
                 botLeft,
                 topRight,
                 prefix='side',
                 color="0.9 0.4 0.6 1"):
    botRight = [topRight[0], botLeft[1]]
    topLeft = [botLeft[0], topRight[1]]

    add_wall(worldbody, botLeft, botRight, '%sS' % prefix, color)
    add_wall(worldbody, botRight, topRight, '%sE' % prefix, color)
    add_wall(worldbody, botLeft, topLeft, '%sW' % prefix, color)
    add_wall(worldbody, topLeft, topRight, '%sN' % prefix, color)


def set_goal(worldbody, position):
    target = worldbody.body(name='target', pos=[*position, 0])
    target.geom(
        name='target_geom',
        conaffinity=2,
        type='sphere',
        size=0.02,
        rgba=[0, 0.9, 0.1, 1])


def draw_wall(ax, start, end):
    if np.isclose(start[0], end[0]):
        ax.vlines(
            start[0],
            start[1],
            end[1],
            linewidth=4,
        )
    elif np.isclose(start[1], end[1]):
        ax.hlines(
            start[1],
            start[0],
            end[0],
            linewidth=4,
        )
    else:
        raise NotImplementedError()


def draw_borders(ax, botLeft, topRight):
    botRight = [topRight[0], botLeft[1]]
    topLeft = [botLeft[0], topRight[1]]

    draw_wall(ax, botLeft, botRight)
    draw_wall(ax, botRight, topRight)
    draw_wall(ax, botLeft, topLeft)
    draw_wall(ax, topLeft, topRight)

    # draw white patch underneath, just in case there's any transparent
    # background
    width = topRight[0] - botLeft[0]
    height = topRight[1] - botLeft[1]
    ax.add_patch(patches.Rectangle(
        botLeft, width, height, fill=True, facecolor='white', zorder=-1))


def draw_start_goal(ax, start, goal):
    ax.scatter([start[0]], [start[1]], c='r', s=400)
    ax.scatter([goal[0]], [goal[1]], c='g', s=400)


def shell_pointmass(start_pos=(0, 0)):
    mjcmodel = MJCModel('pointmass')
    mjcmodel.root.compiler(
        inertiafromgeom="true", angle="radian", coordinate="local")
    mjcmodel.root.option(
        timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(
        friction=".5 .1 .1",
        density="1000",
        margin="0.002",
        condim="1",
        contype="2",
        conaffinity="1")

    worldbody = mjcmodel.root.worldbody()

    particle = worldbody.body(name='particle', pos=[*start_pos, 0])
    particle.geom(
        name='particle_geom',
        type='sphere',
        size='0.03',
        rgba='0.0 0.0 1.0 1',
        contype=1)
    particle.site(name='particle_site', pos=[0, 0, 0], size=0.01)
    particle.joint(name='ball_x', type='slide', pos=[0, 0, 0], axis=[1, 0, 0])
    particle.joint(name='ball_y', type='slide', pos=[0, 0, 0], axis=[0, 1, 0])

    actuator = mjcmodel.root.actuator()
    # XXX: turning up control limits because the ball can't zip around smoothly
    # enough (Sam 2018-09-25)
    actuator.motor(joint="ball_x", ctrlrange=[-5.0, 5.0], ctrllimited=True)
    actuator.motor(joint="ball_y", ctrlrange=[-5.0, 5.0], ctrllimited=True)

    return mjcmodel, worldbody


class EmptyRoom:
    def __init__(self, length=1.2, width=1.2):
        self.length = length
        self.width = width
        self.mjcmodel, self.worldbody = self.create_mjcmodel()

    def create_mjcmodel(self):

        mjcmodel, worldbody = shell_pointmass(self.get_start())
        make_borders(worldbody, (-0.2, -self.width / 2),
                     (self.length - 0.2, self.width / 2))

        set_goal(worldbody, self.get_target())

        return mjcmodel, worldbody

    def get_mjcmodel(self):
        return self.mjcmodel

    def get_start(self):
        return np.array((0, 0))

    def get_target(self):
        return np.array((self.length - 0.5, 0))

    def get_shaped_distance(self, position):
        return np.linalg.norm(position - self.get_target())

    def draw(self, ax=None):

        if ax is None:
            ax = plt.gca()

        draw_borders(ax, (-0.2, -self.width / 2),
                     (self.length - 0.2, self.width / 2))
        draw_start_goal(ax, self.get_start(), self.get_target())

    def XY(self, n=20):
        X = np.linspace(-0.2, self.length - 0.2, n)
        Y = np.linspace(-self.width / 2, self.width / 2, n)
        return np.meshgrid(X, Y)

    def draw_reward(self, reward=None, ax=None):
        if ax is None:
            ax = plt.gca()

        if reward is None:
            reward = lambda x,y: -1 * self.get_shaped_distance(np.array([x,y]))

        X, Y = self.XY()
        H = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                H[i, j] = reward(X[i, j], Y[i, j])
        return ax.contourf(X, Y, H, 30)


class RoomWithWall(EmptyRoom):
    def create_mjcmodel(self):

        mjcmodel, worldbody = shell_pointmass(self.get_start())

        make_borders(worldbody, (-self.length / 2, -self.width / 2),
                     (self.length / 2, self.width / 2))
        add_wall(
            worldbody, (-self.length / 2, 0), (self.length / 6, 0),
            name='wall')

        set_goal(worldbody, self.get_target())
        return mjcmodel, worldbody

    def get_start(self):
        return np.array((-self.length / 6, -self.width / 4))

    def get_target(self):
        return np.array((-self.width / 6, self.width / 4))

    def get_shaped_distance(self, position):
        if position[1] > 0:
            return np.linalg.norm(position - self.get_target())
        else:
            midpoint = np.array([self.length / 6, 0])
            return np.linalg.norm(position - midpoint) + np.linalg.norm(
                midpoint - self.get_target())

    def draw(self, ax=None):

        if ax is None:
            ax = plt.gca()

        draw_borders(ax, (-self.length / 2, -self.width / 2),
                     (self.length / 2, self.width / 2))
        draw_wall(ax, (-self.length / 2, 0), (self.length / 6, 0))
        draw_start_goal(ax, self.get_start(), self.get_target())

    def XY(self, n=20):
        X = np.linspace(-self.length / 2, self.length / 2, n)
        Y = np.linspace(-self.width / 2, self.width / 2, n)
        return np.meshgrid(X, Y)


class TwoRoom(EmptyRoom):
    def __init__(self, length=1.2):
        super().__init__(length=length, width=2 * length)

    def create_mjcmodel(self):

        mjcmodel, worldbody = shell_pointmass(self.get_start())

        make_borders(worldbody, (-self.length / 2, -self.width / 2),
                     (self.length / 2, self.width / 2))

        add_wall(
            worldbody, (-self.length / 2, 0), (-self.length / 6, 0),
            name='lwall')
        add_wall(
            worldbody, (self.length / 6, 0), (self.length / 2, 0),
            name='rwall')

        set_goal(worldbody, self.get_target())

        return mjcmodel, worldbody

    def get_start(self):
        return np.array((0, -self.width / 2 + 0.2))

    def get_target(self):
        return np.array((self.length / 2 - 0.2, self.width / 2 - 0.2))

    def get_shaped_distance(self, position):
        if position[1] > 0.05:
            return np.linalg.norm(position - self.get_target())
        else:
            midpoint = np.array([0, 0])
            return np.linalg.norm(position - midpoint) + np.linalg.norm(
                midpoint - self.get_target())

    def draw(self, ax=None):

        if ax is None:
            ax = plt.gca()

        draw_borders(ax, (-self.length / 2, -self.width / 2),
                     (self.length / 2, self.width / 2))
        draw_wall(ax, (-self.length / 2, 0), (-self.length / 6, 0))
        draw_wall(ax, (self.length / 6, 0), (self.length / 2, 0))

        draw_start_goal(ax, self.get_start(), self.get_target())

    def XY(self, n=20):
        X = np.linspace(-self.length / 2, self.length / 2, n)
        Y = np.linspace(-self.width / 2, self.width / 2, n)
        return np.meshgrid(X, Y)


class MazeRoom(EmptyRoom):
    def __init__(self, length=1.2, n_walls=2, flip=False):
        self.n_walls = n_walls
        self.flip = bool(flip)
        super().__init__(length, width=0.6 * (n_walls + 1))

    def get_start(self):
        return np.array((0, 0.3))

    def get_target(self):
        return np.array((0, self.width - 0.3))

    def get_shaped_distance(self, position):
        # "width" is really the distance along the maze in the direction from
        # start to target (I think of that as length, but evidently that's not
        # what length is here). The cells demarcated by a pair of walls is
        # about 0.6 across, so if the condition below is true then we're in the
        # final cell & can run straight to the target.
        if position[1] > self.width - 0.6 + 0.05:
            return np.linalg.norm(position - self.get_target())
        else:
            distance = 0
            # "triggered" means that we've found the nearest wall that we
            # *haven't* passed yet.
            triggered = False
            for i in range(1, self.n_walls + 1):
                if triggered:
                    # I think this is just the distance from the middle of one
                    # gap to the middle of the next target gap
                    distance += ((self.length / 3)**2 + 0.6**2)**0.5
                elif 0.6 * i + 0.05 > position[1]:
                    # we're just before this wall (but have passed all the
                    # previous ones)
                    triggered = True
                    if self.flip ^ (i % 2 == 1):
                        # I assume this is the point at which the gap occurs
                        point = np.array((self.length / 6 + 0.1, 0.6 * i))
                    else:
                        point = np.array((-self.length / 6 - 0.1, 0.6 * i))
                    distance += np.linalg.norm(point - position)
                else:
                    pass  # Already passed this wall
            distance += ((self.length / 6 + 0.1)**2 + 0.3**2)**0.5
            return distance

    def create_mjcmodel(self):

        mjcmodel, worldbody = shell_pointmass(self.get_start())

        make_borders(worldbody, (-self.length / 2, 0),
                     (self.length / 2, 0.6 * (self.n_walls + 1)))
        for i in range(1, self.n_walls + 1):
            if self.flip ^ (i % 2 == 1):
                add_wall(
                    worldbody, (-self.length / 2, 0.6 * i),
                    (self.length / 6, 0.6 * i),
                    name='wall%d' % i)
            else:
                add_wall(
                    worldbody, (-self.length / 6, 0.6 * i),
                    (self.length / 2, 0.6 * i),
                    name='wall%d' % i)

        set_goal(worldbody, self.get_target())

        return mjcmodel, worldbody

    def draw(self, ax=None):

        if ax is None:
            ax = plt.gca()

        draw_borders(ax, (-self.length / 2, 0),
                     (self.length / 2, 0.6 * (self.n_walls + 1)))
        for i in range(1, self.n_walls + 1):
            if self.flip ^ (i % 2 == 1):
                draw_wall(ax, (-self.length / 2, 0.6 * i),
                          (self.length / 6, 0.6 * i))
            else:
                draw_wall(ax, (-self.length / 6, 0.6 * i),
                          (self.length / 2, 0.6 * i))

        draw_start_goal(ax, self.get_start(), self.get_target())

    def XY(self, n=20):
        X = np.linspace(-self.length / 2, self.length / 2, n)
        Y = np.linspace(0, self.width, n)
        return np.meshgrid(X, Y)


from enum import Enum


class Target(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_LEFT = 2
    BOTTOM_RIGHT = 3


class MultipleTargetRoom(EmptyRoom):
    def __init__(self, length=1.2, width=1.2, target=Target.TOP_LEFT):
        self.target = target
        EmptyRoom.__init__(self, length, width)

    def create_mjcmodel(self):

        mjcmodel, worldbody = shell_pointmass(self.get_start())
        make_borders(worldbody, (-self.length / 2, -self.width / 2),
                     (self.length / 2, self.width / 2))

        set_goal(worldbody, self.get_target())

        return mjcmodel, worldbody

    def get_mjcmodel(self):
        return self.mjcmodel

    def get_start(self):
        return np.array((0, 0))

    def get_target(self):
        targets = {
            Target.TOP_LEFT: (-self.length / 2 + 0.2, self.width / 2 - 0.2),
            Target.TOP_RIGHT: (self.length / 2 - 0.2, self.width / 2 - 0.2),
            Target.BOTTOM_LEFT: (-self.length / 2 + 0.2,
                                 -self.width / 2 + 0.2),
            Target.BOTTOM_RIGHT: (self.length / 2 - 0.2,
                                  -self.width / 2 + 0.2),
        }
        return np.array(targets[self.target])

    def get_shaped_distance(self, position):
        return np.linalg.norm(position - self.get_target())

    def draw(self, ax=None):

        if ax is None:
            ax = plt.gca()

        draw_borders(ax, (-self.length / 2, -self.width / 2),
                     (self.length / 2, self.width / 2))
        draw_start_goal(ax, self.get_start(), self.get_target())

    def XY(self, n=20):
        X = np.linspace(-self.length / 2, self.length / 2, n)
        Y = np.linspace(-self.width / 2, self.width / 2, n)
        return np.meshgrid(X, Y)

    def draw_reward(self, reward=None, ax=None):
        if ax is None:
            ax = plt.gca()

        if reward is None:
            reward = lambda x,y: -1 * self.get_shaped_distance(np.array([x,y]))

        X, Y = self.XY()
        H = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                H[i, j] = reward(X[i, j], Y[i, j])
        return ax.contourf(X, Y, H, 30)


available_rooms = {
    'empty': EmptyRoom,
    'wall': RoomWithWall,
    'maze': MazeRoom,
    'two': TwoRoom,
    'target': MultipleTargetRoom
}

if __name__ == "__main__":
    fig, axes = plt.subplots(3, figsize=(4, 12))
    EmptyRoom().draw(axes[0])
    RoomWithWall().draw(axes[1])
    MazeRoom().draw(axes[2])
    plt.show()
