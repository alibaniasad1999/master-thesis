import os
import urllib.request
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt

if not os.path.isfile("trajectory.csv"):
    url = "https://raw.githubusercontent.com/alibaniasad1999/master-thesis/main/Code/Python/TBP/SAC/trajectory.csv"
    print("Downloading trajectory.csv...")
    urllib.request.urlretrieve(url, "trajectory.csv")
    print("Download complete.")
else:
    print("trajectory.csv already exists.")

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)



# three body problem env
class ThreeBodyEnv(gym.Env):
    def __init__(self, trajectory_, error_range=0.1, final_range=0.1):
        self.trajectory = trajectory_
        self.state = np.zeros(4)
        self.dt = 0.001
        self.mu = 0.012277471
        self.action_space = spaces.Box(low=-4, high=4, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.position = trajectory_[0]
        self.steps = 0
        self.max_steps = 6000
        self.final_range = final_range
        self.error_range = error_range
        self.reward_range = (-float('inf'), float('inf'))
        self.render_logic = False
        # second player
        self.second_player = True
        self.reset()

    def step(self, action, action_2=np.zeros(2)):
        x = self.position[0]
        y = self.position[1]
        xdot = self.position[2]
        ydot = self.position[3]

        # force = action[0] * env.state[2:] + action[1] * env.state[:2]
        a_x = action[0] / 100
        a_y = action[1] / 100
        # add second player action
        a_x_2 = action_2[0] / 200 if self.second_player else 0
        a_y_2 = action_2[1] / 200 if self.second_player else 0

        r1 = np.sqrt((x + self.mu) ** 2 + y ** 2)
        r2 = np.sqrt((x - 1 + self.mu) ** 2 + y ** 2)

        xddot = 2 * ydot + x - (1 - self.mu) * ((x + self.mu) / (r1 ** 3)) - self.mu * (x - 1 + self.mu) / (
                    r2 ** 3) + a_x + a_x_2
        yddot = -2 * xdot + y - (1 - self.mu) * (y / (r1 ** 3)) - self.mu * y / (r2 ** 3) + a_y + a_y_2

        x = x + xdot * self.dt
        y = y + ydot * self.dt

        xdot = xdot + xddot * self.dt
        ydot = ydot + yddot * self.dt

        self.position = np.array([x, y, xdot, ydot])

        self.steps += 1

        self.position2state()

        # plot position
        if self.render_logic:
            plt.plot(x, y, 'ro')
            plt.plot(self.trajectory[:, 0], self.trajectory[:, 1])
            plt.show()

        distance = np.linalg.norm(self.trajectory[:, 0:2] - self.position[0:2],
                                  axis=1)  # just add position and delete velocity
        nearest_idx = np.argmin(distance)
        reward = 100 * (
                    1 - np.linalg.norm(self.state, axis=0) - (a_x / 10) ** 2 - (a_y / 10) ** 2 + (a_x_2 / 10) ** 2 + (
                        a_y_2 / 10) ** 2) - 100
        done = self.steps >= self.max_steps
        if np.linalg.norm(self.position[0:2] - self.trajectory[-1, 0:2]) < self.final_range:
            done = True
            reward = 1000
            print(colorize("done 🥺", 'green', bold=True))
            if self.second_player:
                print(colorize("second player was in the game", 'blue'))
        if self.steps > 20000:
            done = True
            reward = -1000
            print("end time")
            if self.second_player:
                print(colorize("second player was in the game", 'blue'))
        if self.error_calculation() > self.error_range:
            print(self.state)
            done = True
            reward = -1000 + (nearest_idx / 10000) * 1000
            print('idx', nearest_idx / 100000, 'state', np.linalg.norm(self.state, axis=0))
            print(colorize("too much error 🥲😱", 'red', bold=True))
            if self.second_player:
                print(colorize("second player was in the game", 'blue'))

        # print(self.state, reward, done, self.position)
        return 1000 * self.state, reward, done, False, self.position

    def position2state(self):
        # find the nearest point from position to trajectory
        distance = np.linalg.norm(self.trajectory[:, 0:2] - self.position[0:2],
                                  axis=1)  # just add position and delete velocity
        nearest_idx = np.argmin(distance)
        # estate = position - nearest(index)
        self.state = self.position - self.trajectory[nearest_idx]
        # self.state = self.state * np.array([10, 10, 1, 1])

    def error_calculation(self):
        normalized_error = self.state * np.array([1, 1, 0.0, 0.0])  # reduce the effect of velocity error
        return np.linalg.norm(normalized_error)

    def reset(self,
              *,
              seed: 5 = None,
              return_info: bool = False,
              options: 6 = None):
        self.position = self.trajectory[0]
        self.steps = 0
        self.position2state()
        return 1000 * self.state, {}