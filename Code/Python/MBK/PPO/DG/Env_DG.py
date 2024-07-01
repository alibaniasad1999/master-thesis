# Third-party imports
import numpy as np

import gymnasium as gym
import logging

logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)


class MassSpringDamperEnv(gym.Env):

    def __init__(self):
        super(MassSpringDamperEnv, self).__init__()

        # System parameters
        self.step_num = None
        self.last_u = None
        self.state = None
        self.done = None
        self.m = 1.0  # Mass (kg)
        self.k = 1.0  # Spring constant (N/m)
        self.c = 0.1  # Damping coefficient (N*s/m)

        # Simulation parameters
        self.dt = 0.01  # Time step (s)
        self.max_steps = 1000  # Maximum simulation steps
        self.current_step = 0

        # Integrator
        self.integral_error = 0

        # State and action spaces
        self.action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(2,))

    def step(self, action):
        # clip action
        np.clip(action, -1, 1)
        action_1 = action[0]
        action_2 = action[1] if len(action) > 1 else None
        # Apply control action and simulate one time step using Euler integration
        force = (action_1 + action_2) * self.action_space.high[0]
        position, velocity = self.state

        acceleration = (force - self.c * velocity - self.k * position) / self.m
        velocity += acceleration * self.dt
        position += velocity * self.dt

        self.state = np.array([position, velocity])
        self.integral_error += position * self.dt

        # Linear quadratic differential game cost
        costs = (position ** 2 + 0.1 * velocity ** 2 + 0.01 * self.integral_error ** 2 +
                 0.001 * (action_1 ** 2) - 0.001 * (action_2 ** 2)) * self.dt

        # second player cost
        costs_2 = -costs

        self.step_num += 1
        if self.step_num > 1000:
            self.done = True

        # early stop
        if sum(self.state > 20) > 0 or sum(self.state < -20) > 0:
            self.done = True
            costs += 10

        return self._get_obs(), -costs, self.done, False, {}, -costs_2

    def reset(self):
        self.state = np.random.uniform(low=-10, high=10, size=(2,))
        self.current_step = 0
        self.last_u = None
        self.done = False
        self.step_num = 0
        self.integral_error = 0

        return self._get_obs(), {}

    def _get_obs(self):
        position, velocity = (self.state + self.action_space.high[0]) / (
                    self.action_space.high[0] - self.action_space.low[0])  # normalized data
        return np.array([position, velocity], dtype=np.float32)
