#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from typing import Any



class MassSpringDamperEnv():

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
        self.dt = 1  # Time step (s)
        self.max_steps = 1000  # Maximum simulation steps
        self.current_step = 0

        # Integrator
        self.integral_error = 0

        # State and action spaces

    def step(self, action):
        # clip action
        # np.clip(action, -1, 1)
        # Apply control action and simulate one time step using Euler integration
        force = action[0]  # * self.action_space.high[0]
        position, velocity = self.state

        acceleration = (force - self.c * velocity - self.k * position) / self.m
        velocity += acceleration * self.dt
        position += velocity * self.dt

        self.state = np.array([position, velocity])
        self.integral_error += position * self.dt

        costs = (position ** 2 + 0.1 * velocity ** 2
                 + 0.01 * self.integral_error ** 2 + 0.001 * (force ** 2)) * self.dt

        self.step_num += 1
        if self.step_num > 1000:
            self.done = True

        # early stop
        if sum(self.state > 20) > 0 or sum(self.state < -20) > 0:
            self.done = True
            costs += 10

        return self._get_obs(), -costs, self.done, False, {}

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
        self.state = np.random.uniform(low=-10, high=10, size=(2,))
        self.current_step = 0
        self.last_u = None
        self.done = False
        self.step_num = 0
        self.integral_error = 0

        return self._get_obs(), {}

    def _get_obs(self):
        position, velocity = (self.state + 20) / (
                20 + 20)  # normalized data
        return np.array([position, velocity], dtype=np.float32)

    def denormalize(self, state):
        return state * (self.action_space.high[0] - self.action_space.low[0]) - self.action_space.high[0]


class MassSpringDamperEnvPublisher(Node):
    def __init__(self):
        super().__init__("mass_spring_damper_env_publisher")
        self.mass_spring_damper_env_ = MassSpringDamperEnv()
        self.mass_spring_damper_env_.reset()
        self.mass_spring_damper_env_publisher_ = self.create_publisher(Float64MultiArray, "mass_spring_damper_env", 10)
        self.mass_spring_damper_env_timer_ = self.create_timer(1.0, self.publish_mass_spring_damper_env)
        self.get_logger().info("Mass spring damper environment publisher has been started.")

    def publish_mass_spring_damper_env(self):
        msg = Float64MultiArray()
        # random action
        action = np.random.uniform(low=-20, high=20, size=(1,))
        state, _, _, _, _ = self.mass_spring_damper_env_.step(action)
        msg.data = [state[0], state[1]]
        self.mass_spring_damper_env_publisher_.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = MassSpringDamperEnvPublisher()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
