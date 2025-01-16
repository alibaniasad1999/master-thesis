#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from typing import Any
import time
from mbk_pid_controler_interface.srv import ControlCommand

class MassSpringDamperEnv:

    def __init__(self, dt=None):
        super(MassSpringDamperEnv, self).__init__()

        # System parameters
        self.step_num = 0
        self.last_u = 0
        self.state = np.array([0, 0])
        self.done = False
        self.m = 1.0  # Mass (kg)
        self.k = 1.0  # Spring constant (N/m)
        self.c = 0.1  # Damping coefficient (N*s/m)

        self.action_space_high = 20
        self.action_space_low = -20


        # Simulation parameters
        if dt is not None:
            self.dt = dt
        else:
            self.dt = 0.01  # Time step (s)
        self.max_steps = 1000  # Maximum simulation steps
        self.current_step = 0

        # Integrator
        self.integral_error = 0

        # State and action spaces
        # self.action_space = gym.spaces.Box(low=-20.0, high=20.0, shape=(1,))
        # self.observation_space = gym.spaces.Box(low=-100, high=100, shape=(2,))

    def step(self, action):
        # clip action
        # np.clip(action, -1, 1)
        # Apply control action and simulate one time step using Euler integration
        force = action # * self.action_space.high[0]
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
        position, velocity = (self.state + self.action_space_high) / (
                    self.action_space_high - self.action_space_low)  # normalized data
        return np.array([position, velocity], dtype=np.float32)

    def denormalize(self, state):
        return state * (self.action_space_high - self.action_space_low) - self.action_space_high



class ModelServiceNode(Node):
    def __init__(self):
        super().__init__('model_service_node')

        # Declare the 'time_step' parameter with a default value of 1.0 second
        self.declare_parameter('time_step', 1.0)
        self.time_step = self.get_parameter('time_step').get_parameter_value().double_value

        # Validate the 'time_step' parameter
        if self.time_step <= 0.0:
            self.get_logger().warn(f'Invalid time_step ({self.time_step}). Setting to default 1.0s.')
            self.time_step = 1.0

        # Create a client for the 'compute_control_force' service
        self.client = self.create_client(ControlCommand, 'compute_control_force')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for controller service...')

        # Timer to periodically send state and receive control force
        self.mbk_env = MassSpringDamperEnv(dt=self.time_step)
        state, _ = self.mbk_env.reset()
        self.position, self.velocity = state
        self.timer = self.create_timer(self.time_step, self.timer_callback)

        # Initialize timing variables
        self.last_update_time = self.get_clock().now()

        # Timer to periodically send state and receive control force
        # The timer period is set to the 'time_step' parameter
        self.timer = self.create_timer(self.time_step, self.timer_callback)

        self.get_logger().info('Model Service Node has been started.')



    def timer_callback(self):
        # Calculate the actual elapsed time since the last update
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9  # Convert nanoseconds to seconds
        self.last_update_time = current_time

        # Ensure dt is positive and reasonable
        if dt <= 0.0:
            self.get_logger().warn(f'Non-positive dt encountered: {dt:.4f}s. Skipping update.')
            return
        elif dt > 5.0:  # Allow a maximum dt of 5 seconds to prevent unrealistic jumps
            self.get_logger().warn(f'Large dt encountered: {dt:.4f}s. Clamping to 5.0s.')
            dt = 5.0  # Clamp to a maximum value to prevent unrealistic jumps
        # Create a request with the current state
        request = ControlCommand.Request()
        # change types to float64
        request.position = float(self.position)
        request.velocity = float(self.velocity)

        self.get_logger().info(f'Sending state -> Position: {self.position}, Velocity: {self.velocity}')

        # Call the service
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_service_response)

    def handle_service_response(self, future):
        try:
            response = future.result()
            control_force = response.control_force
            self.get_logger().info(f'Received control force: {control_force}')

            # Update the model's state based on the control force
            self.update_state(control_force)

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def update_state(self, control_force):
        state, _, _, _, _ = self.mbk_env.step(control_force)
        self.position, self.velocity = state

        self.get_logger().info(f'Updated state -> Position: {self.position}, Velocity: {self.velocity}')

def main(args=None):
    rclpy.init(args=args)
    model_service_node = ModelServiceNode()
    try:
        rclpy.spin(model_service_node)
    except KeyboardInterrupt:
        pass
    finally:
        model_service_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
