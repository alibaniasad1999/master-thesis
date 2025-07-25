#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
from tbp_interface.srv import ControlCommand
from std_msgs.msg import Float64

# three body problem env
class ThreeBodyEnv:
    def __init__(self, trajectory_, error_range=0.1, final_range=0.1, dt=0.01):
        self.trajectory = trajectory_
        self.state = np.zeros(4)
        self.dt = dt/10
        self.mu = 0.012277471
        # Commented out spaces.Box references
        # self.action_space = spaces.Box(low=-4, high=4, shape=(2,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
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
        # clip action without using action_space
        action = np.clip(action, -4, 4)
        action_2 = np.clip(action_2, -4, 4)

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
            print(("done 🥺"))
            if self.second_player:
                print(("second player was in the game"))
        if self.steps > 20000:
            done = True
            reward = -1000
            print("end time")
            if self.second_player:
                print(("second player was in the game"))
        if self.error_calculation() > self.error_range:
            print(self.state)
            done = True
            reward = -1000 + (nearest_idx / 10000) * 1000
            print('idx', nearest_idx / 100000, 'state', np.linalg.norm(self.state, axis=0))
            print(("too much error 🥲😱"))
            if self.second_player:
                print(("second player was in the game"))

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


class ActuatorDisturbanceNode(Node):
    def __init__(self):
        super().__init__('actuator_disturbance_node')
        df = pd.read_csv('/home/ali/Documents/University/master-thesis/Code/ROS2/src/tbp_model/trajectory.csv')
        df.head()
        # df to numpy array
        data = df.to_numpy()
        print(data.shape)
        trajectory_in = np.delete(data, 2, 1)
        trajectory_in = np.delete(trajectory_in, -1, 1)

        # Declare parameters
        self.declare_parameter('time_step', 1.0)
        self.declare_parameter('disturbance_std', 0.05)  # Standard deviation for actuator disturbance
        self.declare_parameter('noise_std', 0.02)        # Standard deviation for additional noise

        self.time_step = self.get_parameter('time_step').get_parameter_value().double_value
        self.disturbance_std = self.get_parameter('disturbance_std').get_parameter_value().double_value
        self.noise_std = self.get_parameter('noise_std').get_parameter_value().double_value

        # Log parameters
        self.get_logger().info(f'time_step: {self.time_step}')
        self.get_logger().info(f'disturbance_std: {self.disturbance_std}')
        self.get_logger().info(f'noise_std: {self.noise_std}')

        # Validate the parameters
        if self.time_step <= 0.0:
            self.get_logger().warn(f'Invalid time_step ({self.time_step}). Setting to default 1.0s.')
            self.time_step = 1.0
        if self.disturbance_std < 0.0:
            self.get_logger().warn(f'Invalid disturbance_std ({self.disturbance_std}). Setting to default 0.05.')
            self.disturbance_std = 0.05
        if self.noise_std < 0.0:
            self.get_logger().warn(f'Invalid noise_std ({self.noise_std}). Setting to default 0.02.')
            self.noise_std = 0.02

        # Initialize control_force_ as a list
        self.control_force_ = [0.0, 0.0]

        # Create a client for the 'compute_control_force' service
        self.client = self.create_client(ControlCommand, 'compute_control_force')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for controller service...')

        # Timer to periodically send state and receive control force
        self.tbp_env = ThreeBodyEnv(trajectory_in, dt=self.time_step, error_range=0.01, final_range=0.001)
        state, _ = self.tbp_env.reset()
        self.position_x, self.position_y, self.velocity_x, self.velocity_y = state
        self.position_x_pub_data, self.position_y_pub_data, self.velocity_x_pub_data, self.velocity_y_pub_data = trajectory_in[0]
        self.timer = self.create_timer(self.time_step, self.timer_callback)

        # Initialize timing variables
        self.last_update_time = self.get_clock().now()

        # Timer to periodically send state and receive control force
        # The timer period is set to the 'time_step' parameter
        self.timer = self.create_timer(self.time_step, self.timer_callback)

        self.get_logger().info('Actuator Disturbance Node has been started.')

        # publish the position, velocity and control force
        self.position_x_pub = self.create_publisher(Float64, 'position_x', 10)
        self.position_y_pub = self.create_publisher(Float64, 'position_y', 10)
        self.velocity_x_pub = self.create_publisher(Float64, 'velocity_x', 10)
        self.velocity_y_pub = self.create_publisher(Float64, 'velocity_y', 10)
        self.control_force_x_pub = self.create_publisher(Float64, 'control_force_x', 10)
        self.control_force_y_pub = self.create_publisher(Float64, 'control_force_y', 10)
        self.publish_timer = self.create_timer(self.time_step, self.publish_callback)

    def apply_disturbance(self, action):
        """
        Applies Gaussian noise to the action to simulate actuator disturbance.
        """
        action = np.array(action)
        disturbance = np.random.normal(0, self.disturbance_std, size=action.shape)
        return np.clip(action + disturbance, -4, 4)  # Clip to action limits

    def apply_noise(self, action):
        """
        Applies additional random noise to the action.
        """
        noise = np.random.normal(0, self.noise_std, size=action.shape)
        return np.clip(action + noise, -4, 4)  # Clip to action limits

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
        request.position_x = float(self.position_x)
        request.position_y = float(self.position_y)
        request.velocity_x = float(self.velocity_x)
        request.velocity_y = float(self.velocity_y)

        self.get_logger().info(f'Sending state -> Position: ({self.position_x}, {self.position_y}), Velocity: ({self.velocity_x}, {self.velocity_y})')

        # Call the service
        future = self.client.call_async(request)
        future.add_done_callback(self.handle_service_response)

    def handle_service_response(self, future):
        try:
            response = future.result()
            control_force_x = response.control_force_x
            control_force_y = response.control_force_y
            original_control_force = [control_force_x, control_force_y]

            # Apply disturbance and noise to the control force
            disturbed_control_force = self.apply_disturbance(original_control_force)
            noisy_control_force = self.apply_noise(disturbed_control_force)

            self.control_force_ = noisy_control_force
            self.get_logger().info(f'Original control force: {original_control_force}')
            self.get_logger().info(f'Disturbed control force: {self.control_force_}')

            # Update the model's state based on the disturbed control force
            self.update_state(self.control_force_)

        except Exception as e:
            self.get_logger().error(f'Service call failed: {e}')

    def update_state(self, control_force):
        state, reward, done, truncated, position = self.tbp_env.step(control_force)
        self.position_x, self.position_y, self.velocity_x, self.velocity_y = state
        self.position_x_pub_data, self.position_y_pub_data, self.velocity_x_pub_data, self.velocity_y_pub_data = position


        self.get_logger().info(f'Updated state -> Position: ({self.position_x}, {self.position_y}), Velocity: ({self.velocity_x}, {self.velocity_y})')

        # If environment signals it's done, shutdown ROS
        if done:
            self.get_logger().info('Simulation complete. Shutting down ROS node.')
            # Publish final state before shutdown
            self.publish_final_state()
            # Schedule shutdown to allow final messages to be published
            self.create_timer(self.time_step, self.shutdown_ros)

    def publish_final_state(self):
        # Publish the final position, velocity, and control force
        position_x_msg = Float64()
        position_x_msg.data = float(self.position_x_pub_data)
        self.position_x_pub.publish(position_x_msg)

        position_y_msg = Float64()
        position_y_msg.data = float(self.position_y_pub_data)
        self.position_y_pub.publish(position_y_msg)

        velocity_x_msg = Float64()
        velocity_x_msg.data = float(self.velocity_x_pub_data)
        self.velocity_x_pub.publish(velocity_x_msg)

        velocity_y_msg = Float64()
        velocity_y_msg.data = float(self.velocity_y_pub_data)
        self.velocity_y_pub.publish(velocity_y_msg)

        control_force_x_msg = Float64()
        control_force_x_msg.data = float(self.control_force_[0])
        self.control_force_x_pub.publish(control_force_x_msg)

        control_force_y_msg = Float64()
        control_force_y_msg.data = float(self.control_force_[1])
        self.control_force_y_pub.publish(control_force_y_msg)

        self.get_logger().info(f'Final state published -> Position x: {position_x_msg.data}, Position y: {position_y_msg.data}, Velocity x: {velocity_x_msg.data}, Velocity y: {velocity_y_msg.data}, Control Force x: {control_force_x_msg.data}, Control Force y: {control_force_y_msg.data}')

    def shutdown_ros(self):
        # Shutdown ROS node
        self.get_logger().info('Shutting down ROS...')
        rclpy.shutdown()

    def publish_callback(self):
        # Publish the position, velocity, and control force
        position_x_msg = Float64()
        position_x_msg.data = float(self.position_x_pub_data)
        self.position_x_pub.publish(position_x_msg)
        print(position_x_msg.data)

        position_y_msg = Float64()
        position_y_msg.data = float(self.position_y_pub_data)
        self.position_y_pub.publish(position_y_msg)
        print(position_y_msg.data)

        velocity_x_msg = Float64()
        velocity_x_msg.data = float(self.velocity_x_pub_data)
        self.velocity_x_pub.publish(velocity_x_msg)
        print(velocity_x_msg.data)

        velocity_y_msg = Float64()
        velocity_y_msg.data = float(self.velocity_y_pub_data)
        self.velocity_y_pub.publish(velocity_y_msg)
        print(velocity_y_msg.data)

        control_force_x_msg = Float64()
        control_force_x_msg.data = float(self.control_force_[0])
        self.control_force_x_pub.publish(control_force_x_msg)
        print(control_force_x_msg.data)

        control_force_y_msg = Float64()
        control_force_y_msg.data = float(self.control_force_[1])
        self.control_force_y_pub.publish(control_force_y_msg)
        print(control_force_y_msg.data)

        self.get_logger().info(f'Published -> Position x: {position_x_msg.data}, Position y: {position_y_msg.data}, Velocity x: {velocity_x_msg.data}, Velocity y: {velocity_y_msg.data}, Control Force x: {control_force_x_msg.data}, Control Force y: {control_force_y_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    actuator_disturbance_node = ActuatorDisturbanceNode()
    try:
        rclpy.spin(actuator_disturbance_node)
    except KeyboardInterrupt:
        pass
    finally:
        actuator_disturbance_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
