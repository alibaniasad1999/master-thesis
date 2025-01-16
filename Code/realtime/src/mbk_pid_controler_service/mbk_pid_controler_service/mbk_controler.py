#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from mbk_pid_controler_interface.srv import ControlCommand

class ControllerServiceNode(Node):
    def __init__(self):
        super().__init__('controller_service_node')

        # Create the service server
        self.srv = self.create_service(ControlCommand, 'compute_control_force', self.compute_control_callback)

        # Controller parameters
        self.Kp = 1.0  # Proportional gain
        self.setpoint = 10.0  # Desired position

        self.get_logger().info('Controller Service Node has been started and is ready to compute control forces.')

    def compute_control_callback(self, request, response):
        # Extract the state from the request
        current_position = request.position
        current_velocity = request.velocity

        self.get_logger().info(f'Received state -> Position: {current_position}, Velocity: {current_velocity}')

        # Compute the error
        error = self.setpoint - current_position

        # Compute control force using a proportional controller
        control_force = self.Kp * error

        self.get_logger().info(f'Computed control force: {control_force}')

        # Set the response
        response.control_force = float(control_force)

        return response

def main(args=None):
    rclpy.init(args=args)
    controller_service_node = ControllerServiceNode()
    try:
        rclpy.spin(controller_service_node)
    except KeyboardInterrupt:
        pass
    finally:
        controller_service_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
