#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArraySubscriber(Node):
    def __init__(self):
        super().__init__('mbk_subscriber')
        self.ani = None
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'mass_spring_damper_env',
            self.listener_callback,
            10
        )
        self.data = []  # Store data for plotting
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-10, 10)
        self.get_logger().info('Array Subscriber Node has been started.')

    def listener_callback(self, msg):
        # Update the data list with the new message
        self.data = msg.data
        self.get_logger().info(f'Received data: {self.data}')

    def animate(self, frame):
        # Update the plot only if there is data
        if len(self.data) == 2:
            self.line.set_data([0, 1], self.data)
        else:
            self.line.set_data([], [])  # Clear the plot if data is not valid
        return self.line,

    def run_plot(self):
        # Create the animation
        self.ani = FuncAnimation(self.fig, self.animate, blit=True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = ArraySubscriber()

    # Run the plot in the same thread as ROS spin
    node.run_plot()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
