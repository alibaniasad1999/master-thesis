#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class ArraySubscriber(Node):
    def __init__(self):
        super().__init__('mbk_subscriber')
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'mass_spring_damper_env',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning
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
        # Plot the current data (only works for size 2)
        self.line.set_data([0, 1], self.data)
        return self.line,

    def run_plot(self):
        # Create the animation
        ani = FuncAnimation(self.fig, self.animate, blit=True)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = ArraySubscriber()

    # Start the plotting in a separate thread
    from threading import Thread
    plot_thread = Thread(target=node.run_plot)
    plot_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
