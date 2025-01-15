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
        self.data = []  # Store data for plotting
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'r-', lw=2)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(-10, 10)
        self.get_logger().info('Array Subscriber Node has been started.')

        # Create a timer to update the plot periodically
        self.timer = self.create_timer(0.1, self.update_plot)  # 100 ms interval

    def listener_callback(self, msg):
        # Update the data list with the new message
        self.data = msg.data
        self.get_logger().info(f'Received data: {self.data}')

    def update_plot(self):
        # Update the plot with the latest data
        if len(self.data) == 2:
            self.line.set_data([0, 1], self.data)
            self.fig.canvas.draw_idle()  # Force the figure to update

    def run_plot(self):
        # Create the animation (this might not be necessary anymore)
        ani = FuncAnimation(self.fig, self.update_plot, blit=False)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = ArraySubscriber()


    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
