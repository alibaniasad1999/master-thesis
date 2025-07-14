import csv
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class DataListenerNode(Node):
    """ROS 2 node that subscribes to the six Float64 topics and logs
    each message to both the console and a CSV file.
    """

    def __init__(self):
        super().__init__('data_listener_node')

        qos = 10  # Depth 10 to mirror publisher QoS

        # ------------------------------------------------------------------
        # CSV setup
        # ------------------------------------------------------------------
        log_dir = Path.home() / 'ros2_logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = log_dir / f'data_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv'
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'topic', 'value'])
        self.get_logger().info(f'Logging incoming data to {filename}')

        # ------------------------------------------------------------------
        # Subscriptions (one‑liner lambdas share a common handler)
        # ------------------------------------------------------------------
        self.create_subscription(Float64, 'position_x',
                                 lambda msg: self._handle('position_x', msg.data), qos)
        self.create_subscription(Float64, 'position_y',
                                 lambda msg: self._handle('position_y', msg.data), qos)
        self.create_subscription(Float64, 'velocity_x',
                                 lambda msg: self._handle('velocity_x', msg.data), qos)
        self.create_subscription(Float64, 'velocity_y',
                                 lambda msg: self._handle('velocity_y', msg.data), qos)
        self.create_subscription(Float64, 'control_force_x',
                                 lambda msg: self._handle('control_force_x', msg.data), qos)
        self.create_subscription(Float64, 'control_force_y',
                                 lambda msg: self._handle('control_force_y', msg.data), qos)

        self.get_logger().info('Data Listener Node has been started and is listening for data.')

    # ----------------------------------------------------------------------
    # Common message handler
    # ----------------------------------------------------------------------
    def _handle(self, topic: str, value: float):
        timestamp = datetime.now().isoformat(sep=' ', timespec='milliseconds')
        # Write to CSV then flush to avoid data loss on abrupt shutdown
        self.csv_writer.writerow([timestamp, topic, f'{value:.6f}'])
        self.csv_file.flush()
        # Console output
        self.get_logger().info(f'{topic}: {value:.4f}')

    # ----------------------------------------------------------------------
    # Clean shutdown – ensures the CSV file is closed properly
    # ----------------------------------------------------------------------
    def destroy_node(self):
        if not self.csv_file.closed:
            self.csv_file.close()
            self.get_logger().info('CSV log file closed.')
        super().destroy_node()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DataListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('Shutting down Data Listener Node.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
