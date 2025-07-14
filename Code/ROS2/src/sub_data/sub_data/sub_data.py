import csv
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64


class DataListenerNode(Node):
    """ROS 2 node that subscribes to six Float64 topics and logs each
    message both to the console and to a rotating CSV file.

    Fixes / improvements over the previous version:
    ▸ Corrected quoting bug in f‑string used for the filename.
    ▸ Guaranteed file close via rclpy.on_shutdown as well as the finally block.
    ▸ Parameterised log directory ( --ros-args -p log_dir:=... ).
    ▸ Opens the CSV in *append* mode, so re‑running the node won’t wipe earlier data.
    ▸ Keeps a header row only once per new file.
    """

    def __init__(self):
        super().__init__('data_listener_node')

        qos = 10  # match publisher depth

        # ------------------------------------------------------------------
        # CSV setup
        # ------------------------------------------------------------------
        default_log_dir = Path.home() / 'ros2_logs'
        self.declare_parameter('log_dir', str(default_log_dir))
        log_dir = Path(self.get_parameter('log_dir').get_parameter_value().string_value)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Time‑stamped file name (quotes fixed!)
        filename = log_dir / f"data_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        file_exists = filename.exists()
        self.csv_file = open(filename, 'a', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        if not file_exists:  # add header only for new files
            self.csv_writer.writerow(['timestamp', 'topic', 'value'])

        self.get_logger().info(f'Logging incoming data to {filename}')

        # ------------------------------------------------------------------
        # Subscriptions (use late‑binding safe default arg in lambda)
        # ------------------------------------------------------------------
        topics = ['position_x', 'position_y',
                  'velocity_x', 'velocity_y',
                  'control_force_x', 'control_force_y']

        for t in topics:
            self.create_subscription(Float64, t, lambda msg, t=t: self._handle(t, msg.data), qos)

        self.get_logger().info('Data Listener Node has been started and is listening for data.')

        # Ensure graceful shutdown
        rclpy.get_default_context().on_shutdown(self._on_shutdown)

    # ----------------------------------------------------------------------
    # Common message handler
    # ----------------------------------------------------------------------
    def _handle(self, topic: str, value: float):
        timestamp = datetime.now().isoformat(sep=' ', timespec='milliseconds')
        self.csv_writer.writerow([timestamp, topic, f'{value:.6f}'])
        self.csv_file.flush()
        self.get_logger().info(f'{topic}: {value:.4f}')

    # ----------------------------------------------------------------------
    # Shutdown housekeeping
    # ----------------------------------------------------------------------
    def _on_shutdown(self):
        if not self.csv_file.closed:
            self.csv_file.close()
            self.get_logger().info('CSV log file closed.')


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)
    node = DataListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('SIGINT received – shutting down.')
    finally:
        node._on_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
