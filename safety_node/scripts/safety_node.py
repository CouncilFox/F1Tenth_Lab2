#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class SafetyNode(Node):
    """
    Implements a safety mechanism for autonomous vehicles by calculating the Instantaneous Time to Collision (iTTC)
    with obstacles detected via LIDAR scans. If iTTC falls below a predefined threshold, the node issues an emergency
    braking command to prevent a collision.
    """
    def __init__(self):
        super().__init__('safety_node')
        # Publisher for emergency braking commands
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        
        # Subscribers to odometry and LIDAR scan data
        self.subscription_odom = self.create_subscription(Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.subscription_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        # Vehicle's current speed, initialized to 0
        self.speed = 0.0
        
        # For throttling the logging output
        self.last_log_time = time.time()
        self.log_interval = 1.0  # Log at most once per second

    def odom_callback(self, odom_msg):
        """
        Callback for processing odometry messages. Updates the vehicle's current speed.
        
        Parameters:
            odom_msg (Odometry): The incoming odometry message.
        """
        self.speed = odom_msg.twist.twist.linear.x
        # Throttle logging to avoid flooding the console
        current_time = time.time()
        if current_time - self.last_log_time >= self.log_interval:
            self.get_logger().info(f'Current Speed: {self.speed:.2f} m/s')
            self.last_log_time = current_time

    def scan_callback(self, scan_msg):
        """
        Callback for processing LIDAR scan messages. Calculates iTTC for each scan point and issues a braking
        command if a potential collision is detected.
        
        Parameters:
            scan_msg (LaserScan): The incoming LIDAR scan message.
        """
        angle_increment = scan_msg.angle_increment
        current_velocity = self.speed
        ittc_array = []

        for i, range_measurement in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * angle_increment
            range_rate = current_velocity * np.cos(angle)

            if range_rate > 0:
                ittc = range_measurement / range_rate
            else:
                ittc = np.inf  # Treat as no collision threat
            
            ittc_array.append(ittc)

        # Issue a brake command if any iTTC value falls below the safety threshold
        if any(ittc < 2 for ittc in ittc_array):  # Threshold of 2 seconds
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0  # Command to stop the vehicle
            self.publisher_.publish(drive_msg)
            self.get_logger().info('Emergency braking initiated!')

def main(args=None):
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)
    safety_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
