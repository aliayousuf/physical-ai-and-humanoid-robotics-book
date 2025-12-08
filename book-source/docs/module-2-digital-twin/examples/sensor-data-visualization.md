---
title: "Sensor Data Visualization Example"
description: "Visualizing sensor data from robot simulation in various formats"
---

# Sensor Data Visualization Example

## Overview

Effective visualization of sensor data is crucial for understanding robot perception and debugging robotic systems. This example demonstrates various approaches to visualize different types of sensor data from robot simulation, including camera feeds, LiDAR scans, IMU data, and fused sensor information.

## Prerequisites

Before working with sensor data visualization, ensure you have:
- Working robot simulation with sensors
- Basic understanding of ROS message types for sensors
- Knowledge of visualization tools (RViz2, matplotlib, Unity)
- Understanding of coordinate systems and transformations

## Camera Data Visualization

### 1. Basic Camera Feed Display

The simplest form of camera visualization is displaying the raw image feed:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraVisualizer(Node):
    def __init__(self):
        super().__init__('camera_visualizer')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Create subscriber for camera data
        self.camera_sub = self.create_subscription(
            Image,
            '/head_camera/image_raw',
            self.camera_callback,
            10
        )

        # Window name for display
        self.window_name = 'Camera Feed'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.get_logger().info('Camera visualizer initialized')

    def camera_callback(self, msg):
        """Process and display camera image"""
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Optional: Add overlays or annotations
            height, width = cv_image.shape[:2]

            # Draw center crosshair
            center_x, center_y = width // 2, height // 2
            cv2.line(cv_image, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
            cv2.line(cv_image, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)

            # Add timestamp
            timestamp = f'Time: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}'
            cv2.putText(cv_image, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Display the image
            cv2.imshow(self.window_name, cv_image)
            cv2.waitKey(1)  # Process GUI events

        except Exception as e:
            self.get_logger().error(f'Error processing camera image: {e}')

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    visualizer = CameraVisualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        visualizer.get_logger().info('Shutting down camera visualizer')
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. Advanced Camera Processing and Visualization

For more sophisticated camera visualization with feature detection:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Point

class AdvancedCameraVisualizer(Node):
    def __init__(self):
        super().__init__('advanced_camera_visualizer')

        self.bridge = CvBridge()

        # Create subscriber
        self.camera_sub = self.create_subscription(
            Image,
            '/head_camera/image_raw',
            self.advanced_camera_callback,
            10
        )

        # Create publisher for processed image
        self.processed_pub = self.create_publisher(
            Image,
            '/camera/processed_image',
            10
        )

        # Create publisher for detected features
        self.features_pub = self.create_publisher(
            Point,
            '/camera/features',
            10
        )

        self.window_name = 'Advanced Camera Processing'
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # For optical flow tracking
        self.old_gray = None
        self.p0 = None

        self.get_logger().info('Advanced camera visualizer initialized')

    def advanced_camera_callback(self, msg):
        """Process and visualize advanced camera features"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect corners for feature tracking
            corners = cv2.goodFeaturesToTrack(
                gray,
                mask=None,
                **self.feature_params
            )

            # Process corners if detected
            if corners is not None:
                corners = np.int0(corners)

                # Draw circles around detected features
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(cv_image, (x, y), 5, (0, 255, 0), -1)

                    # Publish feature location
                    feature_point = Point()
                    feature_point.x = float(x)
                    feature_point.y = float(y)
                    feature_point.z = 0.0  # Depth would come from stereo/depth camera
                    self.features_pub.publish(feature_point)

            # Add various visualizations
            height, width = cv_image.shape[:2]

            # Draw grid for reference
            grid_size = 50
            for i in range(0, width, grid_size):
                cv2.line(cv_image, (i, 0), (i, height), (100, 100, 100), 1)
            for i in range(0, height, grid_size):
                cv2.line(cv_image, (0, i), (width, i), (100, 100, 100), 1)

            # Add statistics
            avg_color = np.mean(cv_image, axis=(0, 1))
            stats_text = f'Avg RGB: ({avg_color[2]:.1f}, {avg_color[1]:.1f}, {avg_color[0]:.1f})'
            cv2.putText(cv_image, stats_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Convert back to ROS message and publish
            processed_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            processed_msg.header = msg.header
            self.processed_pub.publish(processed_msg)

            # Display the processed image
            cv2.imshow(self.window_name, cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in advanced camera processing: {e}')

def main(args=None):
    rclpy.init(args=args)
    visualizer = AdvancedCameraVisualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        visualizer.get_logger().info('Shutting down advanced camera visualizer')
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## LiDAR Data Visualization

### 1. Basic LiDAR Scan Visualization

Visualize 2D LiDAR scan data in a polar format:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np
import threading
import time

class LidarVisualizer(Node):
    def __init__(self):
        super().__init__('lidar_visualizer')

        # Create subscriber
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        # Data storage
        self.scan_ranges = None
        self.scan_angles = None
        self.scan_time = None

        # Matplotlib setup
        plt.ion()  # Interactive mode
        self.fig, (self.ax_polar, self.ax_cartesian) = plt.subplots(1, 2, figsize=(12, 6))

        # Polar plot
        self.theta = None
        self.ranges_plot = None

        # Cartesian plot
        self.x_coords = None
        self.y_coords = None
        self.cartesian_plot = None

        # Thread for updating plots
        self.plot_thread = threading.Thread(target=self.update_plots, daemon=True)
        self.plot_thread.start()

        self.get_logger().info('LiDAR visualizer initialized')

    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        try:
            # Convert ranges to numpy array
            ranges = np.array(msg.ranges)

            # Create angle array
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            angle_increment = msg.angle_increment

            angles = np.arange(angle_min, angle_max, angle_increment)
            angles = angles[:len(ranges)]  # Ensure same length as ranges

            # Filter out invalid ranges
            valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
            self.scan_ranges = ranges[valid_mask]
            self.scan_angles = angles[valid_mask]
            self.scan_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR data: {e}')

    def update_plots(self):
        """Continuously update plots in separate thread"""
        while rclpy.ok():
            if self.scan_ranges is not None and self.scan_angles is not None:
                try:
                    # Update polar plot
                    self.ax_polar.clear()
                    self.ax_polar.scatter(self.scan_angles, self.scan_ranges, s=1, alpha=0.7)
                    self.ax_polar.set_theta_zero_location('N')  # Zero degrees at top
                    self.ax_polar.set_theta_direction(-1)  # Clockwise
                    self.ax_polar.set_title('LiDAR Scan (Polar)')
                    self.ax_polar.grid(True)

                    # Update cartesian plot
                    x_coords = self.scan_ranges * np.cos(self.scan_angles)
                    y_coords = self.scan_ranges * np.sin(self.scan_angles)

                    self.ax_cartesian.clear()
                    self.ax_cartesian.scatter(x_coords, y_coords, s=1, alpha=0.7)
                    self.ax_cartesian.set_aspect('equal')
                    self.ax_cartesian.grid(True)
                    self.ax_cartesian.set_title('LiDAR Scan (Cartesian)')
                    self.ax_cartesian.set_xlabel('X (m)')
                    self.ax_cartesian.set_ylabel('Y (m)')

                    # Add robot position marker
                    self.ax_cartesian.plot(0, 0, 'ro', markersize=10, label='Robot')
                    self.ax_cartesian.legend()

                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.001)  # Small pause to update

                except Exception as e:
                    self.get_logger().error(f'Error updating plots: {e}')

            time.sleep(0.1)  # Update rate

def main(args=None):
    rclpy.init(args=args)
    visualizer = LidarVisualizer()

    try:
        # Run ROS spinning in a separate thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(visualizer,), daemon=True)
        spin_thread.start()

        # Keep the main thread alive for plotting
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            visualizer.get_logger().info('Shutting down LiDAR visualizer')
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### 2. 3D Point Cloud Visualization

For 3D LiDAR or depth camera data:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np
import struct
from sensor_msgs_py import point_cloud2

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available. 3D visualization will be limited.")

class PointCloudVisualizer(Node):
    def __init__(self):
        super().__init__('pointcloud_visualizer')

        # Create subscriber
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',  # Example topic
            self.pointcloud_callback,
            10
        )

        # Data storage
        self.pointcloud_data = None

        # Visualization options
        self.use_open3d = OPEN3D_AVAILABLE
        if self.use_open3d:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name='Point Cloud', width=800, height=600)
            self.pcd = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.pcd)

        self.get_logger().info('Point cloud visualizer initialized')

    def pointcloud_callback(self, msg):
        """Process point cloud data"""
        try:
            # Extract point cloud data
            points_list = []

            for point in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                points_list.append([point[0], point[1], point[2]])

            if points_list:
                self.pointcloud_data = np.array(points_list)

                if self.use_open3d:
                    # Update Open3D visualization
                    self.pcd.points = o3d.utility.Vector3dVector(self.pointcloud_data)
                    self.vis.update_geometry(self.pcd)
                    self.vis.poll_events()
                    self.vis.update_renderer()

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def destroy_node(self):
        """Cleanup visualization resources"""
        if self.use_open3d:
            self.vis.destroy_window()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    visualizer = PointCloudVisualizer()

    try:
        rclpy.spin(visualizer)
    except KeyboardInterrupt:
        visualizer.get_logger().info('Shutting down point cloud visualizer')
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## IMU Data Visualization

### 1. Orientation Visualization

Visualize IMU orientation data in 3D:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import threading
import time

class IMUVisualizer(Node):
    def __init__(self):
        super().__init__('imu_visualizer')

        # Create subscriber
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Data storage
        self.orientation = None
        self.angular_velocity = None
        self.linear_acceleration = None

        # Matplotlib setup
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))

        # 3D orientation visualization
        self.ax_orientation = self.fig.add_subplot(2, 2, 1, projection='3d')
        self.ax_orientation.set_title('Orientation (3D)')

        # Euler angles over time
        self.ax_euler = self.fig.add_subplot(2, 2, 2)
        self.ax_euler.set_title('Euler Angles')

        # Angular velocity
        self.ax_angular = self.fig.add_subplot(2, 2, 3)
        self.ax_angular.set_title('Angular Velocity')

        # Linear acceleration
        self.ax_acceleration = self.fig.add_subplot(2, 2, 4)
        self.ax_acceleration.set_title('Linear Acceleration')

        # Data buffers for time series
        self.time_buffer = []
        self.roll_buffer = []
        self.pitch_buffer = []
        self.yaw_buffer = []

        # Start update thread
        self.plot_thread = threading.Thread(target=self.update_plots, daemon=True)
        self.plot_thread.start()

        self.get_logger().info('IMU visualizer initialized')

    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            # Extract orientation (convert from quaternion to Euler angles)
            q = msg.orientation
            self.orientation = self.quaternion_to_euler(q.x, q.y, q.z, q.w)

            # Extract angular velocity
            self.angular_velocity = (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            )

            # Extract linear acceleration
            self.linear_acceleration = (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            )

            # Update time series data
            current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            if self.orientation:
                self.time_buffer.append(current_time)
                self.roll_buffer.append(self.orientation[0])
                self.pitch_buffer.append(self.orientation[1])
                self.yaw_buffer.append(self.orientation[2])

                # Keep only recent data (last 100 points)
                if len(self.time_buffer) > 100:
                    self.time_buffer.pop(0)
                    self.roll_buffer.pop(0)
                    self.pitch_buffer.pop(0)
                    self.yaw_buffer.pop(0)

        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return (roll, pitch, yaw)

    def update_plots(self):
        """Update plots continuously"""
        while rclpy.ok():
            try:
                # Clear plots
                self.ax_orientation.clear()
                self.ax_euler.clear()
                self.ax_angular.clear()
                self.ax_acceleration.clear()

                # Update 3D orientation if data available
                if self.orientation:
                    roll, pitch, yaw = self.orientation

                    # Create a simple 3D representation of orientation
                    # Draw coordinate axes
                    origin = np.array([0, 0, 0])
                    scale = 0.5

                    # X-axis (red - forward)
                    x_axis = np.array([scale * np.cos(yaw) * np.cos(pitch),
                                       scale * np.sin(yaw) * np.cos(pitch),
                                       -scale * np.sin(pitch)])

                    # Y-axis (green - left)
                    y_axis = np.array([scale * (-np.sin(yaw)),
                                       scale * np.cos(yaw),
                                       0])

                    # Z-axis (blue - up)
                    z_axis = np.array([scale * np.cos(yaw) * np.sin(pitch),
                                       scale * np.sin(yaw) * np.sin(pitch),
                                       scale * np.cos(pitch)])

                    self.ax_orientation.quiver(*origin, *x_axis, color='red', arrow_length_ratio=0.1)
                    self.ax_orientation.quiver(*origin, *y_axis, color='green', arrow_length_ratio=0.1)
                    self.ax_orientation.quiver(*origin, *z_axis, color='blue', arrow_length_ratio=0.1)

                    self.ax_orientation.set_xlim([-1, 1])
                    self.ax_orientation.set_ylim([-1, 1])
                    self.ax_orientation.set_zlim([-1, 1])
                    self.ax_orientation.set_xlabel('X')
                    self.ax_orientation.set_ylabel('Y')
                    self.ax_orientation.set_zlabel('Z')

                # Update Euler angles plot
                if self.time_buffer and self.roll_buffer:
                    times = np.array(self.time_buffer) - self.time_buffer[0]  # Relative time
                    self.ax_euler.plot(times, np.degrees(self.roll_buffer), label='Roll', color='red')
                    self.ax_euler.plot(times, np.degrees(self.pitch_buffer), label='Pitch', color='green')
                    self.ax_euler.plot(times, np.degrees(self.yaw_buffer), label='Yaw', color='blue')
                    self.ax_euler.set_xlabel('Time (s)')
                    self.ax_euler.set_ylabel('Angle (degrees)')
                    self.ax_euler.legend()
                    self.ax_euler.grid(True)

                # Update angular velocity plot
                if self.angular_velocity:
                    ax_labels = ['X', 'Y', 'Z']
                    ang_vel = self.angular_velocity
                    self.ax_angular.bar(ax_labels, [ang_vel[0], ang_vel[1], ang_vel[2]],
                                      color=['red', 'green', 'blue'])
                    self.ax_angular.set_ylabel('Angular Velocity (rad/s)')
                    self.ax_angular.set_title('Angular Velocity')

                # Update linear acceleration plot
                if self.linear_acceleration:
                    acc_labels = ['X', 'Y', 'Z']
                    acc = self.linear_acceleration
                    self.ax_acceleration.bar(acc_labels, [acc[0], acc[1], acc[2]],
                                           color=['red', 'green', 'blue'])
                    self.ax_acceleration.set_ylabel('Acceleration (m/s²)')
                    self.ax_acceleration.set_title('Linear Acceleration')

                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)

            except Exception as e:
                self.get_logger().error(f'Error updating IMU plots: {e}')

            time.sleep(0.05)  # Update rate

def main(args=None):
    rclpy.init(args=args)
    visualizer = IMUVisualizer()

    try:
        # Run ROS spinning in a separate thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(visualizer,), daemon=True)
        spin_thread.start()

        # Keep the main thread alive for plotting
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            visualizer.get_logger().info('Shutting down IMU visualizer')
    finally:
        visualizer.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-Sensor Fusion Visualization

### 1. Sensor Fusion Dashboard

Create a comprehensive dashboard that combines multiple sensor types:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import threading
import time

class MultiSensorDashboard(Node):
    def __init__(self):
        super().__init__('multi_sensor_dashboard')

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Create subscribers for all sensor types
        self.camera_sub = self.create_subscription(
            Image,
            '/head_camera/image_raw',
            self.camera_callback,
            10
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            10
        )

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Data storage
        self.latest_camera = None
        self.latest_lidar_ranges = None
        self.latest_lidar_angles = None
        self.latest_orientation = None
        self.latest_angular_vel = None
        self.latest_linear_acc = None

        # Matplotlib dashboard setup
        plt.ion()
        self.fig = plt.figure(figsize=(16, 10))

        # Camera feed (top left)
        self.ax_camera = self.fig.add_subplot(2, 3, 1)
        self.ax_camera.set_title('Camera Feed')
        self.ax_camera.axis('off')
        self.camera_image_display = None

        # LiDAR scan (top center)
        self.ax_lidar = self.fig.add_subplot(2, 3, 2, projection='polar')
        self.ax_lidar.set_title('LiDAR Scan')

        # 3D orientation (top right)
        self.ax_orientation = self.fig.add_subplot(2, 3, 3, projection='3d')
        self.ax_orientation.set_title('Orientation')

        # Angular velocity (bottom left)
        self.ax_ang_vel = self.fig.add_subplot(2, 3, 4)
        self.ax_ang_vel.set_title('Angular Velocity')

        # Linear acceleration (bottom center)
        self.ax_lin_acc = self.fig.add_subplot(2, 3, 5)
        self.ax_lin_acc.set_title('Linear Acceleration')

        # Status panel (bottom right)
        self.ax_status = self.fig.add_subplot(2, 3, 6)
        self.ax_status.set_title('System Status')
        self.ax_status.axis('off')

        # Start update thread
        self.dashboard_thread = threading.Thread(target=self.update_dashboard, daemon=True)
        self.dashboard_thread.start()

        self.get_logger().info('Multi-sensor dashboard initialized')

    def camera_callback(self, msg):
        """Process camera data"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_camera = cv_image
        except Exception as e:
            self.get_logger().error(f'Error processing camera: {e}')

    def lidar_callback(self, msg):
        """Process LiDAR data"""
        try:
            ranges = np.array(msg.ranges)
            angle_min = msg.angle_min
            angle_max = msg.angle_max
            angle_increment = msg.angle_increment

            angles = np.arange(angle_min, angle_max, angle_increment)
            angles = angles[:len(ranges)]

            # Filter valid ranges
            valid_mask = (ranges > msg.range_min) & (ranges < msg.range_max)
            self.latest_lidar_ranges = ranges[valid_mask]
            self.latest_lidar_angles = angles[valid_mask]
        except Exception as e:
            self.get_logger().error(f'Error processing LiDAR: {e}')

    def imu_callback(self, msg):
        """Process IMU data"""
        try:
            # Extract orientation
            q = msg.orientation
            self.latest_orientation = self.quaternion_to_euler(q.x, q.y, q.z, q.w)

            # Extract angular velocity
            self.latest_angular_vel = (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            )

            # Extract linear acceleration
            self.latest_linear_acc = (
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            )
        except Exception as e:
            self.get_logger().error(f'Error processing IMU: {e}')

    def quaternion_to_euler(self, x, y, z, w):
        """Convert quaternion to Euler angles"""
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return (roll, pitch, yaw)

    def update_dashboard(self):
        """Update the entire dashboard"""
        while rclpy.ok():
            try:
                # Clear all subplots
                self.ax_camera.clear()
                self.ax_lidar.clear()
                self.ax_orientation.clear()
                self.ax_ang_vel.clear()
                self.ax_lin_acc.clear()
                self.ax_status.clear()

                # Update camera display
                if self.latest_camera is not None:
                    self.ax_camera.imshow(self.latest_camera[:, :, ::-1])  # Convert BGR to RGB
                    self.ax_camera.axis('off')
                    self.ax_camera.set_title('Camera Feed')
                else:
                    self.ax_camera.text(0.5, 0.5, 'No Camera Data',
                                       horizontalalignment='center',
                                       verticalalignment='center',
                                       transform=self.ax_camera.transAxes)
                    self.ax_camera.axis('off')

                # Update LiDAR display
                if self.latest_lidar_ranges is not None and self.latest_lidar_angles is not None:
                    self.ax_lidar.scatter(self.latest_lidar_angles, self.latest_lidar_ranges, s=1, alpha=0.7)
                    self.ax_lidar.set_theta_zero_location('N')
                    self.ax_lidar.set_theta_direction(-1)
                    self.ax_lidar.set_title('LiDAR Scan')
                else:
                    self.ax_lidar.text(0, 0, 'No LiDAR Data',
                                      horizontalalignment='center',
                                      verticalalignment='center')
                    self.ax_lidar.set_title('LiDAR Scan')

                # Update orientation display
                if self.latest_orientation is not None:
                    roll, pitch, yaw = self.latest_orientation

                    # Draw simple coordinate system
                    origin = np.array([0, 0, 0])
                    scale = 0.5

                    # Calculate rotated axes
                    x_axis = np.array([scale * np.cos(yaw) * np.cos(pitch),
                                       scale * np.sin(yaw) * np.cos(pitch),
                                       -scale * np.sin(pitch)])

                    y_axis = np.array([scale * (-np.sin(yaw)),
                                       scale * np.cos(yaw),
                                       0])

                    z_axis = np.array([scale * np.cos(yaw) * np.sin(pitch),
                                       scale * np.sin(yaw) * np.sin(pitch),
                                       scale * np.cos(pitch)])

                    self.ax_orientation.quiver(*origin, *x_axis, color='red', arrow_length_ratio=0.1)
                    self.ax_orientation.quiver(*origin, *y_axis, color='green', arrow_length_ratio=0.1)
                    self.ax_orientation.quiver(*origin, *z_axis, color='blue', arrow_length_ratio=0.1)

                    self.ax_orientation.set_xlim([-1, 1])
                    self.ax_orientation.set_ylim([-1, 1])
                    self.ax_orientation.set_zlim([-1, 1])
                    self.ax_orientation.set_xlabel('X')
                    self.ax_orientation.set_ylabel('Y')
                    self.ax_orientation.set_zlabel('Z')
                    self.ax_orientation.set_title('Orientation')
                else:
                    self.ax_orientation.text(0, 0, 0, 'No IMU Data',
                                            horizontalalignment='center',
                                            verticalalignment='center')
                    self.ax_orientation.set_title('Orientation')

                # Update angular velocity
                if self.latest_angular_vel is not None:
                    ax_labels = ['X', 'Y', 'Z']
                    ang_vel = self.latest_angular_vel
                    bars = self.ax_ang_vel.bar(ax_labels, [ang_vel[0], ang_vel[1], ang_vel[2]],
                                              color=['red', 'green', 'blue'])
                    self.ax_ang_vel.set_ylabel('Angular Velocity (rad/s)')
                    self.ax_ang_vel.set_title('Angular Velocity')

                    # Add value labels on bars
                    for bar, val in zip(bars, ang_vel):
                        height = bar.get_height()
                        self.ax_ang_vel.text(bar.get_x() + bar.get_width()/2., height,
                                            f'{val:.2f}',
                                            ha='center', va='bottom')
                else:
                    self.ax_ang_vel.text(0.5, 0.5, 'No IMU Data',
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        transform=self.ax_ang_vel.transAxes)
                    self.ax_ang_vel.set_title('Angular Velocity')
                    self.ax_ang_vel.axis('off')

                # Update linear acceleration
                if self.latest_linear_acc is not None:
                    acc_labels = ['X', 'Y', 'Z']
                    acc = self.latest_linear_acc
                    bars = self.ax_lin_acc.bar(acc_labels, [acc[0], acc[1], acc[2]],
                                              color=['red', 'green', 'blue'])
                    self.ax_lin_acc.set_ylabel('Acceleration (m/s²)')
                    self.ax_lin_acc.set_title('Linear Acceleration')

                    # Add value labels on bars
                    for bar, val in zip(bars, acc):
                        height = bar.get_height()
                        self.ax_lin_acc.text(bar.get_x() + bar.get_width()/2., height,
                                            f'{val:.2f}',
                                            ha='center', va='bottom')
                else:
                    self.ax_lin_acc.text(0.5, 0.5, 'No IMU Data',
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        transform=self.ax_lin_acc.transAxes)
                    self.ax_lin_acc.set_title('Linear Acceleration')
                    self.ax_lin_acc.axis('off')

                # Update status panel
                self.ax_status.axis('off')
                status_text = "Sensor Status:\n"
                status_text += f"• Camera: {'Active' if self.latest_camera is not None else 'Inactive'}\n"
                status_text += f"• LiDAR: {'Active' if self.latest_lidar_ranges is not None else 'Inactive'}\n"
                status_text += f"• IMU: {'Active' if self.latest_orientation is not None else 'Inactive'}\n"

                if self.latest_orientation is not None:
                    roll, pitch, yaw = self.latest_orientation
                    status_text += f"\nOrientation:\n"
                    status_text += f"• Roll: {np.degrees(roll):.1f}°\n"
                    status_text += f"• Pitch: {np.degrees(pitch):.1f}°\n"
                    status_text += f"• Yaw: {np.degrees(yaw):.1f}°\n"

                self.ax_status.text(0.05, 0.95, status_text,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)

            except Exception as e:
                self.get_logger().error(f'Error updating dashboard: {e}')

            time.sleep(0.05)  # Update rate

def main(args=None):
    rclpy.init(args=args)
    dashboard = MultiSensorDashboard()

    try:
        # Run ROS spinning in a separate thread
        spin_thread = threading.Thread(target=rclpy.spin, args=(dashboard,), daemon=True)
        spin_thread.start()

        # Keep the main thread alive for dashboard
        try:
            plt.show(block=True)
        except KeyboardInterrupt:
            dashboard.get_logger().info('Shutting down multi-sensor dashboard')
    finally:
        dashboard.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## RViz2 Configuration for Sensor Visualization

Create an RViz2 configuration file to visualize sensors effectively:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /Grid1
        - /RobotModel1
        - /TF1
        - /LaserScan1
        - /Image1
        - /PointCloud21
        - /Imu1
      Splitter Ratio: 0.5
    Tree Height: 617
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 0
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: LaserScan
      Position Transformer: XYZ
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Median window: 5
      Min Value: 0
      Name: Image
      Normalize Range: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /head_camera/image_raw
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/PointCloud2
      Color: 255; 255; 255
      Color Transformer: RGB8
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: PointCloud2
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /camera/depth/color/points
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/Imu
      Enabled: true
      Name: Imu
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /imu/data
      Value: true
      fixed_frame_orientation: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /clicked_point
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 10
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1025
  Hide Left Dock: false
  Hide Right Dock: false
  Image:
    collapsed: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000003a7fc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d0000029f000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000003a7fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003d000003a7000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000004420000003efc0100000002fb0000000800540069006d00650100000000000004420000000000000000fb0000000800540069006d00650100000000000004500000000000000000000003a3000003a700000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1853
  X: 67
  Y: 27
```

## Unity Sensor Visualization

For Unity-based sensor visualization:

```csharp
using UnityEngine;
using UnityEngine.UI;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Sensor_msgs;
using System.Collections.Generic;

public class UnitySensorVisualizer : MonoBehaviour
{
    [Header("Sensor Topics")]
    public string cameraTopic = "/head_camera/image_raw";
    public string lidarTopic = "/scan";
    public string imuTopic = "/imu/data";

    [Header("Visualization Elements")]
    public RawImage cameraDisplay;
    public LineRenderer lidarRenderer;
    public Text orientationText;
    public GameObject robotModel;

    [Header("Visualization Settings")]
    public int maxLidarPoints = 360;
    public float lidarScale = 10f;
    public float cameraAspectRatio = 1.33f; // 4:3 aspect ratio

    private RosSocket rosSocket;
    private Texture2D cameraTexture;
    private float[] lidarRanges;
    private Vector3 imuOrientation;

    void Start()
    {
        ConnectToROS();
        InitializeVisualization();
    }

    void ConnectToROS()
    {
        try
        {
            rosSocket = new RosSocket(new WebSocketSharpClient("ws://localhost:9090"));
            Debug.Log("Connected to ROS Bridge");

            // Subscribe to sensor topics
            rosSocket.Subscribe<Image>(cameraTopic, ProcessCameraData);
            rosSocket.Subscribe<LaserScan>(lidarTopic, ProcessLidarData);
            rosSocket.Subscribe<Imu>(imuTopic, ProcessImuData);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to ROS: {e.Message}");
        }
    }

    void InitializeVisualization()
    {
        // Initialize lidar renderer
        if (lidarRenderer != null)
        {
            lidarRenderer.positionCount = maxLidarPoints;
            lidarRenderer.startWidth = 0.05f;
            lidarRenderer.endWidth = 0.05f;
        }

        // Initialize camera texture
        if (cameraDisplay != null)
        {
            cameraTexture = new Texture2D(640, 480);
            cameraDisplay.texture = cameraTexture;
        }
    }

    void ProcessCameraData(Image imageMsg)
    {
        // Convert ROS Image message to Unity Texture2D
        if (imageMsg.data.Length > 0)
        {
            // This is a simplified example - in practice you'd need to properly decode the image format
            // For demonstration, we'll just create a dummy texture
            if (cameraTexture.width != imageMsg.width || cameraTexture.height != imageMsg.height)
            {
                cameraTexture = new Texture2D((int)imageMsg.width, (int)imageMsg.height);
                cameraDisplay.texture = cameraTexture;
            }

            // Update camera display aspect ratio
            if (cameraDisplay.GetComponent<AspectRatioFitter>() != null)
            {
                var fitter = cameraDisplay.GetComponent<AspectRatioFitter>();
                fitter.aspectRatio = (float)imageMsg.width / (float)imageMsg.height;
            }
        }
    }

    void ProcessLidarData(LaserScan scanMsg)
    {
        lidarRanges = new float[scanMsg.ranges.Length];
        for (int i = 0; i < scanMsg.ranges.Length; i++)
        {
            lidarRanges[i] = (float)scanMsg.ranges[i];
        }

        UpdateLidarVisualization();
    }

    void ProcessImuData(Imu imuMsg)
    {
        // Convert quaternion to Euler angles
        Quaternion quat = new Quaternion(
            (float)imuMsg.orientation.x,
            (float)imuMsg.orientation.y,
            (float)imuMsg.orientation.z,
            (float)imuMsg.orientation.w
        );

        imuOrientation = quat.eulerAngles;

        // Update robot model orientation
        if (robotModel != null)
        {
            robotModel.transform.rotation = quat;
        }

        // Update orientation text display
        if (orientationText != null)
        {
            orientationText.text = $"Roll: {imuOrientation.x:F2}\nPitch: {imuOrientation.y:F2}\nYaw: {imuOrientation.z:F2}";
        }
    }

    void UpdateLidarVisualization()
    {
        if (lidarRenderer == null || lidarRanges == null) return;

        Vector3[] positions = new Vector3[lidarRanges.Length];

        for (int i = 0; i < lidarRanges.Length; i++)
        {
            float angle = (float)i / lidarRanges.Length * 2 * Mathf.PI; // Full circle
            float distance = lidarRanges[i] * lidarScale; // Scale for visibility

            if (distance > 0 && distance < scanMsg.range_max * lidarScale) // Valid range
            {
                positions[i] = new Vector3(
                    distance * Mathf.Cos(angle),
                    0,
                    distance * Mathf.Sin(angle)
                );
            }
            else
            {
                positions[i] = Vector3.zero; // Invalid readings at origin
            }
        }

        lidarRenderer.positionCount = positions.Length;
        lidarRenderer.SetPositions(positions);
    }

    void OnDestroy()
    {
        rosSocket?.Close();
    }
}
```

## Best Practices for Sensor Data Visualization

### 1. Performance Optimization
- Use appropriate update rates for each sensor type
- Implement data decimation for high-frequency sensors
- Use efficient rendering techniques for large datasets
- Cache processed data when possible

### 2. Data Validation
- Check for NaN and infinity values
- Validate sensor ranges and bounds
- Implement outlier detection and filtering
- Monitor sensor health and status

### 3. User Experience
- Provide intuitive color schemes and visual cues
- Include reference frames and coordinate systems
- Add interactive controls for exploration
- Implement multiple visualization modes

### 4. Debugging Features
- Overlay sensor data on camera feeds
- Provide coordinate frame visualization
- Include sensor status and health indicators
- Enable data recording and playback

## Learning Objectives

After completing this example, you should understand:
- How to visualize different types of sensor data (camera, LiDAR, IMU)
- Techniques for combining multiple sensor data streams
- Best practices for real-time sensor data visualization
- How to create effective dashboards for sensor monitoring
- Approaches for visualizing sensor data in different environments (matplotlib, RViz2, Unity)

## Next Steps

Continue to learn about [Unity Robot Control](../examples/unity-robot-control) to understand how to implement robot control interfaces in Unity.