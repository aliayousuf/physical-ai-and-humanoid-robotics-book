---
title: "Perception Pipeline Setup"
description: "Setting up perception pipelines using Isaac Sim and Isaac ROS packages"
---

# Perception Pipeline Setup

## Overview

This workflow guide covers setting up perception pipelines using NVIDIA Isaac Sim and Isaac ROS packages. Perception pipelines are critical for enabling robots to understand their environment through sensors like cameras, LiDAR, and IMUs. We'll cover how to integrate Isaac's GPU-accelerated perception packages with realistic simulation data.

## Prerequisites

Before setting up perception pipelines, ensure you have:
- Completed Module 1 (ROS 2 fundamentals)
- Completed Module 2 (Digital Twin simulation)
- Understanding of computer vision concepts
- Basic knowledge of sensor data processing
- Isaac Sim and Isaac ROS installed

## Perception Pipeline Components

### Isaac ROS Packages

The Isaac ROS package collection provides GPU-accelerated perception capabilities:

- **Apriltag**: GPU-accelerated AprilTag detection
- **Visual SLAM**: Hardware-accelerated VSLAM
- **Stereo Dense Reconstruction**: 3D reconstruction from stereo cameras
- **Bi3D**: Instance segmentation for bin picking
- **Depth Segmentation**: Semantic segmentation for depth images
- **OAK**: Intel RealSense and OAK-D camera integration

### Architecture Overview

```
Sensor Data → Isaac ROS Node → GPU Processing → Perception Results
     ↓              ↓                ↓              ↓
  Camera/LiDAR   Hardware     Accelerated    ROS Messages
                 Acceleration   Processing     (Image, PointCloud, etc.)
```

## Setting Up the Environment

### 1. Install Isaac ROS Packages

```bash
# Add NVIDIA ROS repository
curl -sSL https://repos.mapotempo.com/repos-stable.key | sudo apt-key add -
sudo add-apt-repository "deb https://repos.mapotempo.com/apt/ubuntu $(lsb_release -sc) main"
sudo apt update

# Install Isaac ROS packages
sudo apt install ros-humble-isaac-ros-apriltag
sudo apt install ros-humble-isaac-ros-visual-slam
sudo apt install ros-humble-isaac-ros-stereo-dense-reconstruction
sudo apt install ros-humble-isaac-ros-bi3d
sudo apt install ros-humble-isaac-ros-common
```

### 2. Verify Installation

```bash
# Check available Isaac ROS packages
ros2 pkg list | grep isaac_ros

# Verify GPU access
nvidia-smi
```

## Creating a Basic Perception Pipeline

### 1. Simple Camera Perception Node

Let's create a basic camera perception pipeline that processes RGB images:

```python
#!/usr/bin/env python3

"""
Basic camera perception pipeline using Isaac ROS
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class BasicPerceptionNode(Node):
    def __init__(self):
        super().__init__('basic_perception_node')

        # Initialize CvBridge for image conversion
        self.bridge = CvBridge()

        # Create subscriber for camera input
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Create publisher for processed image
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_processed',
            10
        )

        self.get_logger().info('Basic perception node initialized')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Example processing: edge detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            # Convert back to ROS Image
            processed_msg = self.bridge.cv2_to_imgmsg(edges, "mono8")
            processed_msg.header = msg.header

            # Publish processed image
            self.publisher.publish(processed_msg)

            self.get_logger().info('Processed image published')

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')


def main(args=None):
    rclpy.init(args=args)

    perception_node = BasicPerceptionNode()

    try:
        rclpy.spin(perception_node)
    except KeyboardInterrupt:
        perception_node.get_logger().info('Shutting down perception node')
    finally:
        perception_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2. Isaac ROS Visual SLAM Pipeline

For more advanced perception, let's set up a Visual SLAM pipeline:

```python
#!/usr/bin/env python3

"""
Isaac ROS Visual SLAM pipeline setup
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray
import tf2_ros

class IsaacVSLAMPipeline(Node):
    def __init__(self):
        super().__init__('isaac_vslam_pipeline')

        # Initialize parameters
        self.declare_parameter('enable_debug', False)
        self.enable_debug = self.get_parameter('enable_debug').value

        # Create subscribers for stereo camera input
        self.left_image_sub = self.create_subscription(
            Image,
            '/camera/left/image_rect_color',
            self.left_image_callback,
            10
        )

        self.right_image_sub = self.create_subscription(
            Image,
            '/camera/right/image_rect_color',
            self.right_image_callback,
            10
        )

        self.left_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/left/camera_info',
            self.left_info_callback,
            10
        )

        self.right_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/right/camera_info',
            self.right_info_callback,
            10
        )

        # Create subscribers for IMU if available
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self.imu_callback,
            10
        )

        # Publishers for SLAM results
        self.odom_pub = self.create_publisher(
            Odometry,
            '/visual_slam/odometry',
            10
        )

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/visual_slam/pose',
            10
        )

        if self.enable_debug:
            self.debug_pub = self.create_publisher(
                MarkerArray,
                '/visual_slam/debug_features',
                10
            )

        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Initialize SLAM components
        self.initialize_vslam_components()

        self.get_logger().info('Isaac ROS VSLAM pipeline initialized')

    def initialize_vslam_components(self):
        """Initialize VSLAM components"""
        # This would typically interface with Isaac ROS Visual SLAM packages
        # For now, we'll set up placeholders
        self.feature_detector = None
        self.tracker = None
        self.mapper = None
        self.optimizer = None

        self.prev_frame = None
        self.current_pose = None

    def left_image_callback(self, msg):
        """Process left camera image for stereo VSLAM"""
        try:
            # Process with Isaac ROS Visual SLAM
            # This would typically call Isaac ROS packages
            self.process_vslam_frame(msg, 'left')
        except Exception as e:
            self.get_logger().error(f'Error in left image callback: {e}')

    def right_image_callback(self, msg):
        """Process right camera image for stereo VSLAM"""
        try:
            # Process with Isaac ROS Visual SLAM
            self.process_vslam_frame(msg, 'right')
        except Exception as e:
            self.get_logger().error(f'Error in right image callback: {e}')

    def process_vslam_frame(self, image_msg, camera_side):
        """Process a frame through the VSLAM pipeline"""
        # Convert ROS Image to OpenCV
        cv_image = CvBridge().imgmsg_to_cv2(image_msg, "bgr8")

        # In a real implementation, this would call Isaac ROS Visual SLAM nodes
        # For now, we'll simulate the process
        if self.prev_frame is not None:
            # Estimate motion between frames
            motion_estimate = self.estimate_motion(self.prev_frame, cv_image)

            # Update pose
            if self.current_pose is None:
                self.current_pose = np.eye(4)  # Identity pose initially

            # Apply motion estimate to current pose
            self.current_pose = self.update_pose(self.current_pose, motion_estimate)

            # Publish odometry
            self.publish_odometry(self.current_pose, image_msg.header)

        self.prev_frame = cv_image

    def estimate_motion(self, prev_frame, curr_frame):
        """Estimate motion between two frames"""
        # This would use Isaac ROS Visual SLAM algorithms in a real implementation
        # For now, return identity (no motion)
        return np.eye(4)

    def update_pose(self, current_pose, motion_estimate):
        """Update current pose with motion estimate"""
        return np.dot(current_pose, motion_estimate)

    def publish_odometry(self, pose_matrix, header):
        """Publish odometry based on estimated pose"""
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"

        # Convert pose matrix to position and orientation
        position = pose_matrix[:3, 3]
        rotation_matrix = pose_matrix[:3, :3]

        # Convert rotation matrix to quaternion
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)

        odom_msg.pose.pose.position.x = position[0]
        odom_msg.pose.pose.position.y = position[1]
        odom_msg.pose.pose.position.z = position[2]

        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Publish odometry
        self.odom_pub.publish(odom_msg)

        # Broadcast transform
        t = tf2_ros.TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_link"
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.w = qw
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz

        self.tf_broadcaster.sendTransform(t)

    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        # Method to convert rotation matrix to quaternion
        trace = np.trace(R)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return qw, qx, qy, qz

    def left_info_callback(self, msg):
        """Handle left camera calibration info"""
        # Store camera calibration for rectification
        pass

    def right_info_callback(self, msg):
        """Handle right camera calibration info"""
        # Store camera calibration for rectification
        pass

    def imu_callback(self, msg):
        """Handle IMU data for VISUAL-IMU SLAM"""
        # Integrate IMU data for better pose estimation
        # This would be used with Isaac ROS Visual IMU SLAM
        pass


def main(args=None):
    rclpy.init(args=args)

    vslam_pipeline = IsaacVSLAMPipeline()

    try:
        rclpy.spin(vslam_pipeline)
    except KeyboardInterrupt:
        vslam_pipeline.get_logger().info('Shutting down VSLAM pipeline')
    finally:
        vslam_pipeline.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Launch Files for Perception Pipelines

### 1. Basic Perception Launch File

Create a launch file to bring up the basic perception stack:

```python
# perception_pipeline_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    enable_debug = DeclareLaunchArgument(
        'enable_debug',
        default_value='false',
        description='Enable debug visualization'
    )

    # Perception pipeline node
    perception_node = Node(
        package='my_perception_package',
        executable='basic_perception_node',
        name='basic_perception',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'enable_debug': LaunchConfiguration('enable_debug')}
        ],
        remappings=[
            ('/camera/image_raw', '/head_camera/image_raw'),
        ],
        output='screen'
    )

    # Isaac ROS Apriltag node (if using)
    apriltag_node = Node(
        package='isaac_ros_apriltag',
        executable='isaac_ros_apriltag_exe',
        name='isaac_ros_apriltag',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'family': '36h11'},
            {'max_tags': 64},
            {'quad_decimate': 2.0}
        ],
        remappings=[
            ('/image', '/head_camera/image_rect_color'),
            ('/camera_info', '/head_camera/camera_info'),
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        enable_debug,
        perception_node,
        # apriltag_node,  # Uncomment if using Apriltag
    ])
```

### 2. Visual SLAM Launch File

Create a launch file for the Visual SLAM pipeline:

```python
# visual_slam_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Isaac ROS Visual SLAM container
    vslam_container = ComposableNodeContainer(
        name='vslam_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Left rectify
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='stereo_image_proc::RectifyNode',
                name='left_rectify_node',
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
                remappings=[
                    ('/left/image_raw', '/camera/left/image_raw'),
                    ('/left/camera_info', '/camera/left/camera_info'),
                    ('/left/image_rect', '/camera/left/image_rect'),
                ]
            ),

            # Right rectify
            ComposableNode(
                package='isaac_ros_stereo_image_proc',
                plugin='stereo_image_proc::RectifyNode',
                name='right_rectify_node',
                parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
                remappings=[
                    ('/right/image_raw', '/camera/right/image_raw'),
                    ('/right/camera_info', '/camera/right/camera_info'),
                    ('/right/image_rect', '/camera/right/image_rect'),
                ]
            ),

            # Isaac ROS Visual SLAM node
            ComposableNode(
                package='isaac_ros_visual_slam',
                plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
                name='visual_slam',
                parameters=[{
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'enable_rectified_pose': True,
                    'rectified_pose_frame_id': 'camera_link'
                }],
                remappings=[
                    ('/visual_slam/imu', '/imu/data'),
                    ('/visual_slam/left/camera_info', '/camera/left/camera_info'),
                    ('/visual_slam/right/camera_info', '/camera/right/camera_info'),
                    ('/visual_slam/left/image', '/camera/left/image_rect'),
                    ('/visual_slam/right/image', '/camera/right/image_rect'),
                ]
            ),
        ],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time,
        vslam_container,
    ])
```

## Isaac Sim Integration

### 1. Setting Up Sensors in Isaac Sim

For perception pipeline development, you'll need properly configured sensors in Isaac Sim:

```python
# Isaac Sim perception setup
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np

class IsaacSimPerceptionSetup:
    def __init__(self):
        self.world = World(stage_units_in_meters=1.0)

        # Setup stereo cameras for VSLAM
        self.setup_stereo_cameras()

        # Setup additional sensors as needed
        self.setup_imu()

    def setup_stereo_cameras(self):
        """Setup stereo cameras for Visual SLAM"""
        # Left camera
        self.left_camera = Camera(
            prim_path="/World/Robot/Sensors/CameraLeft",
            name="left_camera",
            position=np.array([0.1, 0.05, 0.1]),  # 10cm baseline
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Right camera
        self.right_camera = Camera(
            prim_path="/World/Robot/Sensors/CameraRight",
            name="right_camera",
            position=np.array([0.1, -0.05, 0.1]),  # 10cm baseline
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )

        # Initialize cameras
        self.left_camera.initialize()
        self.right_camera.initialize()

        # Set up render products for each camera
        self.left_camera.add_render_product(resolution=(640, 480))
        self.right_camera.add_render_product(resolution=(640, 480))

        # Set up ROS bridge publishing
        self.left_camera.set_render_product_frequency("publish")
        self.right_camera.set_render_product_frequency("publish")

    def setup_imu(self):
        """Setup IMU sensor for VISUAL-IMU fusion"""
        # In Isaac Sim, IMU sensors can be added to links
        # This would typically be done as part of the robot USD
        pass

    def get_camera_data(self):
        """Get data from both cameras"""
        left_data = self.left_camera.get_render_product().get_data()
        right_data = self.right_camera.get_render_product().get_data()

        return {
            'left': left_data,
            'right': right_data
        }
```

## Performance Optimization

### 1. GPU Acceleration Settings

To maximize performance with Isaac ROS packages:

```yaml
# perception_pipeline_params.yaml
isaac_ros_apriltag:
  ros__parameters:
    family: "36h11"
    max_tags: 64
    size: 0.166  # Tag size in meters
    quad_decimate: 2.0
    quad_sigma: 0.0
    refine_edges: 1
    decode_sharpening: 0.25
    debug: false

isaac_ros_visual_slam:
  ros__parameters:
    enable_debug_mode: false
    enable_mapping: true
    enable_localization: true
    use_sim_time: true
    publish_graph_mesh: false
    publish_tf: true
    publish_odom: true
    min_num_features: 100
    max_num_features: 1000
    tracking_rate: 30.0
    mapping_rate: 1.0
    cuda_device_id: 0
```

### 2. Memory Management

For perception pipelines that process large amounts of data:

```python
class OptimizedPerceptionNode(Node):
    def __init__(self):
        super().__init__('optimized_perception_node')

        # Limit queue sizes to prevent memory buildup
        qos_profile = rclpy.qos.QoSProfile(
            depth=1,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.optimized_image_callback,
            qos_profile
        )

        # Use threading for heavy processing
        self.processing_executor = rclpy.executors.SingleThreadedExecutor()

    def optimized_image_callback(self, msg):
        """Optimized callback with proper resource management"""
        # Process image in separate thread to avoid blocking
        self.processing_executor.add_node(self.process_image_async(msg))
```

## Best Practices

### 1. Pipeline Design
- Use appropriate sensor fusion for robust perception
- Implement proper error handling and fallback behaviors
- Validate sensor data before processing
- Monitor computational resources and adjust accordingly

### 2. Testing and Validation
- Test perception pipelines in simulation first
- Use ground truth data for validation
- Monitor accuracy metrics continuously
- Implement safety checks for perception failures

### 3. Configuration Management
- Use parameter files for pipeline configuration
- Implement runtime reconfiguration
- Validate parameters at startup
- Document parameter effects clearly

## Troubleshooting Common Issues

### 1. Performance Issues
- **Slow Processing**: Check GPU utilization and memory usage
- **Frame Drops**: Reduce image resolution or processing complexity
- **Latency**: Optimize queue sizes and processing pipeline

### 2. Accuracy Issues
- **Drift in SLAM**: Check IMU calibration and visual features
- **False Positives**: Adjust detection thresholds and validation
- **Poor Tracking**: Improve lighting conditions and feature richness

### 3. Integration Issues
- **Topic Mismatch**: Verify topic names and message types
- **Timing Issues**: Check clock synchronization between sensors
- **Coordinate Systems**: Verify frame transformations are correct

## Learning Objectives

After completing this workflow, you should understand:
- How to set up basic perception pipelines with Isaac ROS packages
- How to configure stereo cameras for Visual SLAM
- How to integrate Isaac Sim sensors with ROS perception nodes
- Performance optimization techniques for perception systems
- Best practices for robust perception pipeline design

## Next Steps

Continue to learn about [Navigation Setup with Nav2](./navigation-setup) to understand how to implement robot navigation using the perception data from your pipeline.