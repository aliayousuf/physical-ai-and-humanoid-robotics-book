---
title: "Navigation Setup with Nav2"
description: "Configuring Navigation2 for humanoid path planning and navigation"
---

# Navigation Setup with Nav2

## Overview

Navigation2 (Nav2) is the state-of-the-art navigation framework for ROS 2, designed for autonomous mobile robots. This workflow guide covers setting up Nav2 for humanoid robots, including configuration, parameter tuning, and integration with perception systems. Unlike wheeled robots, humanoid robots have unique navigation requirements due to their bipedal locomotion and stability constraints.

## Prerequisites

Before setting up navigation, ensure you have:
- Completed Module 1 (ROS 2 fundamentals)
- Completed Module 2 (Digital Twin simulation)
- Understanding of basic navigation concepts (path planning, localization)
- Isaac Sim environment with sensors configured
- Perception pipeline for environment sensing

## Nav2 Architecture

### Core Components

#### 1. Navigation Stack Components
- **Navigator**: Coordinates navigation execution
- **Planner Server**: Global path planning
- **Controller Server**: Local path following and obstacle avoidance
- **Recovery Server**: Behavior trees for getting unstuck
- **BT Navigator**: Behavior tree executor for navigation actions

#### 2. Map Management
- **Map Server**: Provides static map
- **Lifecycle Manager**: Manages navigation lifecycle
- **Amcl**: Adaptive Monte Carlo Localization

### Humanoid-Specific Considerations

Humanoid robots require special navigation configurations:
- **Footstep Planning**: Discrete footstep generation instead of continuous motion
- **Stability Constraints**: Maintain center of mass within support polygon
- **Step Height Limitations**: Cannot step over large obstacles
- **Turning Radius**: Different kinematic constraints than wheeled robots

## Installation and Dependencies

### 1. Install Nav2 Packages

```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2
sudo apt install ros-humble-nav2-bringup
sudo apt install ros-humble-dwb-core
sudo apt install ros-humble-nav2-console-tools
sudo apt install ros-humble-nav2-util
sudo apt install ros-humble-nav2-zarr
```

### 2. Install Isaac-Specific Navigation Packages

```bash
# Install Isaac ROS navigation packages
sudo apt install ros-humble-isaac-ros-nav2-btree
sudo apt install ros-humble-isaac-ros-occupancy-grid-node
```

## Basic Navigation Configuration

### 1. Navigation Parameters File

Create a comprehensive navigation configuration file:

```yaml
# nav2_params.yaml
amcl:
  ros__parameters:
    use_sim_time: True
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    base_frame_id: "base_footprint"  # Changed for humanoid
    beam_skip_distance: 0.5
    beam_skip_error_threshold: 0.9
    beam_skip_threshold: 0.3
    do_beamskip: false
    global_frame_id: "map"
    lambda_short: 0.1
    likelihood_max_dist: 2.0
    max_beams: 60
    max_particles: 2000
    min_particles: 500
    odom_frame_id: "odom"
    pf_err: 0.05
    pf_z: 0.99
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    resample_interval: 1
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    save_pose_rate: 0.5
    sigma_hit: 0.2
    tf_broadcast: true
    transform_tolerance: 1.0
    update_min_a: 0.2
    update_min_d: 0.25
    z_hit: 0.5
    z_max: 0.05
    z_rand: 0.5
    z_short: 0.05

amcl_map_client:
  ros__parameters:
    use_sim_time: True

amcl_rclcpp_node:
  ros__parameters:
    use_sim_time: True

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_footprint"  # Humanoid-specific
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "package://nav2_bt_navigator/bt_xml_v0.16/navigate_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "package://nav2_bt_navigator/bt_xml_v0.16/navigate_w_replanning_and_recovery.xml"
    plugin_lib_names:
      - nav2_compute_path_to_pose_action_bt_node
      - nav2_compute_path_through_poses_action_bt_node
      - nav2_smooth_path_action_bt_node
      - nav2_follow_path_action_bt_node
      - nav2_spin_action_bt_node
      - nav2_wait_action_bt_node
      - nav2_assisted_teleop_action_bt_node
      - nav2_back_up_action_bt_node
      - nav2_drive_on_heading_bt_node
      - nav2_clear_costmap_service_bt_node
      - nav2_is_stuck_condition_bt_node
      - nav2_goal_reached_condition_bt_node
      - nav2_initial_pose_received_condition_bt_node
      - nav2_reinitialize_global_localization_service_bt_node
      - nav2_rate_controller_bt_node
      - nav2_distance_controller_bt_node
      - nav2_speed_controller_bt_node
      - nav2_truncate_path_action_bt_node
      - nav2_truncate_path_local_action_bt_node
      - nav2_goal_updater_node_bt_node
      - nav2_recovery_node_bt_node
      - nav2_pipeline_sequence_bt_node
      - nav2_round_robin_node_bt_node
      - nav2_transform_available_condition_bt_node
      - nav2_time_expired_condition_bt_node
      - nav2_path_expiring_timer_condition
      - nav2_distance_traveled_condition_bt_node
      - nav2_time_between_requests_condition_bt_node
      - nav2_is_battery_low_condition_bt_node
      - nav2_navigate_through_poses_action_bt_node
      - nav2_navigate_to_pose_action_bt_node
      - nav2_remove_passed_goals_action_bt_node
      - nav2_planner_selector_bt_node
      - nav2_controller_selector_bt_node
      - nav2_goal_checker_selector_bt_node
      - nav2_controller_cancel_bt_node
      - nav2_path_longer_on_approach_bt_node
      - nav2_wait_cancel_bt_node
      - nav2_spin_cancel_bt_node
      - nav2_backup_cancel_bt_node
      - nav2_assisted_teleop_cancel_bt_node
      - nav2_drive_on_heading_cancel_bt_node
      - nav2_is_battery_charging_condition_bt_node

bt_navigator_rclcpp_node:
  ros__parameters:
    use_sim_time: True

controller_server:
  ros__parameters:
    use_sim_time: True
    controller_frequency: 20.0
    min_x_velocity_threshold: 0.001
    min_y_velocity_threshold: 0.5
    min_theta_velocity_threshold: 0.001
    progress_checker_plugin: "progress_checker"
    goal_checker_plugin: "goal_checker"
    controller_plugins: ["FollowPath"]

    # Humanoid-specific controller for bipedal locomotion
    FollowPath:
      plugin: "nav2_mppi_controller::MPPICtrl"
      debug_enabled: False
      time_steps: 24
      control_freq: 20.0
      time_horizon: 1.2
      feasible_window: 1.0
      discretization: 0.4
      penalty_velocity: 0.0
      penalty_collision: 10000.0
      penalty_goal: 5.0
      penalty_goal_front: 10.0
      penalty_centripetal: 100.0
      penalty_angular_velocity: 1.0
      penalty_steering: 0.0
      weight_adhesion: 0.0
      weight_velocity: 100.0
      weight_smoothness: 100.0
      weight_feasibility: 1.0
      weight_optimization: 100.0
      weight_clearing: 100.0
      weight_regulation: 100.0
      weight_collision: 1000.0
      weight_goal_distance: 100.0
      weight_goal_angle: 100.0
      weight_path_distance: 100.0
      weight_path_angle: 100.0
      weight_path_curvature: 100.0
      weight_path_clearance: 100.0
      weight_path_feasibility: 100.0
      weight_path_smoothness: 100.0
      weight_path_optimization: 100.0

controller_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_footprint  # Changed for humanoid
      use_sim_time: True
      rolling_window: true
      width: 6
      height: 6
      resolution: 0.05
      robot_radius: 0.3  # Adjust for humanoid footprint
      plugins: ["voxel_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55
      voxel_layer:
        plugin: "nav2_costmap_2d::VoxelLayer"
        enabled: True
        publish_voxel_map: True
        origin_z: 0.0
        z_resolution: 0.2
        z_voxels: 10
        max_obstacle_height: 2.0
        mark_threshold: 0
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0

global_costmap:
  global_costmap:
    ros__parameters:
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_footprint
      use_sim_time: True
      robot_radius: 0.3  # Adjust for humanoid
      resolution: 0.05
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 3.0
          raytrace_min_range: 0.0
          obstacle_max_range: 2.5
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 3.0
        inflation_radius: 0.55

planner_server:
  ros__parameters:
    expected_planner_frequency: 20.0
    use_sim_time: True
    planner_plugins: ["GridBased"]
    GridBased:
      plugin: "nav2_navfn_planner::NavfnPlanner"
      tolerance: 0.5
      use_astar: false
      allow_unknown: true

planner_server_rclcpp_node:
  ros__parameters:
    use_sim_time: True

recoveries_server:
  ros__parameters:
    costmap_topic: local_costmap/costmap_raw
    footprint_topic: local_costmap/published_footprint
    cycle_frequency: 10.0
    recovery_plugins: ["spin", "backup", "wait"]
    spin:
      plugin: "nav2_recoveries/Spin"
    backup:
      plugin: "nav2_recoveries/BackUp"
    wait:
      plugin: "nav2_recoveries/Wait"
    global_frame: odom
    robot_base_frame: base_footprint
    transform_timeout: 0.1
    use_sim_time: True
    simulate_ahead_time: 2.0
    max_rotational_vel: 1.0
    min_rotational_vel: 0.4
    rotational_acc_lim: 3.2

waypoint_follower:
  ros__parameters:
    loop_rate: 20
    stop_on_failure: false
    waypoint_task_executor_plugin: "wait_at_waypoint"
    wait_at_waypoint:
      plugin: "nav2_waypoint_follower::WaitAtWaypoint"
      enabled: true
      waypoint_pause_duration: 200
```

### 2. Humanoid-Specific Navigation Node

Create a navigation node tailored for humanoid robots:

```python
#!/usr/bin/env python3

"""
Humanoid Navigation Node with special considerations for bipedal locomotion
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from nav2_msgs.action import NavigateToPose
from builtin_interfaces.msg import Duration
from std_srvs.srv import Empty
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidNavigationNode(Node):
    def __init__(self):
        super().__init__('humanoid_navigation')

        # Initialize navigation client
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Initialize TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Navigation parameters specific to humanoid
        self.step_height = 0.1  # Height to lift foot during stepping
        self.step_spacing = 0.3  # Distance between footsteps
        self.max_step_climb = 0.15  # Maximum height humanoid can step up
        self.support_polygon_radius = 0.2  # Radius of support polygon for stability

        # Publishers and subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.footstep_pub = self.create_publisher(Path, '/footsteps', 10)

        # Services for navigation control
        self.cancel_nav_srv = self.create_service(Empty, 'cancel_navigation', self.cancel_navigation)
        self.pause_nav_srv = self.create_service(Empty, 'pause_navigation', self.pause_navigation)

        self.get_logger().info('Humanoid navigation node initialized')

    def navigate_to_pose(self, x, y, theta):
        """Send navigation goal to Nav2"""
        goal_msg = NavigateToPose.Goal()

        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)
        goal_msg.pose.pose.position.z = 0.0

        # Convert theta (yaw) to quaternion
        q = self.yaw_to_quaternion(theta)
        goal_msg.pose.pose.orientation.w = q[0]
        goal_msg.pose.pose.orientation.x = q[1]
        goal_msg.pose.pose.orientation.y = q[2]
        goal_msg.pose.pose.orientation.z = q[3]

        # Wait for server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Navigation server not available')
            return False

        # Send goal
        future = self.nav_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

        return True

    def goal_response_callback(self, future):
        """Handle navigation goal response"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Handle navigation result"""
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

    def cancel_navigation(self, request, response):
        """Cancel current navigation goal"""
        # Implementation to cancel navigation
        self.get_logger().info('Navigation cancelled')
        return response

    def pause_navigation(self, request, response):
        """Pause current navigation"""
        self.get_logger().info('Navigation paused')
        return response

    def generate_footsteps(self, path):
        """Generate discrete footsteps for humanoid locomotion"""
        footsteps = Path()
        footsteps.header.frame_id = 'map'
        footsteps.header.stamp = self.get_clock().now().to_msg()

        # Convert continuous path to discrete footsteps
        # This is a simplified implementation - real footstep planners are more complex
        for i in range(0, len(path.poses), int(self.step_spacing / 0.1)):
            if i < len(path.poses):
                footsteps.poses.append(path.poses[i])

        # Add intermediate steps for stability
        interpolated_steps = self.interpolate_stable_steps(footsteps)
        return interpolated_steps

    def interpolate_stable_steps(self, footsteps):
        """Add intermediate steps for stability"""
        stable_steps = Path()
        stable_steps.header = footsteps.header

        for i in range(len(footsteps.poses) - 1):
            start_pose = footsteps.poses[i].pose
            end_pose = footsteps.poses[i + 1].pose

            # Calculate intermediate steps based on humanoid kinematics
            dx = end_pose.position.x - start_pose.position.x
            dy = end_pose.position.y - start_pose.position.y
            distance = np.sqrt(dx*dx + dy*dy)

            num_intermediate_steps = int(distance / (self.step_spacing / 2))  # More steps for stability

            for j in range(num_intermediate_steps):
                ratio = j / num_intermediate_steps
                interp_pose = PoseStamped()
                interp_pose.header = footsteps.header
                interp_pose.pose.position.x = start_pose.position.x + ratio * dx
                interp_pose.pose.position.y = start_pose.position.y + ratio * dy
                interp_pose.pose.position.z = start_pose.position.z  # Maintain height

                # Interpolate orientation
                start_quat = [start_pose.orientation.w, start_pose.orientation.x,
                             start_pose.orientation.y, start_pose.orientation.z]
                end_quat = [end_pose.orientation.w, end_pose.orientation.x,
                           end_pose.orientation.y, end_pose.orientation.z]

                # Spherical linear interpolation (SLERP) for quaternions
                interp_quat = self.slerp_quaternions(start_quat, end_quat, ratio)
                interp_pose.pose.orientation.w = interp_quat[0]
                interp_pose.pose.orientation.x = interp_quat[1]
                interp_pose.pose.orientation.y = interp_quat[2]
                interp_pose.pose.orientation.z = interp_quat[3]

                stable_steps.poses.append(interp_pose)

        return stable_steps

    def slerp_quaternions(self, q1, q2, t):
        """Spherical linear interpolation between two quaternions"""
        # Convert to numpy arrays
        q1 = np.array(q1)
        q2 = np.array(q2)

        # Calculate dot product
        dot = np.dot(q1, q2)

        # If dot product is negative, negate one quaternion
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Calculate interpolation
        DOT_THRESHOLD = 0.9995
        if dot > DOT_THRESHOLD:
            # Linear interpolation for very similar quaternions
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        # Calculate angle between quaternions
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        # Interpolation
        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        result = s0 * q1 + s1 * q2
        return result / np.linalg.norm(result)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cq = np.cos(0.0 * 0.5)
        sq = np.sin(0.0 * 0.5)
        cr = np.cos(0.0 * 0.5)
        sr = np.sin(0.0 * 0.5)

        w = cr * cq * cy + sr * sq * sy
        x = sr * cq * cy - cr * sq * sy
        y = cr * sq * cy + sr * cq * sy
        z = cr * cq * sy - sr * sq * cy

        return [w, x, y, z]

    def validate_navigation_plan(self, plan):
        """Validate plan for humanoid-specific constraints"""
        # Check for obstacles too high to step over
        for pose in plan.poses:
            # In a real implementation, this would check against costmap
            # to ensure step height limitations are respected
            pass

        # Check for adequate support polygon clearance
        # Ensure robot can maintain balance along path
        return True

def main(args=None):
    rclpy.init(args=args)
    nav_node = HumanoidNavigationNode()

    try:
        rclpy.spin(nav_node)
    except KeyboardInterrupt:
        nav_node.get_logger().info('Shutting down humanoid navigation node')
    finally:
        nav_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Sim Integration for Navigation

### 1. Isaac Sim Navigation Setup

Integrate Isaac Sim with Nav2 for simulation-based navigation testing:

```python
# isaac_sim_nav_integration.py
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.navigation import NavigationGraph
from omni.isaac.synthetic_utils import SyntheticDataHelper
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

class IsaacSimNavIntegration:
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # Initialize ROS
        rclpy.init()
        self.nav_node = rclpy.create_node('isaac_sim_nav_integration')

        # Publishers for simulated sensor data
        self.scan_pub = self.nav_node.create_publisher(LaserScan, '/scan', 10)
        self.map_pub = self.nav_node.create_publisher(OccupancyGrid, '/map', 10)

        # Timer for publishing simulated data
        self.nav_node.create_timer(0.1, self.publish_simulated_data)  # 10 Hz

        # Setup navigation graph in Isaac Sim
        self.setup_navigation_graph()

    def setup_navigation_graph(self):
        """Setup navigation graph for path planning in Isaac Sim"""
        # In Isaac Sim, navigation graphs can be generated automatically
        # or manually placed for complex environments
        pass

    def generate_simulated_scan(self):
        """Generate simulated LiDAR scan data"""
        # This would interface with Isaac Sim's sensor system
        # to generate realistic scan data based on the environment
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.nav_node.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -np.pi
        scan_msg.angle_max = np.pi
        scan_msg.angle_increment = 0.0174533  # 1 degree
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 30.0

        # Generate sample ranges (this would come from Isaac Sim sensors in reality)
        num_readings = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment)
        scan_msg.ranges = [10.0] * num_readings  # Default to max range

        return scan_msg

    def generate_occupancy_grid(self):
        """Generate occupancy grid from Isaac Sim environment"""
        # Create occupancy grid from Isaac Sim world representation
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.nav_node.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'

        # Define grid properties
        grid_msg.info.resolution = 0.05  # 5cm per cell
        grid_msg.info.width = 200  # 10m x 10m grid
        grid_msg.info.height = 200
        grid_msg.info.origin.position.x = -5.0
        grid_msg.info.origin.position.y = -5.0

        # Generate occupancy data (this would come from Isaac Sim collision geometry)
        # 0 = free, 100 = occupied, -1 = unknown
        grid_msg.data = [0] * (grid_msg.info.width * grid_msg.info.height)

        return grid_msg

    def publish_simulated_data(self):
        """Publish simulated sensor and map data"""
        # Generate and publish simulated laser scan
        scan_data = self.generate_simulated_scan()
        self.scan_pub.publish(scan_data)

        # Generate and publish occupancy grid
        grid_data = self.generate_occupancy_grid()
        self.map_pub.publish(grid_data)

    def get_robot_pose(self):
        """Get robot pose from Isaac Sim"""
        # This would interface with Isaac Sim to get the current robot pose
        # for localization and navigation
        pass

    def execute_navigation_command(self, goal_pose):
        """Execute navigation command in Isaac Sim"""
        # Send navigation command to Isaac Sim robot
        # This would interface with the robot's locomotion system
        pass
```

## Launch Files for Navigation

### 1. Navigation Launch File

Create a launch file to bring up the complete navigation stack:

```python
# navigation_launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
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

    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_navigation'),
            'config',
            'nav2_params.yaml'
        ]),
        description='Full path to params file for navigation nodes'
    )

    autostart = DeclareLaunchArgument(
        'autostart',
        default_value='true',
        description='Automatically startup the nav2 stack'
    )

    map_yaml_file = DeclareLaunchArgument(
        'map',
        description='Full path to map file to load'
    )

    # Map server
    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                   {'yaml_filename': LaunchConfiguration('map')}]
    )

    # Lifecycle manager for map server
    lifecycle_manager_map = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                   {'autostart': LaunchConfiguration('autostart')},
                   {'node_names': ['map_server']}]
    )

    # AMCL (Adaptive Monte Carlo Localization)
    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[LaunchConfiguration('params_file'),
                   {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Lifecycle manager for AMCL
    lifecycle_manager_localization = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                   {'autostart': LaunchConfiguration('autostart')},
                   {'node_names': ['map_server', 'amcl']}]
    )

    # Navigation server container
    navigation_container = ComposableNodeContainer(
        name='navigation_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            # Planner server
            ComposableNode(
                package='nav2_planner',
                plugin='nav2_planner::PlannerServer',
                name='planner_server',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}]
            ),

            # Controller server
            ComposableNode(
                package='nav2_controller',
                plugin='nav2_controller::ControllerServer',
                name='controller_server',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}]
            ),

            # Recoveries server
            ComposableNode(
                package='nav2_recoveries',
                plugin='nav2_recoveries::RecoveryServer',
                name='recoveries_server',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}]
            ),

            # BT navigator
            ComposableNode(
                package='nav2_bt_navigator',
                plugin='nav2_bt_navigator::BtNavigator',
                name='bt_navigator',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}]
            ),

            # Waypoint follower
            ComposableNode(
                package='nav2_waypoint_follower',
                plugin='nav2_waypoint_follower::WaypointFollower',
                name='waypoint_follower',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}]
            )
        ],
        output='screen'
    )

    # Lifecycle manager for navigation
    lifecycle_manager_navigation = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_navigation',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                   {'autostart': LaunchConfiguration('autostart')},
                   {'node_names': ['planner_server',
                                 'controller_server',
                                 'recoveries_server',
                                 'bt_navigator',
                                 'waypoint_follower']}]
    )

    return LaunchDescription([
        use_sim_time,
        params_file,
        autostart,
        map_yaml_file,
        map_server,
        lifecycle_manager_map,
        amcl,
        lifecycle_manager_localization,
        navigation_container,
        lifecycle_manager_navigation
    ])
```

## Performance Optimization

### 1. Costmap Optimization

Optimize costmaps for humanoid navigation:

```yaml
# Optimized costmap parameters for humanoid robots
local_costmap:
  local_costmap:
    ros__parameters:
      # Higher resolution for detailed obstacle detection
      resolution: 0.025  # 2.5cm per cell (higher than default)
      # Larger local costmap for planning ahead
      width: 10.0
      height: 10.0
      # Use robot footprint instead of radius for accurate collision checking
      footprint: [[0.25, 0.25], [0.25, -0.25], [-0.25, -0.25], [-0.25, 0.25]]
      plugins: ["obstacle_layer", "inflation_layer"]
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        # Adjust inflation for humanoid step capabilities
        cost_scaling_factor: 2.0  # Reduced from default for humanoid
        inflation_radius: 0.4     # Adjusted for humanoid safety margin
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 1.8  # Humanoid height consideration
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.1
```

### 2. Controller Tuning for Humanoid Robots

Adjust controller parameters for humanoid-specific movement:

```yaml
# Humanoid-specific controller parameters
controller_server:
  ros__parameters:
    # Lower frequency for more stable humanoid movement
    controller_frequency: 10.0  # Reduced from 20 for humanoid stability
    # Adjust velocity thresholds for humanoid capabilities
    min_x_velocity_threshold: 0.05  # Slower for stability
    min_y_velocity_threshold: 0.1   # Account for humanoid lateral movement
    min_theta_velocity_threshold: 0.05
    # Use MPPI controller for humanoid-specific constraints
    FollowPath:
      plugin: "nav2_mppi_controller::MPPICtrl"
      # Conservative parameters for humanoid stability
      time_steps: 16        # Reduced horizon for faster reaction
      control_freq: 10.0    # Match controller frequency
      time_horizon: 1.6     # Longer horizon for smoother paths
      feasible_window: 0.8  # Conservative feasibility window
      discretization: 0.3   # Appropriate for humanoid step size
      # Weight adjustments for humanoid movement
      weight_velocity: 80.0     # Prioritize forward motion
      weight_smoothness: 120.0  # Prioritize smooth movement for stability
      weight_collision: 1500.0  # High collision avoidance
      weight_goal_distance: 120.0
      weight_goal_angle: 120.0
```

## Troubleshooting Navigation Issues

### 1. Common Navigation Problems

#### Local Minima
- **Symptoms**: Robot gets stuck oscillating near obstacles
- **Solutions**:
  - Increase local costmap inflation
  - Adjust controller parameters for more aggressive obstacle avoidance
  - Use recovery behaviors effectively

#### Footstep Planning Issues
- **Symptoms**: Robot attempts impossible steps or falls
- **Solutions**:
  - Validate step height limitations in costmap
  - Implement proper footstep planning algorithms
  - Add stability constraints to navigation

#### Localization Drift
- **Symptoms**: Robot position estimate becomes inaccurate
- **Solutions**:
  - Ensure proper sensor calibration
  - Check for sufficient visual features in environment
  - Validate IMU integration for VISUAL-IMU SLAM

### 2. Performance Optimization

#### Computing Resource Management
- Monitor CPU/GPU usage of navigation stack
- Adjust update frequencies based on robot capabilities
- Use appropriate costmap resolutions

#### Memory Management
- Limit costmap sizes to necessary areas
- Use rolling windows for local costmaps
- Implement proper cleanup of old path plans

## Best Practices

### 1. Safety Considerations
- Implement proper obstacle detection and avoidance
- Use appropriate safety margins in costmap inflation
- Include recovery behaviors for exceptional situations
- Validate navigation commands before execution

### 2. Testing and Validation
- Test navigation in simulation before real-world deployment
- Validate step height limitations in various terrains
- Verify stability constraints during path following
- Test recovery behaviors in challenging scenarios

### 3. Humanoid-Specific Navigation
- Account for bipedal locomotion dynamics
- Consider balance and stability during movement
- Implement discrete footstep planning
- Adjust velocity profiles for humanoid gait patterns

## Learning Objectives

After completing this workflow, you should understand:
- How to configure Nav2 for humanoid robot navigation
- The special considerations for bipedal locomotion in navigation
- How to integrate Isaac Sim with Nav2 for simulation-based testing
- Performance optimization techniques for navigation systems
- Troubleshooting common navigation issues in humanoid robots

## Next Steps

Continue to learn about [Reinforcement Learning for Robotics](./rl-training) to understand how to train robot navigation behaviors using machine learning approaches.