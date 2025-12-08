---
title: "Nav2 Path Planning"
description: "Using Navigation2 for humanoid path planning and navigation"
---

# Nav2 Path Planning

## Overview

Navigation2 (Nav2) is the state-of-the-art navigation framework for ROS 2, designed for autonomous mobile robots. This system provides a complete solution for path planning, obstacle avoidance, and navigation in complex environments. For humanoid robots, Nav2 requires special configuration to account for their unique kinematics and mobility patterns.

## Key Components

### Navigation System Architecture

#### Global Planner
- Computes optimal path from start to goal
- Considers static map and known obstacles
- Outputs geometric path for the robot to follow
- Common algorithms: A*, Dijkstra, NavFn, RRT*

#### Local Planner
- Performs short-term path following and obstacle avoidance
- Reacts to dynamic obstacles in real-time
- Controls robot velocities to follow global path
- Common algorithms: DWA, TEB, MPC

#### Costmap Layers
- Static layer: Based on static map
- Obstacle layer: From sensor data
- Inflation layer: Safety margins around obstacles
- Additional layers: Dynamic obstacles, footprints, etc.

### Special Considerations for Humanoid Robots

Humanoid robots have unique navigation requirements compared to wheeled robots:

- **Footstep Planning**: Need to plan discrete foot placements
- **Stability Constraints**: Maintain balance during movement
- **Dynamic Walking**: Different from holonomic motion
- **Step Height Limitations**: Cannot step over large obstacles
- **Turning Radius**: Different kinematic constraints

## Installation and Setup

### System Requirements
- ROS 2 Humble Hawksbill or later
- Compatible sensors (LiDAR, cameras, IMU)
- Adequate computational resources
- Navigation-compatible robot base

### Installation Process
```bash
# Install Nav2 packages
sudo apt update
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup
sudo apt install ros-humble-dwb-core ros-humble-nav2-msgs
sudo apt install ros-humble-nav2-behaviors ros-humble-slam-toolbox

# Install additional packages for humanoid navigation
sudo apt install ros-humble-footprint-costmap ros-humble-legged-robots
```

## Configuration for Humanoid Robots

### Basic Configuration File

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
    base_frame_id: "base_link"
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

bt_navigator:
  ros__parameters:
    use_sim_time: True
    global_frame: "map"
    robot_base_frame: "base_link"
    odom_topic: "/odom"
    bt_loop_duration: 10
    default_server_timeout: 20
    enable_groot_monitoring: True
    groot_zmq_publisher_port: 1666
    groot_zmq_server_port: 1667
    default_nav_through_poses_bt_xml: "package://nav2_bt_navigator/behavior_trees/navigate_through_poses_w_replanning_and_recovery.xml"
    default_nav_to_pose_bt_xml: "package://nav2_bt_navigator/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml"
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
      - nav2_are_equal_poses_condition_bt_node
      - nav2_are_goals_equal_condition_bt_node
      - nav2_is_path_valid_condition_bt_node
      - nav2_validate_route_action_bt_node
      - nav2_initial_pose_publisher_bt_node
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
      - nav2_back_up_cancel_bt_node
      - nav2_assisted_teleop_cancel_bt_node
      - nav2_drive_on_heading_cancel_bt_node
      - nav2_is_battery_charging_condition_bt_node

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

    # Humanoid-specific controller
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

local_costmap:
  local_costmap:
    ros__parameters:
      update_frequency: 5.0
      publish_frequency: 2.0
      global_frame: odom
      robot_base_frame: base_link
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
      robot_base_frame: base_link
      use_sim_time: True
      robot_radius: 0.3  # Adjust for humanoid footprint
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
```

## Humanoid-Specific Navigation

### Footstep Planning
For humanoid robots, navigation must account for discrete footstep planning:

```python
# Example: Humanoid footstep planner
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray
import numpy as np

class HumanoidFootstepPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_footstep_planner')

        self.path_sub = self.create_subscription(
            Path,
            '/plan',
            self.path_callback,
            10
        )

        self.footsteps_pub = self.create_publisher(
            MarkerArray,
            '/footsteps',
            10
        )

        self.step_height = 0.1  # Height to lift foot during stepping
        self.step_spacing = 0.3  # Distance between footsteps
        self.support_polygon = self.calculate_support_polygon()

    def path_callback(self, msg):
        """Convert continuous path to discrete footsteps"""
        footsteps = self.discretize_path_to_footsteps(msg.poses)
        self.publish_footsteps(footsteps)

    def discretize_path_to_footsteps(self, poses):
        """Convert continuous path to discrete footsteps for humanoid"""
        footsteps = []

        # Start with current position
        if len(poses) > 0:
            footsteps.append(poses[0])

        # Add footsteps at regular intervals
        cumulative_distance = 0.0
        prev_pose = poses[0] if poses else None

        for pose in poses[1:]:
            # Calculate distance from previous pose
            dx = pose.pose.position.x - prev_pose.pose.position.x
            dy = pose.pose.position.y - prev_pose.pose.position.y
            distance = np.sqrt(dx*dx + dy*dy)

            cumulative_distance += distance

            # Add footsteps at regular intervals
            if cumulative_distance >= self.step_spacing:
                footsteps.append(pose)
                cumulative_distance = 0.0
                prev_pose = pose

        return footsteps

    def calculate_support_polygon(self):
        """Calculate the support polygon for stability"""
        # Define the area where feet must be placed to maintain balance
        # This is simplified - real implementation would be more complex
        return np.array([
            [-0.1, -0.1],  # back left
            [0.1, -0.1],   # back right
            [0.1, 0.1],    # front right
            [-0.1, 0.1]    # front left
        ])

    def publish_footsteps(self, footsteps):
        """Publish footsteps as visualization markers"""
        marker_array = MarkerArray()

        for i, step in enumerate(footsteps):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "footsteps"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose = step.pose
            marker.pose.position.z = self.step_height / 2  # Lift slightly above ground
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.05
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.footsteps_pub.publish(marker_array)
```

### Stability Considerations

Humanoid robots must maintain stability during navigation:

- **Zero-Moment Point (ZMP)**: Ensure foot placements maintain ZMP within support polygon
- **Capture Point**: Plan footsteps to bring robot to a stop at desired location
- **Dynamic Balance**: Adjust walking gait based on terrain and obstacles

## Behavior Trees for Navigation

Nav2 uses behavior trees for complex navigation behaviors:

```xml
<!-- Example behavior tree for humanoid navigation -->
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <SequenceStar name="NavigateToPose">
      <Fallback name="RecoveryFallback">
        <SequenceStar name="NavigateWithReplanning">
          <RecoveryNode number_of_retries="6" name="ComputeAndFollowPathRecovery">
            <PipelineSequence name="ComputeAndFollowPath">
              <RateController hz="1.0" name="ComputePathThroughPoses">
                <ReactiveSequence name="ComputePathThroughPosesWithCancel">
                  <GoalUpdated name="GoalUpdated_3"/>
                  <ComputePathThroughPoses name="ComputePathThroughPoses_2"/>
                  <TruncatePath name="TruncatePath_1"/>
                </ReactiveSequence>
              </RateController>
              <ReactiveSequence name="SmoothPath">
                <GoalUpdated name="GoalUpdated_4"/>
                <SmoothPath name="SmoothPath_3"/>
              </ReactiveSequence>
              <RateController hz="20.0" name="FollowPath">
                <ReactiveSequence name="FollowPathWithCancel">
                  <GoalUpdated name="GoalUpdated_2"/>
                  <FollowPath name="FollowPath_2"/>
                </ReactiveSequence>
              </RateController>
            </PipelineSequence>
            <ClearEntireCostmap name="ClearLocalCostmap" service_name="local_costmap/clear_entirely_local_costmap"/>
          </RecoveryNode>
          <RecoveryNode number_of_retries="2" name="FollowPathRecovery">
            <AlwaysSuccess name="AssistedTeleop"/>
            <ClearEntireCostmap name="ClearLocalCostmap_2" service_name="local_costmap/clear_entirely_local_costmap"/>
          </RecoveryNode>
        </SequenceStar>
        <ReactiveSequence name="OnExceptionRecovery">
          <RemovePassedGoals name="RemovePassedGoals"/>
          <RecoveryNode number_of_retries="1" name="SpinRecovery">
            <Spin name="Spin_2"/>
            <ClearEntireCostmap name="ClearLocalCostmap_3" service_name="local_costmap/clear_entirely_local_costmap"/>
          </RecoveryNode>
          <RecoveryNode number_of_retries="1" name="BackupRecovery">
            <BackUp name="BackUp_2"/>
            <ClearEntireCostmap name="ClearLocalCostmap_4" service_name="local_costmap/clear_entirely_local_costmap"/>
          </RecoveryNode>
          <RecoveryNode number_of_retries="1" name="WaitRecovery">
            <Wait name="Wait_2" wait_duration="5"/>
            <ClearEntireCostmap name="ClearLocalCostmap_5" service_name="local_costmap/clear_entirely_local_costmap"/>
          </RecoveryNode>
        </ReactiveSequence>
      </Fallback>
    </SequenceStar>
  </BehaviorTree>
</root>
```

## Performance Optimization

### Costmap Optimization
- Adjust resolution based on robot size and environment complexity
- Use appropriate robot radius for collision checking
- Optimize update frequencies for real-time performance

### Planning Parameters
- Tune planner tolerances for humanoid kinematics
- Adjust inflation radius for safety margins
- Configure obstacle clearance for step height limitations

### Controller Tuning
- Adjust velocity limits for humanoid walking speeds
- Configure acceleration profiles for stable walking
- Fine-tune controller parameters for smooth motion

## Integration with Isaac ROS

### Sensor Integration
Isaac ROS sensors can feed directly into Nav2:

```python
# Example: Isaac ROS depth image to Nav2 costmap
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import OccupancyGrid
import numpy as np
import cv2

class IsaacDepthToCostmap(Node):
    def __init__(self):
        super().__init__('isaac_depth_to_costmap')

        # Subscribe to Isaac ROS depth data
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_rect_raw',
            self.depth_callback,
            10
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/depth/camera_info',
            self.camera_info_callback,
            10
        )

        # Publish to costmap
        self.costmap_pub = self.create_publisher(
            OccupancyGrid,
            '/local_costmap/costmap',
            10
        )

    def depth_callback(self, msg):
        """Process depth image and convert to costmap"""
        # Convert ROS image to OpenCV
        depth_image = self.ros_image_to_cv2(msg)

        # Process depth data to detect obstacles
        obstacle_map = self.process_depth_for_obstacles(depth_image)

        # Convert to occupancy grid
        occupancy_grid = self.create_occupancy_grid(obstacle_map)

        # Publish to costmap
        self.costmap_pub.publish(occupancy_grid)

    def process_depth_for_obstacles(self, depth_image):
        """Detect obstacles from depth data"""
        # Threshold depth values to identify obstacles
        obstacle_threshold = 1.0  # meters
        obstacles = depth_image < obstacle_threshold

        # Convert to binary costmap
        costmap = np.zeros_like(depth_image, dtype=np.uint8)
        costmap[obstacles] = 100  # Mark as occupied

        return costmap
```

## Troubleshooting

### Common Issues
1. **Path Planning Failures**: Check map resolution and inflation parameters
2. **Local Minima**: Use behavior trees with recovery behaviors
3. **Footstep Planning**: Verify step spacing and height parameters
4. **Stability Issues**: Adjust walking gait parameters for balance

### Performance Monitoring
```bash
# Monitor navigation performance
ros2 run nav2_util nav2_saver --save-map my_map

# Check costmap updates
ros2 run rqt_plot rqt_plot /local_costmap/costmap_updates

# Monitor navigation status
ros2 topic echo /navigation/status
```

## Best Practices

### 1. Parameter Tuning
- Start with default parameters and adjust gradually
- Test in simulation before real-world deployment
- Document working parameters for different environments

### 2. Safety Considerations
- Implement proper obstacle detection and avoidance
- Use appropriate safety margins in costmap inflation
- Include recovery behaviors for exceptional situations

### 3. Humanoid-Specific Configurations
- Account for step height and spacing limitations
- Consider balance and stability constraints
- Adapt velocity profiles for humanoid walking

## Learning Objectives

After completing this section, you should understand:
- The architecture and components of Navigation2
- How to configure Nav2 for humanoid robot navigation
- The role of behavior trees in navigation
- How to integrate Isaac ROS sensors with Nav2
- Performance optimization techniques for navigation systems

## Next Steps

Continue to learn about [Reinforcement Learning for Robotics](./rl-sim-to-real) to understand how to train robot behaviors using machine learning approaches.