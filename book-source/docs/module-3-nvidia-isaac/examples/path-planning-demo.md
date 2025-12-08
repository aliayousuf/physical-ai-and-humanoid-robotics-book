---
title: "Path Planning Demo"
description: "Implementing humanoid path planning using Nav2 and Isaac Sim"
---

# Path Planning Demo

## Overview

This example demonstrates implementing path planning for humanoid robots using Navigation2 (Nav2) with Isaac Sim for simulation and validation. Humanoid path planning presents unique challenges compared to wheeled robots due to their bipedal nature and stability constraints.

## Prerequisites

Before implementing path planning, ensure you have:
- Completed Module 1 (ROS 2 fundamentals)
- Completed Module 2 (Digital Twin simulation)
- Basic understanding of path planning algorithms (A*, Dijkstra, RRT*)
- Isaac Sim with humanoid robot model
- Nav2 packages installed and configured

## Humanoid Path Planning Challenges

### 1. Kinematic Constraints
- Humanoid robots have complex joint constraints
- Balance and stability requirements during locomotion
- Discrete footstep planning instead of continuous motion
- Turning radius and step limitations

### 2. Dynamic Constraints
- Center of Mass (CoM) management
- Zero Moment Point (ZMP) stability
- Swing foot trajectory planning
- Upper body motion coordination

### 3. Navigation Considerations
- Step height limitations (cannot step over large obstacles)
- Slope limitations (steep inclines may be impassable)
- Surface stability requirements
- Multi-contact planning (hands for support)

## Nav2 Configuration for Humanoid Robots

### 1. Custom Costmap Parameters

Humanoid robots require specialized costmap parameters to account for their unique navigation requirements:

```yaml
# humanoid_nav2_params.yaml
local_costmap:
  local_costmap:
    ros__parameters:
      # Use a larger footprint to account for humanoid's stance
      robot_radius: 0.4  # Increased for humanoid stance
      footprint: [[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3]]  # Square footprint
      resolution: 0.025  # Higher resolution for precise planning
      update_frequency: 10.0  # Lower frequency for computation time
      publish_frequency: 5.0
      global_frame: odom
      robot_base_frame: base_link
      use_sim_time: True
      rolling_window: true
      width: 10.0  # Larger window for humanoid planning
      height: 10.0
      trinary_costmap: false
      track_unknown_space: false
      transform_tolerance: 0.5
      plugins: ["obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 1.8  # Humanoid height
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.0
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 2.0  # Adjusted for humanoid safety
        inflation_radius: 0.6     # Larger safety margin for humanoid
        inflate_unknown: false
        inflate_around_unknown: false

global_costmap:
  global_costmap:
    ros__parameters:
      # Use map frame for global costmap
      global_frame: map
      robot_base_frame: base_link
      update_frequency: 1.0
      publish_frequency: 1.0
      resolution: 0.05
      robot_radius: 0.4
      footprint: [[0.3, 0.3], [0.3, -0.3], [-0.3, -0.3], [-0.3, 0.3]]
      track_unknown_space: true
      use_sim_time: True
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: True
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 1.8
          clearing: True
          marking: True
          data_type: "LaserScan"
          raytrace_max_range: 5.0
          raytrace_min_range: 0.0
          obstacle_max_range: 4.0
          obstacle_min_range: 0.0
      static_layer:
        plugin: "nav2_costmap_2d::StaticLayer"
        map_subscribe_transient_local: True
      inflation_layer:
        plugin: "nav2_costmap_2d::InflationLayer"
        cost_scaling_factor: 2.0
        inflation_radius: 0.6
        inflate_unknown: true
        inflate_around_unknown: false
```

### 2. Footstep Planner Integration

For humanoid robots, integrate a footstep planner with Nav2:

```python
# footstep_planner.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidFootstepPlanner(Node):
    def __init__(self):
        super().__init__('humanoid_footstep_planner')

        # Subscribe to global path from Nav2
        self.global_path_sub = self.create_subscription(
            Path,
            '/plan',
            self.global_path_callback,
            10
        )

        # Publish discrete footsteps
        self.footstep_pub = self.create_publisher(
            Path,
            '/footsteps',
            10
        )

        # Publish visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/footstep_markers',
            10
        )

        # Footstep planning parameters
        self.step_spacing = 0.3  # Distance between footsteps
        self.step_height = 0.1   # Height to lift foot during stepping
        self.foot_width = 0.1    # Width of foot
        self.foot_length = 0.15  # Length of foot

        # Stability parameters
        self.support_polygon_radius = 0.2  # Radius for stable foot placement

        self.get_logger().info('Humanoid footstep planner initialized')

    def global_path_callback(self, msg):
        """Convert continuous path to discrete footsteps"""
        try:
            # Convert continuous path to discrete footsteps
            footsteps_path = self.discretize_path_to_footsteps(msg.poses)

            # Validate footsteps for stability
            validated_footsteps = self.validate_footsteps_for_stability(footsteps_path)

            # Publish footsteps
            self.footstep_pub.publish(validated_footsteps)

            # Publish visualization markers
            self.publish_footstep_markers(validated_footsteps)

        except Exception as e:
            self.get_logger().error(f'Error in footstep planning: {e}')

    def discretize_path_to_footsteps(self, poses):
        """Convert continuous path to discrete footsteps for humanoid"""
        footsteps = Path()
        footsteps.header = msg.header

        if len(poses) < 2:
            return footsteps

        # Start with initial pose
        if poses:
            footsteps.poses.append(poses[0])

        # Calculate cumulative distance and add footsteps at regular intervals
        cumulative_distance = 0.0
        prev_pose = poses[0]

        for i in range(1, len(poses)):
            current_pose = poses[i]

            # Calculate distance from previous pose
            dx = current_pose.pose.position.x - prev_pose.pose.position.x
            dy = current_pose.pose.position.y - prev_pose.pose.position.y
            dz = current_pose.pose.position.z - prev_pose.pose.position.z
            distance = np.sqrt(dx*dx + dy*dy + dz*dz)

            cumulative_distance += distance

            # Add footsteps at regular intervals based on step spacing
            while cumulative_distance >= self.step_spacing:
                # Interpolate to find the exact position for the next step
                remaining_distance = cumulative_distance - self.step_spacing
                interpolation_ratio = remaining_distance / distance
                final_ratio = 1.0 - interpolation_ratio

                # Interpolate position
                step_x = prev_pose.pose.position.x + dx * final_ratio
                step_y = prev_pose.pose.position.y + dy * final_ratio
                step_z = prev_pose.pose.position.z + dz * final_ratio

                # Interpolate orientation
                q1 = np.array([
                    prev_pose.pose.orientation.w,
                    prev_pose.pose.orientation.x,
                    prev_pose.pose.orientation.y,
                    prev_pose.pose.orientation.z
                ])

                q2 = np.array([
                    current_pose.pose.orientation.w,
                    current_pose.pose.orientation.x,
                    current_pose.pose.orientation.y,
                    current_pose.pose.orientation.z
                ])

                # Spherical linear interpolation (SLERP)
                interpolated_quat = self.slerp_quaternions(q1, q2, final_ratio)

                # Create step pose
                step_pose = PoseStamped()
                step_pose.header = msg.header
                step_pose.pose.position.x = step_x
                step_pose.pose.position.y = step_y
                step_pose.pose.position.z = step_z  # Adjust for foot height
                step_pose.pose.orientation.w = interpolated_quat[0]
                step_pose.pose.orientation.x = interpolated_quat[1]
                step_pose.pose.orientation.y = interpolated_quat[2]
                step_pose.pose.orientation.z = interpolated_quat[3]

                footsteps.poses.append(step_pose)
                cumulative_distance = remaining_distance

            prev_pose = current_pose

        return footsteps

    def validate_footsteps_for_stability(self, footsteps_path):
        """Validate footsteps for stability and navigability"""
        validated_steps = Path()
        validated_steps.header = footsteps_path.header

        for i, step in enumerate(footsteps_path.poses):
            # Check if step is on navigable terrain
            if self.is_navigable(step.pose.position):
                # Check stability of step placement
                if self.is_stable_placement(step.pose, footsteps_path.poses, i):
                    validated_steps.poses.append(step)
                else:
                    # Try to adjust step for stability
                    adjusted_step = self.adjust_for_stability(step, footsteps_path.poses, i)
                    if adjusted_step:
                        validated_steps.poses.append(adjusted_step)
            else:
                # Find alternative step location
                alternative_step = self.find_alternative_step(step, footsteps_path.poses, i)
                if alternative_step:
                    validated_steps.poses.append(alternative_step)

        return validated_steps

    def is_navigable(self, position):
        """Check if position is navigable for humanoid"""
        # This would check against costmap for obstacles, step height limits, etc.
        # For demo purposes, assume all positions are navigable
        return True

    def is_stable_placement(self, pose, all_steps, current_idx):
        """Check if footstep is stable based on support polygon"""
        # Check if this step maintains stability with previous steps
        if current_idx == 0:
            return True

        # Calculate center of mass relative to support polygon
        # This is a simplified check - real implementation would be more complex
        return True

    def adjust_for_stability(self, step, all_steps, current_idx):
        """Adjust step placement for better stability"""
        # Try to find a nearby stable placement
        # This would involve checking multiple nearby positions
        return step  # Return original for simplicity

    def find_alternative_step(self, step, all_steps, current_idx):
        """Find alternative step when original is not navigable"""
        # Find a nearby navigable position
        return step  # Return original for simplicity

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

    def publish_footstep_markers(self, footsteps_path):
        """Publish visualization markers for footsteps"""
        marker_array = MarkerArray()

        for i, step in enumerate(footsteps_path.poses):
            # Create marker for foot placement
            marker = Marker()
            marker.header = footsteps_path.header
            marker.ns = "footsteps"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose = step.pose
            # Slightly raise the marker to be above ground
            marker.pose.position.z += 0.05

            marker.scale.x = self.foot_length
            marker.scale.y = self.foot_width
            marker.scale.z = 0.02  # Thin cube for foot

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.8

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    footstep_planner = HumanoidFootstepPlanner()

    try:
        rclpy.spin(footstep_planner)
    except KeyboardInterrupt:
        footstep_planner.get_logger().info('Shutting down footstep planner')
    finally:
        footstep_planner.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac Sim Path Planning Integration

### 1. Simulation Environment Setup

Set up Isaac Sim environment for path planning validation:

```python
# isaac_sim_path_planning.py
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.navigation import NavigationGraph
from omni.isaac.core.utils.prims import get_prim_at_path
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class IsaacSimPathPlanning:
    def __init__(self):
        # Initialize Isaac Sim world
        self.world = World(stage_units_in_meters=1.0)

        # Initialize ROS
        rclpy.init()
        self.node = rclpy.create_node('isaac_sim_path_planning')

        # Publishers and subscribers for Nav2 integration
        self.goal_pub = self.node.create_publisher(PoseStamped, '/goal_pose', 10)
        self.path_sub = self.node.create_subscription(
            Path, '/plan', self.path_callback, 10
        )

        # Initialize navigation graph in Isaac Sim
        self.setup_navigation_graph()

        # Robot-specific parameters
        self.robot_position = np.array([0.0, 0.0, 0.0])
        self.robot_orientation = np.array([0.0, 0.0, 0.0, 1.0])

        self.get_logger().info('Isaac Sim path planning initialized')

    def setup_navigation_graph(self):
        """Setup navigation graph for path planning in Isaac Sim"""
        # Create navigation graph for the environment
        # This would typically be done automatically or manually
        # depending on the complexity of the environment
        pass

    def generate_navigation_goal(self, target_x, target_y, target_yaw=0.0):
        """Generate navigation goal for humanoid robot"""
        goal_msg = PoseStamped()
        goal_msg.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'map'

        goal_msg.pose.position.x = target_x
        goal_msg.pose.position.y = target_y
        goal_msg.pose.position.z = 0.0  # Ground level

        # Convert yaw to quaternion
        cy = np.cos(target_yaw * 0.5)
        sy = np.sin(target_yaw * 0.5)
        cq = np.cos(0.0 * 0.5)  # pitch
        sq = np.sin(0.0 * 0.5)
        cr = np.cos(0.0 * 0.5)  # roll
        sr = np.sin(0.0 * 0.5)

        goal_msg.pose.orientation.w = cr * cq * cy + sr * sq * sy
        goal_msg.pose.orientation.x = sr * cq * cy - cr * sq * sy
        goal_msg.pose.orientation.y = cr * sq * cy + sr * cq * sy
        goal_msg.pose.orientation.z = cr * cq * sy - sr * sq * cy

        # Publish goal
        self.goal_pub.publish(goal_msg)

        return goal_msg

    def path_callback(self, msg):
        """Process path plan from Nav2"""
        # This would typically trigger the robot to follow the path
        # In simulation, we might visualize the path or execute it
        self.visualize_path_in_simulation(msg)

    def visualize_path_in_simulation(self, path_msg):
        """Visualize the planned path in Isaac Sim"""
        # Create visual elements in Isaac Sim to show the path
        # This might involve creating path markers or animating the robot
        pass

    def execute_path_in_simulation(self, path_msg):
        """Execute the path in Isaac Sim with humanoid-specific constraints"""
        # This would implement the actual movement of the humanoid robot
        # following the planned path with proper footstep execution
        pass

    def validate_path_feasibility(self, path_msg):
        """Validate path feasibility for humanoid robot"""
        # Check if the path is feasible given humanoid constraints:
        # - Step height limitations
        # - Slope limitations
        # - Surface stability
        # - Balance requirements

        feasible = True
        for i in range(len(path_msg.poses) - 1):
            current_pos = np.array([
                path_msg.poses[i].pose.position.x,
                path_msg.poses[i].pose.position.y,
                path_msg.poses[i].pose.position.z
            ])

            next_pos = np.array([
                path_msg.poses[i+1].pose.position.x,
                path_msg.poses[i+1].pose.position.y,
                path_msg.poses[i+1].pose.position.z
            ])

            # Check step height difference
            height_diff = abs(next_pos[2] - current_pos[2])
            if height_diff > 0.15:  # Humanoid step height limit
                feasible = False
                break

            # Check step distance
            step_distance = np.linalg.norm(next_pos[:2] - current_pos[:2])
            if step_distance > 0.5:  # Maximum step distance
                feasible = False
                break

        return feasible

    def get_logger(self):
        """Helper to access logger"""
        return self.node.get_logger()
```

## Path Planning Algorithm Implementation

### 1. Custom Path Planner for Humanoids

Implement a custom path planner that considers humanoid-specific constraints:

```python
# humanoid_path_planner.py
import numpy as np
from scipy.spatial.distance import euclidean
import heapq
from typing import List, Tuple, Optional

class HumanoidPathPlanner:
    def __init__(self, map_resolution=0.05, step_limit=0.3, height_limit=0.15):
        self.map_resolution = map_resolution
        self.step_limit = step_limit  # Maximum step distance
        self.height_limit = height_limit  # Maximum step height

    def plan_path(self, start: Tuple[float, float, float],
                  goal: Tuple[float, float, float],
                  costmap: np.ndarray) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan path for humanoid robot considering step limitations

        Args:
            start: Starting position (x, y, z)
            goal: Goal position (x, y, z)
            costmap: 2D costmap with obstacle costs

        Returns:
            List of waypoints or None if no path found
        """
        # Convert to grid coordinates
        start_grid = self.world_to_grid(start)
        goal_grid = self.world_to_grid(goal)

        # Use A* with humanoid constraints
        path_grid = self.astar_with_constraints(start_grid, goal_grid, costmap)

        if path_grid is None:
            return None

        # Convert back to world coordinates
        path_world = [self.grid_to_world(pos) for pos in path_grid]

        # Smooth path considering humanoid dynamics
        smoothed_path = self.smooth_path_for_humanoid(path_world, costmap)

        return smoothed_path

    def astar_with_constraints(self, start: Tuple[int, int],
                              goal: Tuple[int, int],
                              costmap: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """A* path planning with humanoid step constraints"""
        rows, cols = costmap.shape
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),  # 4-connectivity
            (1, 1), (1, -1), (-1, 1), (-1, -1)  # 8-connectivity
        ]

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                    continue

                # Check if traversable
                if costmap[neighbor[0], neighbor[1]] >= 99:  # Obstacle
                    continue

                # Calculate tentative g_score
                step_cost = self.calculate_step_cost(current, neighbor, costmap)
                tentative_g_score = g_score[current] + step_cost

                # Check humanoid constraints
                if self.is_step_feasible(current, neighbor):
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)

                        # Add to open set if not already there
                        if neighbor not in [item[1] for item in open_set]:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def is_step_feasible(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if step is feasible for humanoid robot"""
        # Calculate step distance in world coordinates
        world_pos1 = self.grid_to_world(pos1)
        world_pos2 = self.grid_to_world(pos2)

        # Calculate 2D distance (ignore Z for step distance check)
        step_distance = euclidean(world_pos1[:2], world_pos2[:2])

        # Check step distance constraint
        if step_distance > self.step_limit:
            return False

        # Check height difference constraint
        height_diff = abs(world_pos2[2] - world_pos1[2])
        if height_diff > self.height_limit:
            return False

        return True

    def calculate_step_cost(self, pos1: Tuple[int, int], pos2: Tuple[int, int],
                           costmap: np.ndarray) -> float:
        """Calculate cost for a step considering humanoid constraints"""
        base_cost = euclidean(pos1, pos2)
        costmap_cost = costmap[pos2[0], pos2[1]]

        # Add penalties based on humanoid constraints
        humanoid_penalty = 0.0

        # If the step is near the maximum allowed distance, add penalty
        world_pos1 = self.grid_to_world(pos1)
        world_pos2 = self.grid_to_world(pos2)
        step_distance = euclidean(world_pos1[:2], world_pos2[:2])

        if step_distance > self.step_limit * 0.8:  # Close to limit
            humanoid_penalty += 5.0

        return base_cost + costmap_cost * 0.1 + humanoid_penalty

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Heuristic function for A*"""
        return euclidean(pos1, pos2)

    def world_to_grid(self, world_pos: Tuple[float, float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x, y, z = world_pos
        grid_x = int(x / self.map_resolution)
        grid_y = int(y / self.map_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> Tuple[float, float, float]:
        """Convert grid coordinates to world coordinates"""
        grid_x, grid_y = grid_pos
        x = grid_x * self.map_resolution
        y = grid_y * self.map_resolution
        z = 0.0  # Assume ground level for now
        return (x, y, z)

    def smooth_path_for_humanoid(self, path: List[Tuple[float, float, float]],
                                costmap: np.ndarray) -> List[Tuple[float, float, float]]:
        """Smooth path considering humanoid dynamics and stability"""
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]

        i = 0
        while i < len(path) - 2:
            # Try to find the furthest point that can be reached directly
            j = len(path) - 1

            while j > i + 1:
                if self.is_line_of_sight_feasible(path[i], path[j], costmap):
                    smoothed_path.append(path[j])
                    i = j - 1
                    break
                j -= 1

            if j == i + 1:  # No shortcut found, add next point
                smoothed_path.append(path[i + 1])
                i += 1

        # Add goal if not already added
        if smoothed_path[-1] != path[-1]:
            smoothed_path.append(path[-1])

        return smoothed_path

    def is_line_of_sight_feasible(self, start: Tuple[float, float, float],
                                 end: Tuple[float, float, float],
                                 costmap: np.ndarray) -> bool:
        """Check if line of sight is feasible considering humanoid constraints"""
        # This would implement a line-of-sight check with humanoid constraints
        # For simplicity, we'll use a basic Bresenham line algorithm with checks
        start_grid = self.world_to_grid(start)
        end_grid = self.world_to_grid(end)

        # Check all points along the line
        points = self.bresenham_line(start_grid[0], start_grid[1],
                                    end_grid[0], end_grid[1])

        for x, y in points:
            if not (0 <= x < costmap.shape[0] and 0 <= y < costmap.shape[1]):
                return False
            if costmap[x, y] >= 99:  # Obstacle
                return False

        # Check step constraints along the line
        step_distance = euclidean(start[:2], end[:2])
        if step_distance > self.step_limit * 2:  # Allow some flexibility for smoothing
            return False

        return True

    def bresenham_line(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Bresenham line algorithm implementation"""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        x, y = x1, y1

        while True:
            points.append((x, y))

            if x == x2 and y == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points
```

## Launch File for Path Planning Demo

Create a launch file to bring up the complete path planning system:

```python
# path_planning_demo_launch.py
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
            FindPackageShare('humanoid_navigation'),
            'config',
            'humanoid_nav2_params.yaml'
        ]),
        description='Full path to params file for navigation nodes'
    )

    # Navigation container with custom humanoid components
    navigation_container = ComposableNodeContainer(
        name='humanoid_navigation_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # Planner server with humanoid constraints
            ComposableNode(
                package='nav2_planner',
                plugin='nav2_planner::PlannerServer',
                name='planner_server',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),

            # Custom humanoid controller
            ComposableNode(
                package='humanoid_controller',
                plugin='humanoid_controller::FootstepController',
                name='footstep_controller',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),

            # Recovery server
            ComposableNode(
                package='nav2_recoveries',
                plugin='nav2_recoveries::RecoveryServer',
                name='recoveries_server',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),

            # BT navigator
            ComposableNode(
                package='nav2_bt_navigator',
                plugin='nav2_bt_navigator::BtNavigator',
                name='bt_navigator',
                parameters=[LaunchConfiguration('params_file'),
                           {'use_sim_time': LaunchConfiguration('use_sim_time')}],
                extra_arguments=[{'use_intra_process_comms': True}]
            ),
        ],
        output='screen'
    )

    # Footstep planner node
    footstep_planner_node = Node(
        package='humanoid_navigation',
        executable='footstep_planner',
        name='humanoid_footstep_planner',
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            ('/plan', '/global_plan'),
            ('/footsteps', '/humanoid_footsteps'),
        ],
        output='screen'
    )

    # Isaac Sim path planning integration
    isaac_sim_integration_node = Node(
        package='isaac_sim_path_planning',
        executable='isaac_sim_path_planning_node',
        name='isaac_sim_path_planning',
        parameters=[LaunchConfiguration('params_file')],
        output='screen'
    )

    # Lifecycle manager for navigation
    lifecycle_manager = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager',
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')},
                   {'autostart': True},
                   {'node_names': ['planner_server',
                                 'footstep_controller',
                                 'recoveries_server',
                                 'bt_navigator']}]
    )

    return LaunchDescription([
        use_sim_time,
        params_file,
        navigation_container,
        footstep_planner_node,
        isaac_sim_integration_node,
        lifecycle_manager
    ])
```

## Best Practices for Humanoid Path Planning

### 1. Stability Considerations
- Always verify Center of Mass (CoM) stays within support polygon
- Plan footsteps with adequate safety margins
- Consider dynamic balance during movement
- Account for upper body motion during locomotion

### 2. Obstacle Avoidance
- Use appropriate costmap inflation for humanoid safety
- Consider step height limitations when planning around obstacles
- Plan for multiple contact points if needed (using hands for support)

### 3. Performance Optimization
- Use appropriate path resolution for humanoid step size
- Implement efficient path smoothing algorithms
- Consider hierarchical planning (global coarse, local fine)
- Cache computed paths when possible

## Troubleshooting Common Issues

### 1. Path Planning Failures
- **Symptoms**: Nav2 fails to find valid paths
- **Solutions**:
  - Check costmap inflation parameters
  - Verify robot footprint configuration
  - Adjust step height and distance constraints

### 2. Navigation Instability
- **Symptoms**: Robot falls or becomes unstable during navigation
- **Solutions**:
  - Implement proper footstep validation
  - Add balance feedback control
  - Reduce navigation speed for stability

### 3. Computational Performance
- **Symptoms**: Slow path planning or high CPU usage
- **Solutions**:
  - Optimize path resolution
  - Use hierarchical planning approaches
  - Consider approximate algorithms for real-time applications

## Learning Objectives

After completing this example, you should understand:
- How to configure Nav2 for humanoid-specific navigation requirements
- The challenges of path planning for bipedal robots
- How to implement footstep planning for stable locomotion
- How to integrate Isaac Sim with Nav2 for validation
- Best practices for humanoid navigation systems

## Next Steps

Continue to learn about [Reinforcement Learning for Walk Cycles](../examples/rl-walk-cycle) to understand how to train stable walking patterns for humanoid robots using machine learning approaches.