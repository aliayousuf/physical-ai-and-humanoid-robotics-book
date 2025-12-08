---
title: "Manipulation Control Example"
description: "Complete example demonstrating robot manipulation control with VLA integration"
---

# Manipulation Control Example

## Overview

This example demonstrates how to implement sophisticated manipulation control in Vision-Language-Action systems. We'll build a complete manipulation pipeline that can grasp, transport, and place objects based on voice commands, with proper safety monitoring and error handling.

## Complete Implementation

### 1. Manipulation Control System

```python
import numpy as np
import math
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class GraspType(Enum):
    """Types of grasps that can be performed"""
    PINCH = "pinch"
    POWER = "power"
    PRECISION = "precision"
    SUCTION = "suction"

class ManipulationState(Enum):
    """Current state of manipulation system"""
    IDLE = "idle"
    APPROACHING = "approaching"
    GRASPING = "grasping"
    HOLDING = "holding"
    TRANSPORTING = "transporting"
    PLACING = "placing"
    ERROR = "error"

@dataclass
class ObjectProperties:
    """Properties of an object for manipulation planning"""
    name: str
    weight: float  # in kg
    dimensions: Tuple[float, float, float]  # width, height, depth in meters
    center_of_mass: Tuple[float, float, float]  # relative to object center
    grasp_points: List[Tuple[float, float, float]]  # potential grasp points
    material: str  # "plastic", "metal", "glass", "fabric", etc.
    fragile: bool
    graspable: bool

@dataclass
class ManipulationAction:
    """Represents a manipulation action to be executed"""
    action_type: str  # 'grasp', 'transport', 'place', 'release'
    object_name: str
    target_pose: Tuple[float, float, float, float, float, float]  # x, y, z, roll, pitch, yaw
    grasp_type: GraspType
    grasp_point: Optional[Tuple[float, float, float]] = None
    force_limit: float = 50.0  # Newtons
    speed: float = 0.1  # m/s

class ManipulationController:
    """Controls robot manipulation operations"""

    def __init__(self):
        self.state = ManipulationState.IDLE
        self.held_object: Optional[str] = None
        self.held_object_properties: Optional[ObjectProperties] = None
        self.robot_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.arm_joints = np.zeros(6)  # 6 DOF arm
        self.gripper_position = 0.0  # 0.0 = open, 1.0 = closed
        self.object_database = self._initialize_object_database()

    def _initialize_object_database(self) -> Dict[str, ObjectProperties]:
        """Initialize database of known object properties"""
        return {
            'cup': ObjectProperties(
                name='cup',
                weight=0.2,
                dimensions=(0.08, 0.1, 0.08),
                center_of_mass=(0.0, 0.0, 0.0),
                grasp_points=[(0.0, 0.05, 0.05)],  # Handle location
                material='ceramic',
                fragile=True,
                graspable=True
            ),
            'bottle': ObjectProperties(
                name='bottle',
                weight=0.5,
                dimensions=(0.07, 0.25, 0.07),
                center_of_mass=(0.0, 0.0, 0.0),
                grasp_points=[(0.0, 0.0, 0.05)],  # Neck area
                material='plastic',
                fragile=False,
                graspable=True
            ),
            'book': ObjectProperties(
                name='book',
                weight=0.8,
                dimensions=(0.2, 0.02, 0.25),
                center_of_mass=(0.0, 0.0, 0.0),
                grasp_points=[(0.0, 0.0, 0.1), (0.0, 0.0, -0.1)],  # Spine area
                material='paper',
                fragile=False,
                graspable=True
            ),
            'ball': ObjectProperties(
                name='ball',
                weight=0.15,
                dimensions=(0.1, 0.1, 0.1),
                center_of_mass=(0.0, 0.0, 0.0),
                grasp_points=[(0.0, 0.0, 0.0)],
                material='rubber',
                fragile=False,
                graspable=True
            )
        }

    def plan_grasp(self, object_name: str, object_pose: Tuple[float, float, float, float, float, float]) -> Optional[ManipulationAction]:
        """Plan a grasp action for the specified object"""
        if object_name not in self.object_database:
            print(f"Unknown object: {object_name}")
            return None

        obj_props = self.object_database[object_name]
        if not obj_props.graspable:
            print(f"Object {object_name} is not graspable")
            return None

        # Determine appropriate grasp type based on object properties
        grasp_type = self._select_grasp_type(obj_props)

        # Select grasp point (for now, use the first available)
        grasp_point = obj_props.grasp_points[0] if obj_props.grasp_points else (0.0, 0.0, 0.0)

        # Calculate target pose for approach (above object)
        approach_pose = list(object_pose)
        approach_pose[2] += 0.1  # Approach 10cm above object

        return ManipulationAction(
            action_type='grasp',
            object_name=object_name,
            target_pose=tuple(approach_pose),
            grasp_type=grasp_type,
            grasp_point=grasp_point,
            force_limit=30.0 if obj_props.fragile else 80.0,
            speed=0.05
        )

    def plan_transport(self, destination_pose: Tuple[float, float, float, float, float, float]) -> ManipulationAction:
        """Plan transport action to destination"""
        if not self.held_object:
            raise ValueError("No object is currently held")

        return ManipulationAction(
            action_type='transport',
            object_name=self.held_object,
            target_pose=destination_pose,
            grasp_type=GraspType.POWER,  # Maintain power grasp during transport
            force_limit=50.0,
            speed=0.1
        )

    def plan_place(self, placement_pose: Tuple[float, float, float, float, float, float]) -> ManipulationAction:
        """Plan place action for held object"""
        if not self.held_object:
            raise ValueError("No object is currently held")

        return ManipulationAction(
            action_type='place',
            object_name=self.held_object,
            target_pose=placement_pose,
            grasp_type=GraspType.POWER,
            force_limit=10.0,  # Gentle release
            speed=0.05
        )

    def _select_grasp_type(self, obj_props: ObjectProperties) -> GraspType:
        """Select appropriate grasp type based on object properties"""
        if obj_props.fragile:
            return GraspType.PRECISION
        elif obj_props.dimensions[0] < 0.05 and obj_props.dimensions[1] < 0.05:
            return GraspType.PINCH
        else:
            return GraspType.POWER

    def execute_manipulation_action(self, action: ManipulationAction) -> bool:
        """Execute a single manipulation action"""
        print(f"Executing {action.action_type} action for {action.object_name}")

        if action.action_type == 'grasp':
            return self._execute_grasp(action)
        elif action.action_type == 'transport':
            return self._execute_transport(action)
        elif action.action_type == 'place':
            return self._execute_place(action)
        elif action.action_type == 'release':
            return self._execute_release(action)

        return False

    def _execute_grasp(self, action: ManipulationAction) -> bool:
        """Execute grasp action"""
        self.state = ManipulationState.APPROACHING
        print(f"Approaching {action.object_name}")

        # Move to approach position above object
        approach_pose = list(action.target_pose)
        approach_pose[2] -= 0.1  # Move down to object level
        success = self._move_to_pose(approach_pose, action.speed)
        if not success:
            self.state = ManipulationState.ERROR
            return False

        self.state = ManipulationState.GRASPING
        print(f"Grasping {action.object_name}")

        # Close gripper with appropriate force
        success = self._close_gripper(action.force_limit)
        if success:
            self.held_object = action.object_name
            self.held_object_properties = self.object_database[action.object_name]
            self.state = ManipulationState.HOLDING
            print(f"Successfully grasped {action.object_name}")
            return True
        else:
            self.state = ManipulationState.ERROR
            return False

    def _execute_transport(self, action: ManipulationAction) -> bool:
        """Execute transport action"""
        if not self.held_object:
            print("Error: No object to transport")
            return False

        self.state = ManipulationState.TRANSPORTING
        print(f"Transporting {self.held_object} to destination")

        # Move to destination with safe trajectory
        success = self._move_to_pose(action.target_pose, action.speed)
        if success:
            print(f"Successfully transported {self.held_object}")
            return True
        else:
            self.state = ManipulationState.ERROR
            return False

    def _execute_place(self, action: ManipulationAction) -> bool:
        """Execute place action"""
        if not self.held_object:
            print("Error: No object to place")
            return False

        self.state = ManipulationState.PLACING
        print(f"Placing {self.held_object}")

        # Move to placement position
        success = self._move_to_pose(action.target_pose, action.speed)
        if not success:
            self.state = ManipulationState.ERROR
            return False

        # Release object
        success = self._open_gripper()
        if success:
            self.held_object = None
            self.held_object_properties = None
            self.state = ManipulationState.IDLE
            print(f"Successfully placed {action.object_name}")
            return True
        else:
            self.state = ManipulationState.ERROR
            return False

    def _execute_release(self, action: ManipulationAction) -> bool:
        """Execute release action"""
        if self.held_object:
            success = self._open_gripper()
            if success:
                self.held_object = None
                self.held_object_properties = None
                self.state = ManipulationState.IDLE
                return True

        return False

    def _move_to_pose(self, pose: Tuple[float, float, float, float, float, float], speed: float) -> bool:
        """Simulate moving robot to specified pose"""
        # In a real system, this would interface with robot controllers
        print(f"Moving to pose: {pose[:3]} at speed {speed}m/s")
        time.sleep(1)  # Simulate movement time
        return True

    def _close_gripper(self, force_limit: float) -> bool:
        """Simulate closing gripper with force control"""
        print(f"Closing gripper with force limit {force_limit}N")
        time.sleep(0.5)  # Simulate grasp time
        return True

    def _open_gripper(self) -> bool:
        """Simulate opening gripper"""
        print("Opening gripper")
        time.sleep(0.5)  # Simulate release time
        return True

    def get_manipulation_state(self) -> ManipulationState:
        """Get current manipulation state"""
        return self.state

    def is_object_held(self) -> bool:
        """Check if an object is currently held"""
        return self.held_object is not None
```

### 2. Manipulation Planning System

```python
class ManipulationPlanner:
    """Plans complex manipulation sequences based on high-level commands"""

    def __init__(self):
        self.manipulation_controller = ManipulationController()

    def plan_manipulation_sequence(self, command_entities: Dict[str, str],
                                 object_poses: Dict[str, Tuple[float, float, float, float, float, float]],
                                 destination_poses: Dict[str, Tuple[float, float, float, float, float, float]]) -> List[ManipulationAction]:
        """Plan a sequence of manipulation actions"""
        sequence = []

        # Extract object and destination from command
        target_object = command_entities.get('object')
        destination = command_entities.get('destination', 'default')

        if not target_object:
            print("No target object specified in command")
            return []

        # Get object pose
        if target_object not in object_poses:
            print(f"Object {target_object} not found in environment")
            return []

        object_pose = object_poses[target_object]

        # Plan grasp action
        grasp_action = self.manipulation_controller.plan_grasp(target_object, object_pose)
        if grasp_action:
            sequence.append(grasp_action)

        # Get destination pose
        destination_pose = destination_poses.get(destination)
        if not destination_pose:
            print(f"Destination {destination} not found")
            # Use default destination
            destination_pose = (1.0, 0.0, 0.5, 0.0, 0.0, 0.0)

        # Plan transport action
        if grasp_action:  # Only add transport if grasp was successful
            transport_action = self.manipulation_controller.plan_transport(destination_pose)
            sequence.append(transport_action)

        # Plan place action
        place_action = self.manipulation_controller.plan_place(destination_pose)
        sequence.append(place_action)

        return sequence

    def validate_manipulation_sequence(self, sequence: List[ManipulationAction]) -> Tuple[bool, List[str]]:
        """Validate manipulation sequence for safety and feasibility"""
        issues = []

        # Check for object conflicts
        for i, action in enumerate(sequence):
            if action.action_type == 'grasp' and self.manipulation_controller.is_object_held():
                issues.append(f"Action {i}: Attempting to grasp while holding object")

            if action.action_type in ['transport', 'place'] and not self.manipulation_controller.is_object_held():
                issues.append(f"Action {i}: Attempting to transport/place without holding object")

            # Check force limits based on object fragility
            if action.action_type == 'grasp':
                obj_props = self.manipulation_controller.object_database.get(action.object_name)
                if obj_props and obj_props.fragile and action.force_limit > 50.0:
                    issues.append(f"Action {i}: Force limit too high for fragile object {action.object_name}")

        return len(issues) == 0, issues

    def optimize_manipulation_sequence(self, sequence: List[ManipulationAction]) -> List[ManipulationAction]:
        """Optimize manipulation sequence for efficiency"""
        optimized_sequence = []

        for action in sequence:
            # Adjust parameters based on object properties
            if action.action_type == 'grasp':
                obj_props = self.manipulation_controller.object_database.get(action.object_name)
                if obj_props:
                    # Adjust approach height based on object size
                    adjusted_pose = list(action.target_pose)
                    if obj_props.dimensions[1] > 0.1:  # If object is tall
                        adjusted_pose[2] += 0.05  # Higher approach
                    action = ManipulationAction(
                        action_type=action.action_type,
                        object_name=action.object_name,
                        target_pose=tuple(adjusted_pose),
                        grasp_type=action.grasp_type,
                        grasp_point=action.grasp_point,
                        force_limit=action.force_limit,
                        speed=action.speed
                    )

            optimized_sequence.append(action)

        return optimized_sequence
```

### 3. Safety and Force Control System

```python
class SafetyController:
    """Manages safety during manipulation operations"""

    def __init__(self):
        self.safety_limits = {
            'max_force': 100.0,  # Newtons
            'max_velocity': 0.2,  # m/s
            'max_acceleration': 0.5,  # m/s²
            'collision_threshold': 0.1  # meters
        }
        self.force_sensors = {}
        self.collision_avoidance = True

    def check_safety_before_action(self, action: ManipulationAction, environment_state: Dict) -> Tuple[bool, List[str]]:
        """Check if action is safe to execute"""
        issues = []

        # Check force limits
        if action.force_limit > self.safety_limits['max_force']:
            issues.append(f"Force limit {action.force_limit}N exceeds maximum {self.safety_limits['max_force']}N")

        # Check if destination is in collision zone
        if 'obstacles' in environment_state:
            for obstacle in environment_state['obstacles']:
                if self._is_pose_in_collision_zone(action.target_pose, obstacle):
                    issues.append(f"Destination pose in collision with obstacle at {obstacle['position']}")

        # Check if robot is in safe configuration
        if 'robot_config' in environment_state:
            if not self._is_robot_config_safe(environment_state['robot_config']):
                issues.append("Robot configuration is not safe for manipulation")

        return len(issues) == 0, issues

    def monitor_execution(self, action: ManipulationAction, current_state: Dict) -> Tuple[bool, List[str]]:
        """Monitor ongoing manipulation for safety violations"""
        issues = []

        # Monitor force sensors
        if 'force_sensors' in current_state:
            for joint, force in current_state['force_sensors'].items():
                if force > self.safety_limits['max_force']:
                    issues.append(f"Force limit exceeded at joint {joint}: {force}N")

        # Monitor velocity
        if 'velocity' in current_state:
            velocity = np.linalg.norm(current_state['velocity'])
            if velocity > self.safety_limits['max_velocity']:
                issues.append(f"Velocity limit exceeded: {velocity}m/s")

        # Check for unexpected collisions
        if 'collision_detected' in current_state and current_state['collision_detected']:
            issues.append("Unexpected collision detected")

        return len(issues) == 0, issues

    def _is_pose_in_collision_zone(self, pose: Tuple[float, float, float, float, float, float],
                                 obstacle: Dict) -> bool:
        """Check if pose is in collision zone with obstacle"""
        pose_pos = np.array(pose[:3])
        obstacle_pos = np.array(obstacle['position'])
        distance = np.linalg.norm(pose_pos - obstacle_pos)
        safe_distance = obstacle.get('radius', 0.1) + 0.05  # Add safety margin
        return distance < safe_distance

    def _is_robot_config_safe(self, config: Dict) -> bool:
        """Check if robot configuration is safe"""
        # Check joint limits
        if 'joint_positions' in config:
            for joint_pos in config['joint_positions']:
                if abs(joint_pos) > 3.14:  # Check if joint is in extreme position
                    return False

        # Check for self-collision
        if 'self_collision' in config and config['self_collision']:
            return False

        return True

    def emergency_stop(self):
        """Execute emergency stop procedure"""
        print("Emergency stop activated! Stopping all manipulation operations.")
        # In a real system, this would interface with robot controllers
        return True
```

### 4. Vision-Guided Manipulation System

```python
class VisionGuidedManipulator:
    """Integrates vision information with manipulation control"""

    def __init__(self):
        self.manipulation_planner = ManipulationPlanner()
        self.safety_controller = SafetyController()
        self.object_detector = None  # Would be connected to object detection system

    def execute_vision_guided_manipulation(self, command_entities: Dict[str, str],
                                         detected_objects: List[Dict],
                                         camera_intrinsics: Dict) -> bool:
        """Execute manipulation guided by vision information"""
        # Create object poses from detection results
        object_poses = self._create_object_poses(detected_objects, camera_intrinsics)
        destination_poses = self._create_destination_poses()

        # Plan manipulation sequence
        sequence = self.manipulation_planner.plan_manipulation_sequence(
            command_entities, object_poses, destination_poses
        )

        # Validate sequence
        is_valid, validation_issues = self.manipulation_planner.validate_manipulation_sequence(sequence)
        if not is_valid:
            print(f"Validation failed: {validation_issues}")
            return False

        # Optimize sequence
        optimized_sequence = self.manipulation_planner.optimize_manipulation_sequence(sequence)

        # Execute sequence with safety monitoring
        return self._execute_sequence_with_monitoring(optimized_sequence)

    def _create_object_poses(self, detected_objects: List[Dict], camera_intrinsics: Dict) -> Dict[str, Tuple]:
        """Convert detected objects to 3D poses"""
        object_poses = {}

        for obj in detected_objects:
            # Convert 2D image coordinates to 3D world coordinates
            # This is a simplified conversion - in practice, you'd use depth information
            x_2d, y_2d = obj['center']
            z_depth = obj.get('depth', 1.0)  # Default depth if not available

            # Convert to 3D using camera intrinsics
            x_3d = (x_2d - camera_intrinsics['cx']) * z_depth / camera_intrinsics['fx']
            y_3d = (y_2d - camera_intrinsics['cy']) * z_depth / camera_intrinsics['fy']
            z_3d = z_depth

            # Create pose (x, y, z, roll, pitch, yaw)
            pose = (x_3d, y_3d, z_3d, 0.0, 0.0, 0.0)
            object_poses[obj['name']] = pose

        return object_poses

    def _create_destination_poses(self) -> Dict[str, Tuple]:
        """Create predefined destination poses"""
        return {
            'table': (0.5, 0.0, 0.75, 0.0, 0.0, 0.0),
            'shelf': (0.8, 0.3, 1.2, 0.0, 0.0, 0.0),
            'bin': (0.2, -0.5, 0.6, 0.0, 0.0, 0.0),
            'kitchen': (1.0, 0.5, 0.75, 0.0, 0.0, 0.0),
            'default': (0.5, 0.0, 0.75, 0.0, 0.0, 0.0)
        }

    def _execute_sequence_with_monitoring(self, sequence: List[ManipulationAction]) -> bool:
        """Execute manipulation sequence with safety monitoring"""
        controller = self.manipulation_planner.manipulation_controller

        for i, action in enumerate(sequence):
            print(f"Executing action {i+1}/{len(sequence)}: {action.action_type} {action.object_name}")

            # Check safety before execution
            env_state = self._get_environment_state()
            is_safe, safety_issues = self.safety_controller.check_safety_before_action(action, env_state)
            if not is_safe:
                print(f"Safety check failed: {safety_issues}")
                return False

            # Execute action
            success = controller.execute_manipulation_action(action)
            if not success:
                print(f"Action {i+1} failed")
                return False

            # Monitor during execution
            current_state = self._get_current_robot_state()
            is_safe, monitoring_issues = self.safety_controller.monitor_execution(action, current_state)
            if not is_safe:
                print(f"Safety monitoring detected issues: {monitoring_issues}")
                self.safety_controller.emergency_stop()
                return False

        return True

    def _get_environment_state(self) -> Dict:
        """Get current environment state"""
        # In a real system, this would interface with sensors
        return {
            'obstacles': [],
            'robot_config': {'joint_positions': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        }

    def _get_current_robot_state(self) -> Dict:
        """Get current robot state"""
        # In a real system, this would interface with robot controllers
        return {
            'force_sensors': {'gripper': 10.0, 'wrist': 5.0},
            'velocity': [0.05, 0.02, 0.01],
            'collision_detected': False
        }
```

### 5. Complete Manipulation Control Example

```python
class ManipulationControlSystem:
    """Complete system for vision-language-action manipulation control"""

    def __init__(self):
        self.vision_guided_manipulator = VisionGuidedManipulator()
        self.manipulation_controller = ManipulationController()
        self.manipulation_planner = ManipulationPlanner()
        self.safety_controller = SafetyController()

    def execute_manipulation_command(self, command_entities: Dict[str, str],
                                   environment_state: Dict) -> bool:
        """Execute manipulation command based on voice command and environment"""
        print(f"Executing manipulation command: {command_entities}")

        # Get detected objects from environment state
        detected_objects = environment_state.get('detected_objects', [])
        camera_intrinsics = environment_state.get('camera_intrinsics', {
            'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5
        })

        # Execute vision-guided manipulation
        success = self.vision_guided_manipulator.execute_vision_guided_manipulation(
            command_entities, detected_objects, camera_intrinsics
        )

        return success

    def run_demo(self):
        """Run a demonstration of manipulation control"""
        print("Manipulation Control System Demo")
        print("=" * 40)

        # Example environment state
        environment_state = {
            'detected_objects': [
                {
                    'name': 'cup',
                    'center': (320, 240),  # Image coordinates
                    'depth': 0.8,  # Meters
                    'confidence': 0.9
                },
                {
                    'name': 'book',
                    'center': (400, 300),
                    'depth': 0.75,
                    'confidence': 0.85
                }
            ],
            'camera_intrinsics': {
                'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5
            }
        }

        # Example commands
        commands = [
            {'object': 'cup', 'action': 'grasp', 'destination': 'table'},
            {'object': 'book', 'action': 'move', 'destination': 'shelf'}
        ]

        for cmd in commands:
            print(f"\nProcessing command: {cmd}")
            success = self.execute_manipulation_command(cmd, environment_state)
            print(f"Command {'succeeded' if success else 'failed'}")

        print("\nDemo completed!")

    def simulate_manipulation_scenario(self):
        """Simulate a complete manipulation scenario"""
        print("Simulating manipulation scenario...")

        # Create a manipulation planner
        planner = self.manipulation_planner

        # Define object and destination poses
        object_poses = {
            'bottle': (0.3, 0.2, 0.75, 0.0, 0.0, 0.0),
            'cup': (0.4, -0.1, 0.75, 0.0, 0.0, 0.0)
        }

        destination_poses = {
            'table': (0.8, 0.0, 0.75, 0.0, 0.0, 0.0),
            'shelf': (0.9, 0.3, 1.2, 0.0, 0.0, 0.0)
        }

        # Plan and execute "move the bottle to the table"
        command_entities = {'object': 'bottle', 'destination': 'table'}
        sequence = planner.plan_manipulation_sequence(command_entities, object_poses, destination_poses)

        print(f"Planned sequence: {[action.action_type for action in sequence]}")

        # Validate and execute
        is_valid, issues = planner.validate_manipulation_sequence(sequence)
        if is_valid:
            print("Sequence validation passed")
            success = self.vision_guided_manipulator._execute_sequence_with_monitoring(sequence)
            print(f"Execution {'succeeded' if success else 'failed'}")
        else:
            print(f"Validation failed: {issues}")

# Example usage
def main():
    """Main function to demonstrate manipulation control"""
    system = ManipulationControlSystem()

    print("Manipulation Control System")
    print("1. Run basic demo")
    print("2. Run manipulation scenario simulation")

    choice = input("Choose option (1 or 2): ").strip()

    if choice == "1":
        system.run_demo()
    elif choice == "2":
        system.simulate_manipulation_scenario()
    else:
        print("Invalid choice. Running basic demo...")
        system.run_demo()

if __name__ == "__main__":
    main()
```

## ROS 2 Integration Example

### 1. Manipulation Control ROS 2 Node

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import JointTrajectoryControllerState
import json

class ManipulationControlROSNode(Node):
    def __init__(self):
        super().__init__('manipulation_control_node')

        # Initialize manipulation system
        self.manipulation_system = ManipulationControlSystem()

        # Publishers
        self.joint_trajectory_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10
        )
        self.gripper_command_pub = self.create_publisher(
            String, '/gripper_command', 10
        )
        self.status_pub = self.create_publisher(String, '/manipulation_status', 10)

        # Subscribers
        self.manipulation_cmd_sub = self.create_subscription(
            String, '/manipulation_commands', self.manipulation_command_callback, 10
        )
        self.object_detection_sub = self.create_subscription(
            String, '/object_detections', self.object_detection_callback, 10
        )
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Timer for safety monitoring
        self.safety_timer = self.create_timer(0.1, self.safety_monitor)

        self.get_logger().info('Manipulation Control ROS Node initialized')

    def manipulation_command_callback(self, msg):
        """Handle manipulation commands"""
        try:
            command_data = json.loads(msg.data)
            command_entities = command_data.get('entities', {})
            environment_state = command_data.get('environment', {})

            # Execute manipulation
            success = self.manipulation_system.execute_manipulation_command(
                command_entities, environment_state
            )

            # Publish status
            status_msg = String()
            status_msg.data = f"Manipulation {'succeeded' if success else 'failed'}"
            self.status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing manipulation command: {e}')

    def object_detection_callback(self, msg):
        """Handle object detection results"""
        try:
            detection_data = json.loads(msg.data)
            # Store detection results for manipulation planning
            self.current_detections = detection_data
        except Exception as e:
            self.get_logger().error(f'Error processing object detections: {e}')

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.current_joint_states = msg

    def safety_monitor(self):
        """Monitor manipulation safety"""
        # Check for safety violations
        if self.manipulation_system.manipulation_controller.get_manipulation_state() != ManipulationState.IDLE:
            # Check joint limits, force limits, etc.
            if self._check_safety_violations():
                self._execute_emergency_stop()

    def _check_safety_violations(self) -> bool:
        """Check for safety violations"""
        # Implement safety checks
        return False

    def _execute_emergency_stop(self):
        """Execute emergency stop"""
        self.get_logger().error('Safety violation detected! Stopping manipulation.')
        # Publish stop trajectory
        stop_trajectory = JointTrajectory()
        stop_trajectory.joint_names = []  # Fill with actual joint names
        self.joint_trajectory_pub.publish(stop_trajectory)

    def send_joint_trajectory(self, trajectory_points: List[JointTrajectoryPoint]):
        """Send joint trajectory to robot"""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # Actual joint names
        trajectory_msg.points = trajectory_points
        self.joint_trajectory_pub.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ManipulationControlROSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Control Techniques

### 1. Impedance Control for Safe Manipulation

```python
class ImpedanceController:
    """Implements impedance control for safe and compliant manipulation"""

    def __init__(self):
        self.stiffness = np.diag([1000.0, 1000.0, 1000.0, 100.0, 100.0, 50.0])  # Cartesian stiffness
        self.damping = np.diag([2.0 * np.sqrt(1000.0), 2.0 * np.sqrt(1000.0), 2.0 * np.sqrt(1000.0),
                               2.0 * np.sqrt(100.0), 2.0 * np.sqrt(100.0), 2.0 * np.sqrt(50.0)])  # Critical damping
        self.mass = np.diag([1.0, 1.0, 1.0, 0.1, 0.1, 0.1])  # Effective mass matrix

    def calculate_impedance_force(self, desired_pose: np.ndarray, current_pose: np.ndarray,
                                desired_velocity: np.ndarray, current_velocity: np.ndarray) -> np.ndarray:
        """Calculate impedance control force"""
        # Position error
        pos_error = desired_pose[:3] - current_pose[:3]

        # Orientation error (using rotation vector representation)
        orientation_error = self._rotation_vector_error(desired_pose[3:], current_pose[3:])

        # Combined error
        error = np.concatenate([pos_error, orientation_error])

        # Velocity error
        vel_error = desired_velocity - current_velocity

        # Impedance force: F = M * (xdd_des - xdd_actual) + D * (xd_des - xd_actual) + K * (x_des - x_actual)
        # For position-based control: F = K * (x_des - x_actual) + D * (xd_des - xd_actual)
        impedance_force = self.stiffness @ error[:6] + self.damping @ vel_error

        return impedance_force

    def _rotation_vector_error(self, desired_orientation: np.ndarray, current_orientation: np.ndarray) -> np.ndarray:
        """Calculate rotation vector error between two orientations"""
        # Convert to rotation matrices and compute error
        # This is a simplified implementation
        return desired_orientation - current_orientation

    def adjust_impedance_for_task(self, task_type: str):
        """Adjust impedance parameters based on task requirements"""
        if task_type == 'delicate':
            # Low stiffness for fragile objects
            self.stiffness = np.diag([100.0, 100.0, 100.0, 50.0, 50.0, 25.0])
        elif task_type == 'rigid':
            # High stiffness for stable manipulation
            self.stiffness = np.diag([2000.0, 2000.0, 2000.0, 200.0, 200.0, 100.0])
        elif task_type == 'compliant':
            # Very compliant for contact tasks
            self.stiffness = np.diag([50.0, 50.0, 50.0, 25.0, 25.0, 10.0])
```

## Key Features

1. **Advanced Grasp Planning**: Selects appropriate grasp types based on object properties
2. **Safety Monitoring**: Comprehensive safety checks and emergency stop procedures
3. **Vision Integration**: Uses visual information to guide manipulation actions
4. **Force Control**: Implements impedance control for safe interaction
5. **ROS 2 Integration**: Example of integration with ROS 2 robotics framework
6. **Error Handling**: Robust error detection and recovery mechanisms

## Learning Outcomes

After implementing this example, you should understand:

- How to plan and execute complex manipulation sequences
- Techniques for safe and compliant manipulation control
- Integration of vision systems with manipulation planning
- Force control and impedance control concepts
- Safety considerations in robotic manipulation
- ROS 2 integration patterns for manipulation systems

## Next Steps

This example can be extended with:

- Advanced grasp synthesis algorithms
- Learning-based manipulation policies
- Multi-fingered robotic hands with complex grasps
- Integration with whole-body motion planning
- Advanced force control for assembly tasks
- Machine learning for grasp success prediction