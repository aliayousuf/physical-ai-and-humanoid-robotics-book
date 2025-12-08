---
title: "Python Control Loop Example"
description: "Implementing control systems in Python using rclpy with feedback and error handling"
---

# Python Control Loop Example

## Overview

This example demonstrates how to implement a control system in Python using ROS 2 and rclpy. We'll create a simple feedback control loop that could be used for robot navigation, motor control, or other robotic applications. The example includes proper error handling, parameter configuration, and real-time performance considerations.

## Complete Control Loop Example

Create `my_robot_package/my_robot_package/control_loop_node.py`:

```python
#!/usr/bin/env python3

"""
Control loop example demonstrating feedback control with ROS 2.
This example implements a simple PID controller for maintaining a desired position.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
import time
import math


class ControlLoopNode(Node):
    """
    A control loop node that implements feedback control.
    This example uses a simple PID controller to maintain a desired position.
    """
    def __init__(self):
        super().__init__('control_loop_node')

        # Declare parameters with default values
        self.declare_parameter('control_rate', 50.0)  # Hz
        self.declare_parameter('kp', 1.0)  # Proportional gain
        self.declare_parameter('ki', 0.1)  # Integral gain
        self.declare_parameter('kd', 0.05)  # Derivative gain
        self.declare_parameter('max_output', 10.0)  # Maximum output value
        self.declare_parameter('min_output', -10.0)  # Minimum output value
        self.declare_parameter('desired_position', 5.0)  # Desired position

        # Get parameter values
        self.control_rate = self.get_parameter('control_rate').value
        self.kp = self.get_parameter('kp').value
        self.ki = self.get_parameter('ki').value
        self.kd = self.get_parameter('kd').value
        self.max_output = self.get_parameter('max_output').value
        self.min_output = self.get_parameter('min_output').value
        self.desired_position = self.get_parameter('desired_position').value

        # PID controller variables
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = self.get_clock().now()

        # Current position (for simulation purposes)
        self.current_position = 0.0
        self.current_velocity = 0.0

        # Create publishers and subscribers
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Publisher for control output (e.g., velocity commands)
        self.control_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)

        # Publisher for current position (for monitoring)
        self.position_pub = self.create_publisher(Float64, 'current_position', qos_profile)

        # Publisher for desired position (for monitoring)
        self.desired_pub = self.create_publisher(Float64, 'desired_position', qos_profile)

        # Subscriber for external position updates (if available)
        self.position_sub = self.create_subscription(
            Float64,
            'actual_position',
            self.position_callback,
            qos_profile
        )

        # Create timer for control loop
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        # Create timer for publishing status (at lower rate)
        self.status_timer = self.create_timer(0.1, self.publish_status)

        self.get_logger().info(f'Control loop node initialized with rate: {self.control_rate} Hz')
        self.get_logger().info(f'PID gains - P: {self.kp}, I: {self.ki}, D: {self.kd}')

    def position_callback(self, msg):
        """
        Callback for receiving external position updates.
        """
        try:
            self.current_position = msg.data
            self.get_logger().debug(f'Received position update: {self.current_position}')
        except Exception as e:
            self.get_logger().error(f'Error in position callback: {e}')

    def compute_pid(self, error, dt):
        """
        Compute PID control output.

        Args:
            error: Current error (desired - actual)
            dt: Time delta since last computation

        Returns:
            Control output value
        """
        # Proportional term
        proportional = self.kp * error

        # Integral term (with anti-windup)
        self.integral += error * dt
        # Prevent integral windup
        integral_saturation = (self.max_output - self.min_output) * 0.5
        self.integral = max(min(self.integral, integral_saturation), -integral_saturation)

        # Derivative term
        derivative = 0.0
        if dt > 0:
            derivative = self.kd * (error - self.previous_error) / dt

        # Compute output
        output = proportional + (self.ki * self.integral) + derivative

        # Saturate output
        output = max(min(output, self.max_output), self.min_output)

        # Store current error for next derivative calculation
        self.previous_error = error

        return output

    def control_loop(self):
        """
        Main control loop executed at regular intervals.
        """
        try:
            # Get current time
            current_time = self.get_clock().now()
            dt = (current_time - self.previous_time).nanoseconds / 1e9  # Convert to seconds

            # Update previous time
            self.previous_time = current_time

            # For this example, we'll simulate the current position
            # In a real system, this would come from sensor feedback
            if dt > 0:
                # Simple simulation: update position based on velocity
                self.current_position += self.current_velocity * dt

            # Calculate error
            error = self.desired_position - self.current_position

            # Compute control output using PID
            control_output = self.compute_pid(error, dt)

            # Update simulated velocity based on control output
            # This is a simple model - in reality, this would depend on system dynamics
            self.current_velocity = control_output * 0.1  # Scale factor for simulation

            # Create and publish control command
            cmd_msg = Twist()
            cmd_msg.linear.x = control_output  # Forward velocity
            cmd_msg.angular.z = 0.0  # No rotation for this example

            self.control_pub.publish(cmd_msg)

            # Log control information periodically
            if int(current_time.nanoseconds / 1e9) % 5 == 0:  # Every 5 seconds
                self.get_logger().info(
                    f'Error: {error:.3f}, Output: {control_output:.3f}, '
                    f'Position: {self.current_position:.3f}'
                )

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')
            # In a real system, you might want to implement safety measures here

    def publish_status(self):
        """
        Publish status information at a lower rate.
        """
        try:
            # Publish current position
            pos_msg = Float64()
            pos_msg.data = self.current_position
            self.position_pub.publish(pos_msg)

            # Publish desired position
            desired_msg = Float64()
            desired_msg.data = self.desired_position
            self.desired_pub.publish(desired_msg)

        except Exception as e:
            self.get_logger().error(f'Error in status publishing: {e}')

    def reset_controller(self):
        """
        Reset PID controller internal variables.
        """
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = self.get_clock().now()
        self.get_logger().info('Controller reset')


def main(args=None):
    """
    Main function to initialize and run the control loop node.
    """
    rclpy.init(args=args)

    control_loop_node = ControlLoopNode()

    try:
        # Keep the node running and processing callbacks
        rclpy.spin(control_loop_node)
    except KeyboardInterrupt:
        control_loop_node.get_logger().info('KeyboardInterrupt, shutting down.')
    except Exception as e:
        control_loop_node.get_logger().error(f'Unexpected error: {e}')
    finally:
        # Clean up the node
        control_loop_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Advanced Control Loop with Multiple Inputs

Create `my_robot_package/my_robot_package/advanced_control_loop.py` for a more complex example:

```python
#!/usr/bin/env python3

"""
Advanced control loop example with multiple inputs and safety features.
This example demonstrates a more sophisticated control system with safety checks.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64, Bool
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import LaserScan
import numpy as np
import time


class AdvancedControlLoopNode(Node):
    """
    Advanced control loop with multiple inputs and safety features.
    """
    def __init__(self):
        super().__init__('advanced_control_loop')

        # Declare parameters
        self.declare_parameter('control_rate', 30.0)  # Hz
        self.declare_parameter('safety_distance', 0.5)  # meters
        self.declare_parameter('max_linear_vel', 1.0)  # m/s
        self.declare_parameter('max_angular_vel', 1.0)  # rad/s
        self.declare_parameter('kp_linear', 1.0)
        self.declare_parameter('kp_angular', 2.0)

        # Get parameter values
        self.control_rate = self.get_parameter('control_rate').value
        self.safety_distance = self.get_parameter('safety_distance').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.kp_angular = self.get_parameter('kp_angular').value

        # State variables
        self.current_linear_error = 0.0
        self.current_angular_error = 0.0
        self.obstacle_detected = False
        self.emergency_stop = False
        self.target_reached = False

        # Laser scan data
        self.laser_ranges = []

        # QoS profile
        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', qos_profile)
        self.safety_status_pub = self.create_publisher(Bool, 'safety_status', qos_profile)

        # Subscribers
        self.laser_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.laser_callback,
            qos_profile
        )

        self.target_sub = self.create_subscription(
            Vector3,
            'target_position',
            self.target_callback,
            qos_profile
        )

        # Control timer
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        self.get_logger().info('Advanced control loop node initialized')

    def laser_callback(self, msg):
        """
        Callback for laser scan data.
        """
        try:
            self.laser_ranges = list(msg.ranges)
            # Check for obstacles within safety distance
            valid_ranges = [r for r in self.laser_ranges if msg.range_min < r < msg.range_max]
            if valid_ranges:
                min_distance = min(valid_ranges)
                self.obstacle_detected = min_distance < self.safety_distance
        except Exception as e:
            self.get_logger().error(f'Error in laser callback: {e}')

    def target_callback(self, msg):
        """
        Callback for target position.
        """
        try:
            # For this example, assume Vector3 represents target position
            # In practice, you might use Pose or Point messages
            self.current_linear_error = msg.x  # Linear distance to target
            self.current_angular_error = msg.y  # Angular error to target
        except Exception as e:
            self.get_logger().error(f'Error in target callback: {e}')

    def compute_control_output(self):
        """
        Compute control output based on current state.
        """
        # Safety check
        if self.emergency_stop or self.obstacle_detected:
            return Twist()  # Stop the robot

        # Compute linear and angular velocities
        linear_vel = self.kp_linear * self.current_linear_error
        angular_vel = self.kp_angular * self.current_angular_error

        # Saturate velocities
        linear_vel = max(min(linear_vel, self.max_linear_vel), -self.max_linear_vel)
        angular_vel = max(min(angular_vel, self.max_angular_vel), -self.max_angular_vel)

        # Create Twist message
        cmd_msg = Twist()
        cmd_msg.linear.x = linear_vel
        cmd_msg.angular.z = angular_vel

        return cmd_msg

    def control_loop(self):
        """
        Main control loop with safety features.
        """
        try:
            # Compute control output
            cmd_msg = self.compute_control_output()

            # Publish safety status
            safety_msg = Bool()
            safety_msg.data = not (self.emergency_stop or self.obstacle_detected)
            self.safety_status_pub.publish(safety_msg)

            # Publish command
            self.cmd_pub.publish(cmd_msg)

            # Log information
            self.get_logger().debug(
                f'Linear error: {self.current_linear_error:.3f}, '
                f'Angular error: {self.current_angular_error:.3f}, '
                f'Linear vel: {cmd_msg.linear.x:.3f}, '
                f'Angular vel: {cmd_msg.angular.z:.3f}, '
                f'Obstacle: {self.obstacle_detected}'
            )

        except Exception as e:
            self.get_logger().error(f'Error in control loop: {e}')

    def emergency_stop_callback(self, msg):
        """
        Callback for emergency stop.
        """
        self.emergency_stop = msg.data
        if self.emergency_stop:
            self.get_logger().warn('Emergency stop activated!')
        else:
            self.get_logger().info('Emergency stop cleared')


def main(args=None):
    """
    Main function for advanced control loop.
    """
    rclpy.init(args=args)

    advanced_control_node = AdvancedControlLoopNode()

    try:
        rclpy.spin(advanced_control_node)
    except KeyboardInterrupt:
        advanced_control_node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        advanced_control_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Package Configuration

Update the `setup.py` file to include the new control loop executables:

```python
from setuptools import setup

package_name = 'my_robot_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', ['urdf/my_robot.urdf']),
        ('share/' + package_name + '/urdf', ['urdf/my_robot.xacro']),
        ('share/' + package_name + '/rviz', ['rviz/robot_display.rviz']),
        ('share/' + package_name + '/launch', ['launch/display_robot.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Example ROS 2 package with control systems',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.publisher_member_function:main',
            'listener = my_robot_package.subscriber_member_function:main',
            'control_loop = my_robot_package.control_loop_node:main',
            'advanced_control = my_robot_package.advanced_control_loop:main',
        ],
    },
)
```

Make the Python files executable:

```bash
chmod +x my_robot_package/my_robot_package/control_loop_node.py
chmod +x my_robot_package/my_robot_package/advanced_control_loop.py
```

## Launch File for Control Loop

Create `my_robot_package/launch/control_loop.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    control_rate = DeclareLaunchArgument(
        'control_rate',
        default_value='50.0',
        description='Control loop rate in Hz'
    )

    kp = DeclareLaunchArgument(
        'kp',
        default_value='1.0',
        description='Proportional gain for PID controller'
    )

    ki = DeclareLaunchArgument(
        'ki',
        default_value='0.1',
        description='Integral gain for PID controller'
    )

    kd = DeclareLaunchArgument(
        'kd',
        default_value='0.05',
        description='Derivative gain for PID controller'
    )

    # Launch configuration
    control_rate_config = LaunchConfiguration('control_rate')
    kp_config = LaunchConfiguration('kp')
    ki_config = LaunchConfiguration('ki')
    kd_config = LaunchConfiguration('kd')

    # Control loop node
    control_loop_node = Node(
        package='my_robot_package',
        executable='control_loop',
        name='control_loop_node',
        parameters=[
            {'control_rate': control_rate_config},
            {'kp': kp_config},
            {'ki': ki_config},
            {'kd': kd_config},
        ],
        output='screen'
    )

    # RViz node for visualization
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=[
            '-d',
            PathJoinSubstitution([
                FindPackageShare('my_robot_package'),
                'rviz',
                'robot_display.rviz'
            ])
        ],
        output='screen'
    )

    return LaunchDescription([
        control_rate,
        kp,
        ki,
        kd,
        # Launch RViz first
        rviz_node,
        # Then launch control loop after a short delay
        TimerAction(
            period=2.0,
            actions=[control_loop_node]
        )
    ])
```

## Building and Running

After creating the control loop nodes, rebuild your package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

### Running the Basic Control Loop:
```bash
ros2 run my_robot_package control_loop
```

### Running with Parameters:
```bash
ros2 run my_robot_package control_loop --ros-args -p kp:=2.0 -p ki:=0.2 -p control_rate:=100.0
```

### Using the Launch File:
```bash
ros2 launch my_robot_package control_loop.launch.py kp:=1.5 ki:=0.15
```

## Understanding Control Loop Components

### 1. Timing and Rate Control
```python
# Create timer for consistent control rate
self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
```

### 2. PID Controller Implementation
```python
def compute_pid(self, error, dt):
    # Proportional: responds to current error
    proportional = self.kp * error

    # Integral: responds to accumulated past error
    self.integral += error * dt

    # Derivative: responds to rate of change of error
    derivative = self.kd * (error - self.previous_error) / dt

    # Combine all terms
    output = proportional + (self.ki * self.integral) + derivative
    return output
```

### 3. Safety Features
```python
# Parameter bounds checking
output = max(min(output, self.max_output), self.min_output)

# Integral windup prevention
integral_saturation = (self.max_output - self.min_output) * 0.5
self.integral = max(min(self.integral, integral_saturation), -integral_saturation)
```

## Real-Time Performance Considerations

### 1. Deterministic Execution
```python
# Ensure consistent timing
current_time = self.get_clock().now()
dt = (current_time - self.previous_time).nanoseconds / 1e9
```

### 2. Computational Efficiency
- Keep control loop computations simple
- Avoid complex operations in the main loop
- Use lookup tables for complex calculations when possible

### 3. Message Rate Management
```python
# Use appropriate QoS settings for real-time performance
qos_profile = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
```

## Common Control Loop Issues and Solutions

### 1. Tuning PID Parameters
- Start with low gains and gradually increase
- Use Ziegler-Nichols method for initial tuning
- Consider system-specific tuning methods

### 2. Timing Inconsistencies
- Use ROS time instead of system time
- Monitor actual control rate vs. desired rate
- Account for processing delays

### 3. Sensor Noise
- Implement filtering for noisy sensor data
- Use appropriate low-pass filters
- Consider sensor fusion techniques

## Learning Objectives

After completing this example, you should be able to:
- Implement feedback control loops in Python using ROS 2
- Design PID controllers with proper parameter tuning
- Handle real-time performance requirements
- Implement safety features in control systems
- Structure control code for maintainability and reliability

## Next Steps

Continue to learn about [Module 1 Learning Outcomes](../outcomes) to review what you've learned and how to apply it.