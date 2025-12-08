---
title: "Launch Files Configuration"
description: "Understanding and creating launch files for complex ROS 2 system configurations"
---

# Launch Files Configuration

## Overview

Launch files in ROS 2 allow you to start multiple nodes with a single command and manage their configuration. They provide a way to define complex system setups with parameters, remappings, and conditional execution. Launch files are written in Python and use the `launch` package.

## Why Use Launch Files?

Launch files solve several challenges in robot system management:
- **Convenience**: Start multiple nodes with a single command
- **Configuration**: Set parameters and remappings for all nodes at once
- **Reusability**: Define common system configurations once and reuse
- **Flexibility**: Use arguments and conditionals for different scenarios
- **Organization**: Group related nodes together logically

## Launch File Structure

A basic launch file follows this structure:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Define launch arguments
    # Create nodes
    # Return launch description
    pass
```

## Creating a Simple Launch File

Create `my_robot_package/launch/simple_launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='simple_publisher',
            name='simple_publisher_node',
            parameters=[
                # Add parameters here
            ],
            remappings=[
                # Add remappings here
            ],
            output='screen'
        )
    ])
```

## Launch Arguments

Launch arguments allow you to parameterize your launch files:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    # Use launch configurations
    use_sim_time_config = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        use_sim_time,
        Node(
            package='my_robot_package',
            executable='simple_publisher',
            name='simple_publisher_node',
            parameters=[{'use_sim_time': use_sim_time_config}],
            output='screen'
        )
    ])
```

## Multiple Nodes in Launch Files

Launch files can start multiple nodes:

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_package',
            executable='simple_publisher',
            name='publisher_node',
            output='screen'
        ),
        Node(
            package='my_robot_package',
            executable='simple_subscriber',  # Assuming you have this
            name='subscriber_node',
            output='screen'
        )
    ])
```

## Parameters and Remappings

### Parameters
Parameters allow you to configure node behavior:

```python
Node(
    package='my_robot_package',
    executable='my_node',
    name='my_node',
    parameters=[
        {'param1': 'value1'},
        {'param2': 10},
        {'param3': True},
        # Load from file
        '/path/to/params.yaml'
    ],
    output='screen'
)
```

### Remappings
Remappings allow you to change topic/service names:

```python
Node(
    package='my_robot_package',
    executable='my_node',
    name='my_node',
    remappings=[
        ('original_topic', 'new_topic'),
        ('original_service', 'new_service')
    ],
    output='screen'
)
```

## Conditional Launch

Use conditions to start nodes based on arguments:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    debug_mode = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug mode'
    )

    debug_config = LaunchConfiguration('debug')

    debug_node = Node(
        package='my_robot_package',
        executable='debug_node',
        name='debug_node',
        condition=IfCondition(debug_config),
        output='screen'
    )

    return LaunchDescription([
        debug_mode,
        debug_node
    ])
```

## Loading Parameters from YAML

Create a YAML file for parameters:

`my_robot_package/config/robot_params.yaml`:
```yaml
my_robot_node:
  ros__parameters:
    param1: "value1"
    param2: 42
    param3: true
```

Use it in your launch file:
```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    params_file = DeclareLaunchArgument(
        'params_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_package'),
            'config',
            'robot_params.yaml'
        ]),
        description='Path to parameters file'
    )

    return LaunchDescription([
        params_file,
        Node(
            package='my_robot_package',
            executable='my_node',
            name='my_robot_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        )
    ])
```

## Complex Launch Example

Here's a more comprehensive example:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Launch arguments
    namespace = DeclareLaunchArgument(
        'namespace',
        default_value='my_robot',
        description='Robot namespace'
    )

    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )

    enable_logging = DeclareLaunchArgument(
        'enable_logging',
        default_value='true',
        description='Enable logging'
    )

    # Nodes
    robot_controller = Node(
        package='my_robot_package',
        executable='robot_controller',
        name='robot_controller',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        remappings=[
            ('cmd_vel', 'cmd_vel_unstamped')
        ],
        output='screen'
    )

    sensor_processor = Node(
        package='my_robot_package',
        executable='sensor_processor',
        name='sensor_processor',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        output='screen'
    )

    # Conditional logging node
    logging_node = Node(
        package='my_robot_package',
        executable='logger',
        name='logger',
        condition=IfCondition(LaunchConfiguration('enable_logging')),
        output='screen'
    )

    return LaunchDescription([
        namespace,
        use_sim_time,
        enable_logging,
        # Push namespace for grouped nodes
        GroupAction(
            condition=IfCondition(
                PythonExpression(["'", LaunchConfiguration('namespace'), "' != ''"])
            ),
            actions=[
                PushRosNamespace(LaunchConfiguration('namespace')),
                robot_controller,
                sensor_processor,
                logging_node
            ]
        ),
        # Log startup info
        LogInfo(
            msg=['Starting robot system with namespace: ', LaunchConfiguration('namespace')]
        )
    ])
```

## Running Launch Files

Run a launch file:

```bash
ros2 launch my_robot_package simple_launch.py
```

With arguments:

```bash
ros2 launch my_robot_package simple_launch.py use_sim_time:=true debug:=true
```

## Launch File Best Practices

1. **Use descriptive names**: Name launch files to clearly indicate their purpose
2. **Parameterize configurations**: Use launch arguments for flexibility
3. **Group related nodes**: Organize nodes logically in launch files
4. **Use YAML for parameters**: Keep parameter definitions in separate files
5. **Provide defaults**: Always provide sensible default values for arguments
6. **Document arguments**: Include descriptions for all launch arguments
7. **Use conditions**: Enable/disable nodes based on arguments when appropriate
8. **Test thoroughly**: Verify launch files work in different scenarios

## Common Launch File Patterns

### Robot Bringup
Launch all nodes needed to operate a robot:
```bash
ros2 launch my_robot_bringup robot.launch.py
```

### Simulation
Launch robot with simulation environment:
```bash
ros2 launch my_robot_gazebo simulation.launch.py
```

### Debugging
Launch with additional debug tools:
```bash
ros2 launch my_robot_bringup debug.launch.py
```

## Learning Objectives

After completing this tutorial, you should be able to:
- Create launch files to start multiple nodes
- Use launch arguments to parameterize launch files
- Configure parameters and remappings in launch files
- Use conditional logic in launch files
- Organize complex robot systems using launch files

## Next Steps

Continue to learn about [Python-ROS Integration with rclpy](./rclpy-integration) to understand how to bridge Python agents to ROS controllers.