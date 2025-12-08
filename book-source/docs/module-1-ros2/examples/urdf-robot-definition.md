---
title: "URDF Robot Definition Example"
description: "Creating a simple robot model using URDF with visual and collision properties"
---

# URDF Robot Definition Example

## Overview

This example demonstrates how to create a simple robot model using URDF (Unified Robot Description Format). We'll build a basic wheeled robot with a chassis, wheels, and a sensor mast to illustrate the key concepts of URDF modeling.

## Complete URDF Example

Create `my_robot_package/urdf/my_robot.urdf`:

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- MATERIALS -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>

  <material name="green">
    <color rgba="0.0 1.0 0.0 1.0"/>
  </material>

  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>

  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- BASE LINK -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 0.075" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.15"/>
      </geometry>
      <origin xyz="0 0 0.075" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- LEFT WHEEL -->
  <link name="left_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="white"/>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- RIGHT WHEEL -->
  <link name="right_wheel">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <material name="white"/>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.04"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <!-- FRONT WHEEL (CASTER) -->
  <link name="front_caster">
    <visual>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- SENSOR MAST -->
  <link name="sensor_mast">
    <visual>
      <geometry>
        <cylinder radius="0.01" length="0.1"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.01" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.02"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0" iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <!-- JOINTS -->
  <!-- Left wheel joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel"/>
    <origin xyz="0.0 0.175 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right wheel joint -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel"/>
    <origin xyz="0.0 -0.175 0.0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Front caster joint (fixed) -->
  <joint name="front_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="front_caster"/>
    <origin xyz="0.2 0.0 0.0" rpy="0 0 0"/>
  </joint>

  <!-- Sensor mast joint -->
  <joint name="sensor_mast_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_mast"/>
    <origin xyz="0.0 0.0 0.15" rpy="0 0 0"/>
  </joint>

</robot>
```

## Xacro Version (Recommended)

For more complex robots, use Xacro to make URDF more maintainable. Create `my_robot_package/urdf/my_robot.xacro`:

```xml
<?xml version="1.0"?>
<robot name="my_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- PROPERTY DEFINITIONS -->
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Robot dimensions -->
  <xacro:property name="chassis_length" value="0.5" />
  <xacro:property name="chassis_width" value="0.3" />
  <xacro:property name="chassis_height" value="0.15" />
  <xacro:property name="wheel_radius" value="0.05" />
  <xacro:property name="wheel_width" value="0.04" />
  <xacro:property name="base_mass" value="1.0" />
  <xacro:property name="wheel_mass" value="0.1" />

  <!-- MATERIALS -->
  <material name="blue">
    <color rgba="0.0 0.0 1.0 1.0"/>
  </material>

  <material name="green">
    <color rgba="0.0 1.0 0.0 1.0"/>
  </material>

  <material name="red">
    <color rgba="1.0 0.0 0.0 1.0"/>
  </material>

  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>

  <!-- MACRO FOR WHEEL -->
  <xacro:macro name="wheel" params="prefix parent x_reflect y_reflect">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="white"/>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
      </collision>
      <inertial>
        <mass value="${wheel_mass}"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0"
                 iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="0 ${y_reflect*(chassis_width/2 + wheel_width/2)} 0" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- BASE LINK -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 ${chassis_height/2}" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="${chassis_length} ${chassis_width} ${chassis_height}"/>
      </geometry>
      <origin xyz="0 0 ${chassis_height/2}" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="${base_mass}"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0"
               iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- WHEELS -->
  <xacro:wheel prefix="left" parent="base_link" x_reflect="1" y_reflect="1"/>
  <xacro:wheel prefix="right" parent="base_link" x_reflect="1" y_reflect="-1"/>

  <!-- FRONT CASTER -->
  <link name="front_caster">
    <visual>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.03"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0"
               iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="front_caster_joint" type="fixed">
    <parent link="base_link"/>
    <child link="front_caster"/>
    <origin xyz="${chassis_length/2} 0.0 0.0" rpy="0 0 0"/>
  </joint>

  <!-- SENSOR MAST -->
  <link name="sensor_mast">
    <visual>
      <geometry>
        <cylinder radius="0.01" length="0.1"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.01" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.02"/>
      <inertia ixx="0.00001" ixy="0.0" ixz="0.0"
               iyy="0.00001" iyz="0.0" izz="0.00001"/>
    </inertial>
  </link>

  <joint name="sensor_mast_joint" type="fixed">
    <parent link="base_link"/>
    <child link="sensor_mast"/>
    <origin xyz="0.0 0.0 ${chassis_height}" rpy="0 0 0"/>
  </joint>

</robot>
```

## Launch File for Visualization

Create `my_robot_package/launch/display_robot.launch.py` to visualize the robot:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    urdf_model = DeclareLaunchArgument(
        'urdf_model',
        default_value=PathJoinSubstitution([
            FindPackageShare('my_robot_package'),
            'urdf',
            'my_robot.xacro'
        ]),
        description='Path to URDF/XACRO file'
    )

    # Launch configuration
    urdf_model_path = LaunchConfiguration('urdf_model')

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[{
            'robot_description': open(urdf_model_path.perform({})).read()
        }],
        output='screen'
    )

    # Joint state publisher (GUI for joint visualization)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_gui': True
        }],
        output='screen'
    )

    # RViz2 for visualization
    rviz2 = Node(
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
        urdf_model,
        robot_state_publisher,
        joint_state_publisher,
        rviz2
    ])
```

## RViz Configuration

Create `my_robot_package/rviz/robot_display.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
      Splitter Ratio: 0.5
    Tree Height: 617
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
        base_link:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        front_caster:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        left_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        right_wheel:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
        sensor_mast:
          Alpha: 1
          Show Axes: false
          Show Trail: false
          Value: true
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
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
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 1.5
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
  Height: 846
  Width: 1200
```

## Package Configuration

Update the `package.xml` to include visualization dependencies:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package with URDF robot model</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>tf2_ros</depend>
  <depend>robot_state_publisher</depend>
  <depend>joint_state_publisher</depend>
  <depend>rviz2</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Running the Visualization

First, create the necessary directories:

```bash
mkdir -p ~/ros2_ws/src/my_robot_package/urdf
mkdir -p ~/ros2_ws/src/my_robot_package/rviz
mkdir -p ~/ros2_ws/src/my_robot_package/launch
```

Copy the URDF file to the urdf directory, then build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

Run the visualization:

```bash
ros2 launch my_robot_package display_robot.launch.py
```

## Validating URDF

You can validate your URDF file using the check_urdf command:

```bash
# Install urdfdom package if not already installed
sudo apt install ros-humble-urdfdom

# Check your URDF file
check_urdf ~/ros2_ws/src/my_robot_package/urdf/my_robot.urdf
```

## Common URDF Issues and Solutions

### 1. Missing Inertial Properties
```xml
<!-- Always include inertial properties for physics simulation -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
</inertial>
```

### 2. Incorrect Joint Origins
```xml
<!-- Make sure joint origins are correctly positioned -->
<joint name="left_wheel_joint" type="continuous">
  <parent link="base_link"/>
  <child link="left_wheel"/>
  <origin xyz="0.0 0.175 0.0" rpy="0 0 0"/>  <!-- Position relative to parent -->
  <axis xyz="0 0 1"/>  <!-- Rotation axis -->
</joint>
```

### 3. Visual vs Collision Mismatch
```xml
<!-- Use simpler collision geometry for better performance -->
<collision>
  <geometry>
    <box size="0.5 0.3 0.15"/>  <!-- Simplified box -->
  </geometry>
</collision>
<visual>
  <geometry>
    <mesh filename="package://my_robot_package/meshes/detailed_model.dae"/>  <!-- Detailed visual -->
  </geometry>
</visual>
```

## Advanced URDF Features

### 1. Transmission Elements
For actual actuation, add transmissions:

```xml
<transmission name="left_wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_wheel_joint">
    <hardwareInterface>hardware_interface/VelocityJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_wheel_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### 2. Gazebo Integration
Add Gazebo-specific elements:

```xml
<gazebo reference="base_link">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>
```

## Learning Objectives

After completing this example, you should be able to:
- Create complete robot models using URDF
- Use Xacro to make URDF more maintainable
- Visualize robot models in RViz
- Understand the relationship between visual, collision, and inertial properties
- Set up launch files for robot visualization

## Next Steps

Continue to learn about [Python Control Loop Example](../examples/python-control-loop) to see how to implement control systems in Python using rclpy.