---
title: "URDF: Unified Robot Description Format"
description: "Defining humanoid structure and robot models using URDF"
---

# URDF: Unified Robot Description Format

## Overview

URDF (Unified Robot Description Format) is an XML-based format used in ROS to describe robot models. It defines the physical and visual properties of a robot, including its links (rigid parts), joints (connections between links), and other properties like inertia, visual representation, and collision properties.

## Purpose of URDF

URDF serves several critical functions in robotics:
- **Kinematic description**: Defines the physical structure and joint relationships
- **Visual representation**: Specifies how the robot appears in simulation and visualization
- **Collision properties**: Defines shapes for collision detection
- **Inertial properties**: Provides mass, center of mass, and inertia tensor information
- **Transmission interfaces**: Defines how joints connect to actuators

## URDF Structure

A URDF file contains several key elements:

### Robot Element
The root element that contains the entire robot description:
```xml
<robot name="my_robot">
  <!-- Robot components go here -->
</robot>
```

### Links
Links represent rigid bodies in the robot:
```xml
<link name="link_name">
  <visual>
    <!-- Visual properties -->
  </visual>
  <collision>
    <!-- Collision properties -->
  </collision>
  <inertial>
    <!-- Mass and inertia properties -->
  </inertial>
</link>
```

### Joints
Joints connect links and define their relative motion:
```xml
<joint name="joint_name" type="joint_type">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="x y z" rpy="roll pitch yaw"/>
  <axis xyz="x y z"/>
  <limit lower="min" upper="max" effort="max_effort" velocity="max_velocity"/>
</joint>
```

## Joint Types

URDF supports several joint types:
- **revolute**: Rotational joint with limited range
- **continuous**: Rotational joint with unlimited range
- **prismatic**: Linear sliding joint with limited range
- **fixed**: No movement (welded connection)
- **floating**: 6-DOF movement
- **planar**: Motion on a plane

## Visual and Collision Properties

### Visual Properties
Define how the link appears in visualization:
- **Geometry**: Shape (box, cylinder, sphere, mesh)
- **Material**: Color and texture properties
- **Origin**: Position and orientation relative to the link frame

### Collision Properties
Define shapes used for collision detection:
- **Geometry**: Similar to visual but often simplified
- **Origin**: Position and orientation relative to the link frame

## Inertial Properties

Specify the physical properties needed for dynamics simulation:
- **Mass**: Mass of the link
- **Inertia matrix**: 3x3 inertia tensor
- **Origin**: Center of mass location

## Materials and Colors

Materials can be defined once and reused:
```xml
<material name="red">
  <color rgba="1 0 0 1"/>
</material>
```

## Complete Example

Here's a simple 2-link robot example:
```xml
<robot name="simple_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Rotating arm -->
  <link name="arm_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.5"/>
      </geometry>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.05 0.05 0.5"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint connecting base and arm -->
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.15" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
  </joint>
</robot>
```

## URDF Tools and Visualization

ROS provides several tools for working with URDF:
- **rviz**: Visualize robot models
- **robot_state_publisher**: Publish joint states and transforms
- **joint_state_publisher**: Publish joint state messages
- **check_urdf**: Validate URDF files
- **urdf_to_graphiz**: Generate visual representation of kinematic tree

## Best Practices

1. **Use consistent naming**: Follow naming conventions for links and joints
2. **Separate visual and collision**: Use different geometries for visualization and collision
3. **Define proper inertial properties**: Essential for accurate simulation
4. **Use xacro for complex models**: Xacro extends URDF with macros and variables
5. **Validate URDF**: Always check your URDF files before use
6. **Consider the base frame**: Define a clear base coordinate frame

## Learning Objectives

After completing this section, you should understand:
- The structure and components of URDF files
- How to define links, joints, and their properties
- The importance of visual, collision, and inertial properties
- How URDF integrates with ROS and simulation environments

## Next Steps

Continue to learn about [Creating Your First ROS 2 Package](../workflows/creating-first-package) to start building ROS 2 applications.