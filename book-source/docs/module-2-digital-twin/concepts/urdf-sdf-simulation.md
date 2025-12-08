---
title: "URDF/SDF Simulation Integration"
description: "Understanding the relationship between URDF and SDF for robot simulation"
---

# URDF/SDF Simulation Integration

## Overview

URDF (Unified Robot Description Format) and SDF (Simulation Description Format) are two XML-based formats used to describe robots and environments in simulation. While URDF is primarily used in ROS for robot description, SDF is the native format for Gazebo simulation. Understanding how these formats work together is crucial for creating accurate robot simulations.

## URDF vs SDF: Key Differences

### URDF (Unified Robot Description Format)
- **Primary Purpose**: Robot description in ROS
- **Scope**: Describes robot kinematics, dynamics, and basic visual/collision properties
- **Usage**: Mainly for ROS-based robot applications
- **Limitations**: Limited simulation-specific features

### SDF (Simulation Description Format)
- **Primary Purpose**: Complete simulation description
- **Scope**: Describes robots, environments, sensors, physics, plugins, and simulation parameters
- **Usage**: Native format for Gazebo simulation
- **Advantages**: Rich simulation features, plugins, and environment description

## How URDF Integrates with SDF

### Automatic Conversion
Gazebo can automatically convert URDF files to SDF using the `libgazebo_ros_factory.so` plugin. This allows you to use your existing URDF robot descriptions directly in Gazebo simulations.

### The Conversion Process
1. URDF is parsed and converted to an intermediate representation
2. Additional simulation-specific elements are added
3. The result is a complete SDF model ready for simulation

Example of a URDF that gets converted:
```xml
<!-- In URDF -->
<link name="link1">
  <visual>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <box size="1 1 1"/>
    </geometry>
  </collision>
  <inertial>
    <mass value="1.0"/>
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
</link>
```

Becomes in SDF:
```xml
<!-- Converted SDF -->
<link name="link1">
  <visual name="visual">
    <geometry>
      <box>
        <size>1 1 1</size>
      </box>
    </geometry>
  </visual>
  <collision name="collision">
    <geometry>
      <box>
        <size>1 1 1</size>
      </box>
    </geometry>
  </collision>
  <inertial>
    <mass>1.0</mass>
    <inertia>
      <ixx>1.0</ixx>
      <ixy>0.0</ixy>
      <ixz>0.0</ixz>
      <iyy>1.0</iyy>
      <iyz>0.0</iyz>
      <izz>1.0</izz>
    </inertia>
  </inertial>
</link>
```

## SDF Extensions to URDF

### Simulation-Specific Elements
SDF adds several elements that URDF doesn't support:

#### Dynamics Parameters
```xml
<link name="link1">
  <!-- Additional dynamics properties -->
  <dynamics damping="0.1" friction="0.01"/>
</link>
```

#### Sensor Definitions
```xml
<link name="sensor_link">
  <sensor name="camera" type="camera">
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
      <image>
        <width>640</width>
        <height>480</height>
      </image>
      <clip>
        <near>0.1</near>
        <far>100</far>
      </clip>
    </camera>
  </sensor>
</link>
```

#### Joint Transmissions
```xml
<transmission name="joint1_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="joint1">
    <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
  </joint>
  <actuator name="joint1_motor">
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

## Gazebo-Specific Tags in URDF

When using URDF with Gazebo, you can add Gazebo-specific tags to enhance the simulation:

### Material Definitions
```xml
<gazebo reference="link1">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>  <!-- Friction coefficient -->
  <mu2>0.2</mu2>  <!-- Second friction coefficient -->
  <kp>1000000.0</kp>  <!-- Contact stiffness -->
  <kd>1.0</kd>    <!-- Contact damping -->
</gazebo>
```

### Plugin Integration
```xml
<gazebo>
  <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
    <left_joint>left_wheel_joint</left_joint>
    <right_joint>right_wheel_joint</right_joint>
    <wheel_separation>0.3</wheel_separation>
    <wheel_diameter>0.15</wheel_diameter>
  </plugin>
</gazebo>
```

### Sensor Integration
```xml
<gazebo reference="camera_link">
  <sensor type="camera" name="camera1">
    <update_rate>30.0</update_rate>
    <camera name="head">
      <horizontal_fov>1.3962634</horizontal_fov>
      <image>
        <width>800</width>
        <height>600</height>
        <format>R8G8B8</format>
      </image>
      <clip>
        <near>0.01</near>
        <far>100</far>
      </clip>
    </camera>
    <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
      <frame_name>camera_link</frame_name>
    </plugin>
  </sensor>
</gazebo>
```

## Creating SDF Models from URDF

### Using xacro for Complex Models
Xacro (XML Macros) can be used to create parameterized URDF files that are easier to manage:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="my_robot">

  <xacro:property name="wheel_radius" value="0.1" />
  <xacro:property name="wheel_width" value="0.05" />

  <xacro:macro name="wheel" params="prefix *origin">
    <link name="${prefix}_wheel">
      <visual>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="1.0"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel prefix="front_left">
    <origin xyz="0.2 0.1 0"/>
  </xacro:wheel>

</robot>
```

### Converting URDF to SDF
You can convert URDF to SDF using the `gz sdf` command:

```bash
# Convert URDF to SDF
gz sdf -p robot.urdf > robot.sdf

# Or with xacro preprocessing
xacro robot.xacro | gz sdf -p /dev/stdin > robot.sdf
```

## Model Structure in Gazebo

### Model Directory Structure
Gazebo expects models to be organized in a specific directory structure:

```
~/.gazebo/models/my_robot/
├── model.config      # Model metadata
├── model.sdf         # Model description
└── meshes/           # 3D model files
    └── link1.dae
└── materials/
    └── textures/
```

### Model Configuration File
The `model.config` file contains metadata about the model:

```xml
<?xml version="1.0"?>
<model>
  <name>My Robot</name>
  <version>1.0</version>
  <sdf version='1.6'>model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>
    A description of your robot model.
  </description>
</model>
```

## Simulation Considerations

### Performance Optimization
- **Simplified collision geometry**: Use simpler shapes for collision detection than for visualization
- **Appropriate mass values**: Ensure realistic mass properties for stable simulation
- **Joint limits**: Define realistic joint limits and friction parameters

### Accuracy vs Performance Trade-offs
- **Time step**: Smaller time steps improve accuracy but reduce performance
- **Solver iterations**: More iterations improve accuracy but increase computation time
- **Contact parameters**: Properly tuned contact parameters improve stability

## Best Practices for URDF/SDF Integration

### 1. Maintain Separate Files
Keep your base URDF for ROS operations and extend it with Gazebo-specific tags when needed:

```xml
<!-- robot.urdf (base) -->
<robot name="my_robot">
  <!-- Basic URDF elements -->
  <link name="base_link">
    <!-- ... -->
  </link>

  <!-- Gazebo-specific extensions -->
  <gazebo reference="base_link">
    <material>Gazebo/Orange</material>
  </gazebo>
</robot>
```

### 2. Use xacro for Complex Models
Xacro makes it easier to maintain parameterized models:

```xml
<xacro:property name="robot_name" value="my_robot" />
<xacro:if value="$(arg use_gazebo)">
  <gazebo>
    <!-- Gazebo-specific content -->
  </gazebo>
</xacro:if>
```

### 3. Validate Your Models
Use tools to check your models before simulation:
```bash
# Check URDF validity
check_urdf my_robot.urdf

# Check SDF validity
gz sdf -k model.sdf
```

## Common Integration Issues and Solutions

### 1. Model Not Loading in Gazebo
- Check that all mesh files are in the correct location
- Verify that the model follows Gazebo's directory structure
- Ensure all referenced files exist and are accessible

### 2. Physics Issues
- Check that inertial properties are properly defined
- Verify that mass values are reasonable
- Ensure collision geometries are properly specified

### 3. Plugin Issues
- Verify that required Gazebo plugins are installed
- Check that plugin parameters are correctly specified
- Ensure proper joint and link names match between URDF and plugins

## Advanced SDF Features

### World Files
SDF can also describe complete simulation worlds:

```xml
<sdf version="1.6">
  <world name="my_world">
    <include>
      <uri>model://ground_plane</uri>
    </include>
    <include>
      <uri>model://sun</uri>
    </include>
    <model name="my_robot">
      <!-- Robot definition -->
    </model>
  </world>
</sdf>
```

### Physics Configuration
Fine-tune physics parameters for your specific simulation needs:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>10</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.0</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Learning Objectives

After completing this section, you should understand:
- The differences between URDF and SDF formats
- How URDF models are converted and extended for simulation
- How to add Gazebo-specific elements to URDF files
- Best practices for maintaining simulation-ready robot models
- How to structure models for use in Gazebo

## Next Steps

Continue to learn about [Sensor Simulation](./sensor-simulation) to understand how to simulate various robot sensors in Gazebo.