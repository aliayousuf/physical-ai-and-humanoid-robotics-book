---
title: "Gazebo Environment Setup"
description: "Creating and configuring simulation environments in Gazebo"
---

# Gazebo Environment Setup

## Overview

Creating realistic and functional simulation environments is crucial for effective robot testing and development. This workflow guide covers the process of setting up Gazebo environments, from basic world creation to complex scene configuration with models, lighting, and physics properties.

## Prerequisites

Before setting up Gazebo environments, ensure you have:
- Gazebo installed (Gazebo Garden or Fortress recommended)
- Basic understanding of SDF (Simulation Description Format)
- Knowledge of robot models and URDF/SDF integration
- Understanding of coordinate systems and transformations

## Basic Environment Creation

### 1. World File Structure

A basic Gazebo world file follows this structure:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="my_world">
    <!-- World properties -->
    <light name="sun" type="directional">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>1000</range>
        <constant>0.9</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <direction>-0.6 0.4 -0.8</direction>
    </light>

    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Your models and objects will go here -->

    <!-- Physics engine configuration -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

### 2. Creating a Simple Room Environment

Let's create a simple room environment with walls, floor, and ceiling:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="simple_room">
    <!-- Lighting -->
    <light name="room_light" type="point">
      <pose>0 0 2 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <attenuation>
        <range>10</range>
        <constant>0.9</constant>
        <linear>0.045</linear>
        <quadratic>0.0075</quadratic>
      </attenuation>
    </light>

    <!-- Floor -->
    <model name="floor">
      <pose>0 0 0 0 0 0</pose>
      <static>true</static>
      <link name="floor_link">
        <collision name="floor_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="floor_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Walls -->
    <!-- Left wall -->
    <model name="left_wall">
      <pose>-5 0 2.5 0 0 0</pose>
      <static>true</static>
      <link name="left_wall_link">
        <collision name="left_wall_collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="left_wall_visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Right wall -->
    <model name="right_wall">
      <pose>5 0 2.5 0 0 0</pose>
      <static>true</static>
      <link name="right_wall_link">
        <collision name="right_wall_collision">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="right_wall_visual">
          <geometry>
            <box>
              <size>0.1 10 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Front wall -->
    <model name="front_wall">
      <pose>0 -5 2.5 0 0 0</pose>
      <static>true</static>
      <link name="front_wall_link">
        <collision name="front_wall_collision">
          <geometry>
            <box>
              <size>10 0.1 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="front_wall_visual">
          <geometry>
            <box>
              <size>10 0.1 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Back wall -->
    <model name="back_wall">
      <pose>0 5 2.5 0 0 0</pose>
      <static>true</static>
      <link name="back_wall_link">
        <collision name="back_wall_collision">
          <geometry>
            <box>
              <size>10 0.1 5</size>
            </box>
          </geometry>
        </collision>
        <visual name="back_wall_visual">
          <geometry>
            <box>
              <size>10 0.1 5</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.5 0.5 1</ambient>
            <diffuse>0.5 0.5 0.5 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Ceiling -->
    <model name="ceiling">
      <pose>0 0 5 0 0 0</pose>
      <static>true</static>
      <link name="ceiling_link">
        <collision name="ceiling_collision">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
        </collision>
        <visual name="ceiling_visual">
          <geometry>
            <box>
              <size>10 10 0.1</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.7 0.7 1</ambient>
            <diffuse>0.7 0.7 0.7 1</diffuse>
            <specular>0.1 0.1 0.1 1</specular>
          </material>
        </visual>
      </link>
    </model>

    <!-- Physics configuration -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
  </world>
</sdf>
```

## Advanced Environment Setup

### 1. Using Built-in Models

Gazebo provides many built-in models that can be included in your world:

```xml
<!-- Include built-in models -->
<include>
  <uri>model://ground_plane</uri>
</include>

<include>
  <uri>model://sun</uri>
</include>

<!-- Include a table -->
<include>
  <uri>model://table</uri>
  <pose>2 0 0 0 0 0</pose>
</include>

<!-- Include a box -->
<include>
  <uri>model://box</uri>
  <pose>-2 1 0.5 0 0 0</pose>
  <box>
    <size>1 1 1</size>
  </box>
</include>
```

### 2. Adding Custom Models

To add your own custom models:

1. Create a model directory structure:
```
~/.gazebo/models/my_custom_model/
├── model.config
└── model.sdf
```

2. Create the `model.config` file:
```xml
<?xml version="1.0"?>
<model>
  <name>My Custom Model</name>
  <version>1.0</version>
  <sdf version="1.7">model.sdf</sdf>
  <author>
    <name>Your Name</name>
    <email>your.email@example.com</email>
  </author>
  <description>A custom model for my simulation.</description>
</model>
```

3. Include your custom model in the world file:
```xml
<include>
  <uri>model://my_custom_model</uri>
  <pose>0 0 0 0 0 0</pose>
</include>
```

### 3. Physics Configuration

Configure physics properties for your environment:

```xml
<physics name="my_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1</real_time_factor>
  <real_time_update_rate>1000</real_time_update_rate>
  <gravity>0 0 -9.8</gravity>
  <ode>
    <solver>
      <type>quick</type>
      <iters>1000</iters>
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

## Environment Customization

### 1. Terrain and Outdoor Environments

For outdoor environments, you can create terrains:

```xml
<model name="terrain">
  <static>true</static>
  <link name="terrain_link">
    <collision name="terrain_collision">
      <geometry>
        <heightmap>
          <uri>file://path/to/heightmap.png</uri>
          <size>100 100 20</size>
          <pos>0 0 0</pos>
        </heightmap>
      </geometry>
    </collision>
    <visual name="terrain_visual">
      <geometry>
        <heightmap>
          <uri>file://path/to/heightmap.png</uri>
          <size>100 100 20</size>
        </heightmap>
      </geometry>
    </visual>
  </link>
</model>
```

### 2. Weather and Environmental Effects

While Gazebo doesn't have built-in weather effects, you can simulate some conditions:

```xml
<!-- Fog effect -->
<scene>
  <fog type="linear">
    <color>0.8 0.8 0.8</color>
    <density>0.5</density>
    <range>1 10</range>
  </fog>
</scene>
```

### 3. Plugins for Advanced Features

Add plugins to enhance your environment:

```xml
<!-- Add a plugin for custom behavior -->
<plugin name="my_world_plugin" filename="libMyWorldPlugin.so">
  <update_rate>1.0</update_rate>
  <!-- Plugin-specific parameters -->
</plugin>
```

## Workflow: Setting Up a Complete Environment

### Step 1: Plan Your Environment
- Determine the purpose of your simulation
- Identify required objects and obstacles
- Consider lighting and environmental conditions
- Plan for robot starting positions and goals

### Step 2: Create the Base World
- Start with a basic world template
- Add essential elements (ground plane, lighting)
- Configure basic physics properties

### Step 3: Add Static Objects
- Include walls, furniture, or terrain as needed
- Position objects appropriately
- Ensure collision properties are set correctly

### Step 4: Add Dynamic Elements
- Include movable objects if needed
- Set up any animated or interactive elements
- Configure joint properties for moving parts

### Step 5: Configure Sensors and Cameras
- Add sensor models if needed for environment monitoring
- Set up cameras for visualization
- Ensure proper positioning and parameters

### Step 6: Test and Refine
- Load the world in Gazebo
- Test robot navigation and interaction
- Adjust parameters as needed for performance and realism

## Best Practices

### 1. Performance Optimization
- Use simple collision geometries when possible
- Limit the number of complex models in a scene
- Use appropriate physics parameters for your use case
- Consider level-of-detail for distant objects

### 2. Reusability
- Create modular environment components
- Use parameters to make environments configurable
- Document your environment setup for others
- Follow consistent naming conventions

### 3. Safety and Validation
- Ensure all models have proper collision properties
- Verify that the environment is physically stable
- Test with your robot model to ensure compatibility
- Validate that sensors work correctly in the environment

## Common Issues and Solutions

### 1. Model Not Loading
- Check that model files exist in the correct location
- Verify that URIs in include statements are correct
- Ensure model.config file is properly formatted
- Check Gazebo model path configuration

### 2. Physics Instability
- Increase solver iterations
- Reduce time step size
- Check mass and inertial properties of objects
- Ensure proper collision geometry definition

### 3. Performance Issues
- Reduce model complexity
- Lower physics update rates if acceptable
- Simplify collision geometries
- Limit the number of active sensors

## Advanced Configuration

### 1. Multi-Robot Environments
For simulations with multiple robots, consider:
- Separate spawn areas for each robot
- Appropriate spacing to avoid interference
- Individual lighting and sensor configurations
- Coordinated physics parameters

### 2. Scenario-Based Environments
Create environments for specific scenarios:
- Navigation testing: Various obstacle layouts
- Manipulation: Tables, objects, and tools
- Search and rescue: Complex, cluttered environments
- Human-robot interaction: Socially relevant spaces

## Running Your Environment

### Loading the World in Gazebo
```bash
# Load your custom world
gz sim -r my_world.sdf

# Or if using the older gazebo command
gazebo my_world.sdf
```

### Launching with ROS
Create a launch file to start your environment with ROS integration:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    world_path = PathJoinSubstitution([
        FindPackageShare('my_robot_package'),
        'worlds',
        'my_world.sdf'
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_path,
            'verbose': 'true'
        }.items()
    )

    return LaunchDescription([
        gazebo
    ])
```

## Learning Objectives

After completing this workflow, you should be able to:
- Create basic and complex Gazebo world files
- Add and configure various models in your environment
- Set up appropriate physics properties for your simulation
- Optimize environments for performance and realism
- Troubleshoot common environment setup issues

## Next Steps

Continue to learn about [Unity Scene Creation](../workflows/unity-scenes) to understand how to create Unity environments for robot visualization and interaction.