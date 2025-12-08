---
title: "Gazebo Physics Fundamentals"
description: "Understanding physics simulation: gravity, collisions, rigid body dynamics"
---

# Gazebo Physics Fundamentals

## Overview

Gazebo is a powerful physics simulation engine that provides realistic simulation of robots in complex environments. Understanding Gazebo's physics capabilities is essential for creating accurate digital twins of physical robots. This section covers the fundamental physics concepts that govern robot simulation in Gazebo.

## Physics Engine Architecture

Gazebo uses one of several physics engines to simulate the physical world:
- **ODE (Open Dynamics Engine)**: Default engine, good for general-purpose simulation
- **Bullet**: Good for robust collision detection
- **DART**: Advanced dynamics and collisions
- **Simbody**: High-fidelity multibody dynamics

Each engine has its strengths and is chosen based on the specific requirements of the simulation.

## Core Physics Concepts

### Gravity
Gravity is a fundamental force in Gazebo simulations that affects all objects with mass. It's defined in the world file and typically set to Earth's gravity (9.81 m/s²) but can be adjusted for different environments:

```xml
<world name="default">
  <gravity>0 0 -9.8</gravity>
  <!-- Other world elements -->
</world>
```

### Rigid Body Dynamics
In Gazebo, objects are treated as rigid bodies, meaning their shape and size remain constant during simulation. Rigid body dynamics involve:
- **Position and orientation**: The pose of the object in 3D space
- **Velocity and angular velocity**: How the object moves through space
- **Mass and inertia**: Physical properties that affect how forces affect motion
- **Forces and torques**: External influences that change the object's motion

### Collision Detection
Gazebo uses collision detection to determine when objects make contact. There are two types of collision geometries:
- **Visual geometry**: How the object appears in the simulation
- **Collision geometry**: Simplified shapes used for collision detection (often less detailed than visual geometry for performance)

Common collision shapes include boxes, spheres, cylinders, and meshes.

## Forces and Constraints

### Contact Forces
When objects collide, Gazebo calculates contact forces based on:
- **Material properties**: Friction coefficients, restitution (bounciness)
- **Contact geometry**: Points and normals of contact
- **Relative velocities**: How fast objects are moving relative to each other

### Joint Constraints
Joints in Gazebo constrain the motion between two bodies:
- **Revolute joints**: Allow rotation around a single axis
- **Prismatic joints**: Allow linear motion along a single axis
- **Fixed joints**: Rigidly connect two bodies
- **Ball joints**: Allow rotation around multiple axes

## Physics Parameters

### Material Properties
Different materials have different physical properties that affect simulation:

- **Friction**: Determines how much resistance there is to sliding motion
  - Static friction: Resistance to initial motion
  - Dynamic friction: Resistance during motion
- **Restitution**: How "bouncy" collisions are (0 = no bounce, 1 = perfectly elastic)
- **Damping**: How quickly motion slows down over time

### Inertial Properties
For accurate simulation, each link needs proper inertial properties:
- **Mass**: How much matter the object contains
- **Center of mass**: The point where mass is balanced
- **Inertia tensor**: How mass is distributed relative to rotation axes

Example inertial specification:
```xml
<inertial>
  <mass>1.0</mass>
  <inertia>
    <ixx>0.01</ixx>
    <ixy>0.0</ixy>
    <ixz>0.0</ixz>
    <iyy>0.01</iyy>
    <iyz>0.0</iyz>
    <izz>0.01</izz>
  </inertia>
</inertial>
```

## Simulation Accuracy Considerations

### Time Step
The physics simulation advances in discrete time steps. Smaller time steps provide more accurate simulation but require more computational resources. The time step is typically set in the world file:

```xml
<physics type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
</physics>
```

### Numerical Stability
Large forces or small masses can cause numerical instability. Best practices include:
- Using appropriate mass values (not too small or too large)
- Limiting force magnitudes
- Using appropriate time steps for the system being simulated

## Gazebo-Specific Physics Features

### Buoyancy
Gazebo can simulate buoyancy effects for underwater robots by adding buoyancy plugins that calculate forces based on displaced fluid volume.

### Wind Simulation
Wind forces can be applied to simulate outdoor conditions, affecting lightweight objects and aerial vehicles.

### Contact Sensors
Gazebo provides contact sensors that can detect when objects touch, useful for grippers, feet, or other interaction points.

## Integration with ROS

Gazebo integrates seamlessly with ROS through:
- **Gazebo ROS packages**: Provide bridges between Gazebo and ROS topics/services
- **URDF integration**: Allows direct use of URDF robot descriptions
- **Sensor plugins**: Publish sensor data to ROS topics
- **Controller plugins**: Accept commands from ROS topics

## Performance Optimization

### Simplified Collision Models
Use simpler collision geometries than visual models to improve performance:
- Use primitive shapes (boxes, spheres, cylinders) where possible
- Reduce mesh complexity for collision detection
- Use bounding box approximations for complex shapes

### Level of Detail (LOD)
Implement different levels of detail for objects based on their importance and distance from the robot.

## Common Physics Issues and Solutions

### Robot Falling Through Ground
- Check that collision geometries are properly defined
- Verify that the ground plane has appropriate mass and static properties
- Ensure proper material properties are set

### Unstable Joint Motion
- Check that inertial properties are properly configured
- Verify that joint limits and friction parameters are reasonable
- Consider reducing the simulation time step

### Objects Passing Through Each Other
- Increase contact surface layers in physics parameters
- Verify collision geometries are properly defined
- Check that objects have appropriate mass values

## Best Practices

1. **Start Simple**: Begin with basic shapes and add complexity gradually
2. **Validate Physics**: Test individual components before complex systems
3. **Match Real-World Properties**: Use actual robot mass, inertia, and material properties
4. **Monitor Performance**: Balance accuracy with computational requirements
5. **Iterate and Test**: Continuously validate simulation behavior against expectations

## Learning Objectives

After completing this section, you should understand:
- The fundamental physics concepts that govern Gazebo simulation
- How to configure basic physics parameters for accurate simulation
- The relationship between visual and collision geometries
- Common physics issues and their solutions

## Next Steps

Continue to learn about [URDF/SDF Simulation](./urdf-sdf-simulation) to understand how robot models are integrated into Gazebo.