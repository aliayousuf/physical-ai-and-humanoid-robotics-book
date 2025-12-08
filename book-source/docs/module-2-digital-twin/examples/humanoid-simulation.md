---
title: "Humanoid Simulation Example"
description: "Complete simulation example with humanoid robot in Gazebo and Unity"
---

# Humanoid Simulation Example

## Overview

This example demonstrates a complete humanoid robot simulation integrating Gazebo physics with Unity visualization. We'll create a basic humanoid model with sensors, implement a complete simulation environment, and show how to visualize the robot in both simulation and Unity for human-robot interaction.

## Complete Humanoid Robot Model (URDF)

First, let's create a complete humanoid robot model in URDF format:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

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

  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>

  <!-- BASE LINK (Pelvis) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <material name="blue"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.2 0.4"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0.0" ixz="0.0" iyy="0.3" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <!-- TORSO -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.25 0.15 0.5"/>
      </geometry>
      <material name="white"/>
      <origin xyz="0 0 0.35" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.25 0.15 0.5"/>
      </geometry>
      <origin xyz="0 0 0.35" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.3" ixy="0.0" ixz="0.0" iyy="0.4" iyz="0.0" izz="0.2"/>
    </inertial>
  </link>

  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
  </joint>

  <!-- HEAD -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="white"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
  </link>

  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.5" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- LEFT ARM -->
  <link name="left_shoulder">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.15"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.15"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.15 0.1 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_elbow_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.5" effort="50" velocity="2.0"/>
  </joint>

  <link name="left_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.035" length="0.25"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.035" length="0.25"/>
      </geometry>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="left_wrist_joint" type="revolute">
    <parent link="left_upper_arm"/>
    <child link="left_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="2.0"/>
  </joint>

  <!-- RIGHT ARM (symmetric to left) -->
  <link name="right_shoulder">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.15"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="0 1.57 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.15"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 1.57 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="right_shoulder"/>
    <origin xyz="0.15 -0.1 0.2" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_elbow_joint" type="revolute">
    <parent link="right_shoulder"/>
    <child link="right_upper_arm"/>
    <origin xyz="0 0 -0.15" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.5" effort="50" velocity="2.0"/>
  </joint>

  <link name="right_lower_arm">
    <visual>
      <geometry>
        <cylinder radius="0.035" length="0.25"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.035" length="0.25"/>
      </geometry>
      <origin xyz="0 0 -0.125" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.003"/>
    </inertial>
  </link>

  <joint name="right_wrist_joint" type="revolute">
    <parent link="right_upper_arm"/>
    <child link="right_lower_arm"/>
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.57" upper="1.57" effort="30" velocity="2.0"/>
  </joint>

  <!-- LEFT LEG -->
  <link name="left_hip">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="1.57 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.57 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="left_hip"/>
    <origin xyz="-0.1 0.1 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="left_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_knee_joint" type="revolute">
    <parent link="left_hip"/>
    <child link="left_upper_leg"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.2" effort="150" velocity="1.0"/>
  </joint>

  <link name="left_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.045" length="0.4"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.045" length="0.4"/>
      </geometry>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="left_ankle_joint" type="revolute">
    <parent link="left_upper_leg"/>
    <child link="left_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="80" velocity="1.0"/>
  </joint>

  <link name="left_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="left_foot_joint" type="fixed">
    <parent link="left_lower_leg"/>
    <child link="left_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  </joint>

  <!-- RIGHT LEG (symmetric to left) -->
  <link name="right_hip">
    <visual>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <material name="red"/>
      <origin xyz="0 0 0" rpy="1.57 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.06" length="0.1"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.57 0 0"/>
    </collision>
    <inertial>
      <mass value="1.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_hip_joint" type="revolute">
    <parent link="base_link"/>
    <child link="right_hip"/>
    <origin xyz="-0.1 -0.1 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <link name="right_upper_leg">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.4"/>
      </geometry>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="3.0"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_knee_joint" type="revolute">
    <parent link="right_hip"/>
    <child link="right_upper_leg"/>
    <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="0.2" effort="150" velocity="1.0"/>
  </joint>

  <link name="right_lower_leg">
    <visual>
      <geometry>
        <cylinder radius="0.045" length="0.4"/>
      </geometry>
      <material name="green"/>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.045" length="0.4"/>
      </geometry>
      <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="2.5"/>
      <inertia ixx="0.04" ixy="0.0" ixz="0.0" iyy="0.04" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="right_ankle_joint" type="revolute">
    <parent link="right_upper_leg"/>
    <child link="right_lower_leg"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.5" upper="0.5" effort="80" velocity="1.0"/>
  </joint>

  <link name="right_foot">
    <visual>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <material name="black"/>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.1 0.05"/>
      </geometry>
      <origin xyz="0 0 -0.025" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.005"/>
    </inertial>
  </link>

  <joint name="right_foot_joint" type="fixed">
    <parent link="right_lower_leg"/>
    <child link="right_foot"/>
    <origin xyz="0 0 -0.4" rpy="0 0 0"/>
  </joint>

  <!-- SENSORS -->
  <!-- IMU in torso -->
  <gazebo reference="torso">
    <sensor type="imu" name="torso_imu">
      <always_on>true</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2e-4</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>1.7e-2</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
      <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
        <topicName>imu/data</topicName>
        <bodyName>torso</bodyName>
        <frameName>torso_imu_frame</frameName>
        <serviceName>imu/service</serviceName>
      </plugin>
    </sensor>
  </gazebo>

  <!-- Camera in head -->
  <gazebo reference="head">
    <sensor type="camera" name="head_camera">
      <update_rate>30.0</update_rate>
      <camera name="head_camera">
        <horizontal_fov>1.047</horizontal_fov>
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>300</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
        <frame_name>head_camera_frame</frame_name>
        <min_depth>0.1</min_depth>
        <max_depth>300.0</max_depth>
      </plugin>
    </sensor>
  </gazebo>

  <!-- LiDAR on torso -->
  <gazebo reference="torso">
    <sensor type="ray" name="torso_lidar">
      <update_rate>10</update_rate>
      <ray>
        <scan>
          <horizontal>
            <samples>360</samples>
            <resolution>1</resolution>
            <min_angle>-3.14159</min_angle>
            <max_angle>3.14159</max_angle>
          </horizontal>
        </scan>
        <range>
          <min>0.10</min>
          <max>10.0</max>
          <resolution>0.01</resolution>
        </range>
      </ray>
      <plugin name="lidar_controller" filename="libgazebo_ros_laser.so">
        <topic_name>scan</topic_name>
        <frame_name>torso_lidar_frame</frame_name>
      </plugin>
    </sensor>
  </gazebo>

  <!-- TRANSMISSIONS for actuators -->
  <transmission name="left_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="left_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="left_knee_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="left_knee_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_hip_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_hip_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_hip_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="right_knee_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_knee_joint">
      <hardwareInterface>hardware_interface/PositionJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_knee_motor">
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

</robot>
```

## Gazebo World for Humanoid Simulation

Create a world file for the humanoid simulation:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Lighting -->
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

    <!-- Simple environment with obstacles -->
    <!-- Table -->
    <model name="table">
      <pose>-2 0 0.5 0 0 0</pose>
      <link name="table_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>1 0.8 0.8</size>
            </box>
          </geometry>
          <material>
            <ambient>0.5 0.3 0.1 1</ambient>
            <diffuse>0.5 0.3 0.1 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>10.0</mass>
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
    </model>

    <!-- Chair -->
    <model name="chair">
      <pose>-2 1.2 0.3 0 0 0</pose>
      <link name="chair_base">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.5 0.5 0.6</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.5 0.5 0.6</size>
            </box>
          </geometry>
          <material>
            <ambient>0.4 0.2 0.0 1</ambient>
            <diffuse>0.4 0.2 0.0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>5.0</mass>
          <inertia>
            <ixx>0.5</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.5</iyy>
            <iyz>0.0</iyz>
            <izz>0.5</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Box obstacle -->
    <model name="box_obstacle">
      <pose>1 0 0.2 0 0 0</pose>
      <link name="box_link">
        <collision name="collision">
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box>
              <size>0.4 0.4 0.4</size>
            </box>
          </geometry>
          <material>
            <ambient>0.7 0.0 0.0 1</ambient>
            <diffuse>0.7 0.0 0.0 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>2.0</mass>
          <inertia>
            <ixx>0.04</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>0.04</iyy>
            <iyz>0.0</iyz>
            <izz>0.04</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Include the humanoid robot -->
    <!-- This would be loaded separately in a launch file -->

    <!-- Physics configuration -->
    <physics name="default_physics" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>100</iters>
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
  </world>
</sdf>
```

## ROS Control Configuration

Create a control configuration file for the humanoid:

`humanoid_control.yaml`:
```yaml
# Joint state controller
joint_state_controller:
  type: joint_state_controller/JointStateController
  publish_rate: 50

# Controllers for different joints
left_leg_controller:
  type: position_controllers/JointTrajectoryController
  joints:
    - left_hip_joint
    - left_knee_joint
    - left_ankle_joint
  gains:
    left_hip_joint: {p: 100.0, i: 0.01, d: 10.0}
    left_knee_joint: {p: 100.0, i: 0.01, d: 10.0}
    left_ankle_joint: {p: 50.0, i: 0.01, d: 5.0}

right_leg_controller:
  type: position_controllers/JointTrajectoryController
  joints:
    - right_hip_joint
    - right_knee_joint
    - right_ankle_joint
  gains:
    right_hip_joint: {p: 100.0, i: 0.01, d: 10.0}
    right_knee_joint: {p: 100.0, i: 0.01, d: 10.0}
    right_ankle_joint: {p: 50.0, i: 0.01, d: 5.0}

left_arm_controller:
  type: position_controllers/JointTrajectoryController
  joints:
    - left_shoulder_joint
    - left_elbow_joint
    - left_wrist_joint
  gains:
    left_shoulder_joint: {p: 50.0, i: 0.01, d: 5.0}
    left_elbow_joint: {p: 50.0, i: 0.01, d: 5.0}
    left_wrist_joint: {p: 30.0, i: 0.01, d: 3.0}

right_arm_controller:
  type: position_controllers/JointTrajectoryController
  joints:
    - right_shoulder_joint
    - right_elbow_joint
    - right_wrist_joint
  gains:
    right_shoulder_joint: {p: 50.0, i: 0.01, d: 5.0}
    right_elbow_joint: {p: 50.0, i: 0.01, d: 5.0}
    right_wrist_joint: {p: 30.0, i: 0.01, d: 3.0}
```

## Launch File for Simulation

Create a launch file to start the complete simulation:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare arguments
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation time'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': PathJoinSubstitution([
                FindPackageShare('my_robot_package'),
                'worlds',
                'humanoid_world.sdf'
            ])
        }.items()
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'robot_description': open(PathJoinSubstitution([
                FindPackageShare('my_robot_package'),
                'urdf',
                'simple_humanoid.urdf'
            ]).perform({})).read()}
        ]
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0.0',
            '-y', '0.0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Load and start controllers
    joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_controller'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    left_leg_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_leg_controller'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    right_leg_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_leg_controller'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    left_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_arm_controller'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    right_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_arm_controller'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        use_sim_time,
        gazebo,
        robot_state_publisher,
        spawn_entity,
        joint_state_broadcaster,
        left_leg_controller,
        right_leg_controller,
        left_arm_controller,
        right_arm_controller
    ])
```

## Basic Walking Controller

Create a simple walking controller for the humanoid:

```python
#!/usr/bin/env python3

"""
Simple walking controller for the humanoid robot
This is a basic example - real walking controllers are much more complex
"""

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
import math
import time

class HumanoidWalkingController(Node):
    def __init__(self):
        super().__init__('humanoid_walking_controller')

        # Joint names for the legs
        self.left_leg_joints = [
            'left_hip_joint',
            'left_knee_joint',
            'left_ankle_joint'
        ]

        self.right_leg_joints = [
            'right_hip_joint',
            'right_knee_joint',
            'right_ankle_joint'
        ]

        # Publishers for joint trajectories
        self.left_leg_pub = self.create_publisher(
            JointTrajectory,
            '/left_leg_controller/joint_trajectory',
            10
        )

        self.right_leg_pub = self.create_publisher(
            JointTrajectory,
            '/right_leg_controller/joint_trajectory',
            10
        )

        # Timer for walking pattern
        self.walk_timer = self.create_timer(0.1, self.walk_callback)
        self.walk_phase = 0.0
        self.walk_frequency = 0.5  # Hz

        self.get_logger().info('Humanoid walking controller initialized')

    def walk_callback(self):
        """Generate walking pattern and publish trajectories"""
        self.walk_phase += 2 * math.pi * self.walk_frequency * 0.1

        # Generate walking pattern - this is a simplified example
        # Real walking controllers use much more sophisticated algorithms
        left_trajectory = self.generate_leg_trajectory(
            self.left_leg_joints,
            self.walk_phase,
            is_left=True
        )

        right_trajectory = self.generate_leg_trajectory(
            self.right_leg_joints,
            self.walk_phase + math.pi,  # Phase offset for alternating legs
            is_left=False
        )

        # Publish trajectories
        self.left_leg_pub.publish(left_trajectory)
        self.right_leg_pub.publish(right_trajectory)

    def generate_leg_trajectory(self, joint_names, phase, is_left=True):
        """Generate a simple walking trajectory for one leg"""
        trajectory = JointTrajectory()
        trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()

        # Simple walking pattern - in reality this would be much more complex
        # with proper inverse kinematics and balance control
        hip_angle = 0.2 * math.sin(phase)
        knee_angle = -0.3 * math.sin(phase + math.pi/2)
        ankle_angle = 0.1 * math.sin(phase)

        point.positions = [hip_angle, knee_angle, ankle_angle]

        # Set velocities and accelerations (for smooth motion)
        point.velocities = [0.0] * len(joint_names)
        point.accelerations = [0.0] * len(joint_names)

        # Set time from start (100ms in the future for smooth motion)
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100000000  # 100ms

        trajectory.points = [point]
        return trajectory

def main(args=None):
    rclpy.init(args=args)
    controller = HumanoidWalkingController()

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info('Shutting down walking controller')
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Unity Visualization Setup

For Unity visualization, create a script to connect to the ROS bridge and visualize the humanoid:

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Std_msgs;
using RosSharp.Messages.Sensor_msgs;
using RosSharp.Messages.Control_msgs;
using System.Collections.Generic;

public class HumanoidVisualizer : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeUrl = "ws://localhost:9090";

    [Header("Humanoid Model")]
    public GameObject humanoidModel;
    public Transform[] jointTransforms;  // Assign in inspector

    [Header("Sensor Visualization")]
    public GameObject cameraDisplay;
    public LineRenderer lidarRenderer;

    private RosSocket rosSocket;
    private Dictionary<string, Transform> jointMap = new Dictionary<string, Transform>();

    void Start()
    {
        ConnectToROS();
        SetupJointMap();
        SubscribeToTopics();
    }

    void ConnectToROS()
    {
        try
        {
            rosSocket = new RosSocket(new WebSocketSharpClient(rosBridgeUrl));
            Debug.Log("Connected to ROS Bridge");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to ROS: {e.Message}");
        }
    }

    void SetupJointMap()
    {
        // Map joint names to transforms
        if (humanoidModel != null)
        {
            // Find all joint transforms in the humanoid model
            Transform[] allTransforms = humanoidModel.GetComponentsInChildren<Transform>();

            foreach (Transform t in allTransforms)
            {
                // Assuming joint names match ROS joint names
                if (t.name.EndsWith("_joint") || t.name.EndsWith("_link"))
                {
                    jointMap[t.name] = t;
                }
            }
        }
    }

    void SubscribeToTopics()
    {
        if (rosSocket != null)
        {
            // Subscribe to joint states
            rosSocket.Subscribe<JointState>("/joint_states", ProcessJointStates);

            // Subscribe to sensor data
            rosSocket.Subscribe<Image>("/head_camera/image_raw", ProcessCameraData);
            rosSocket.Subscribe<LaserScan>("/scan", ProcessLidarData);
        }
    }

    void ProcessJointStates(JointState jointStateMsg)
    {
        if (jointStateMsg.name.Length != jointStateMsg.position.Length)
        {
            Debug.LogWarning("Joint names and positions array length mismatch");
            return;
        }

        for (int i = 0; i < jointStateMsg.name.Length; i++)
        {
            string jointName = jointStateMsg.name[i];
            double position = jointStateMsg.position[i];

            if (jointMap.ContainsKey(jointName))
            {
                Transform jointTransform = jointMap[jointName];

                // Update joint rotation based on position
                // This assumes revolute joints rotating around Z axis
                jointTransform.localRotation = Quaternion.Euler(0, 0, (float)position * Mathf.Rad2Deg);
            }
        }
    }

    void ProcessCameraData(Image imageMsg)
    {
        // Convert ROS image message to Unity texture and display
        // This would involve more complex image processing in a real implementation
    }

    void ProcessLidarData(LaserScan scanMsg)
    {
        // Visualize LiDAR data in Unity
        if (lidarRenderer != null)
        {
            int numPoints = Mathf.Min(scanMsg.ranges.Length, 100); // Limit for performance
            lidarRenderer.positionCount = numPoints;

            for (int i = 0; i < numPoints; i++)
            {
                float angle = scanMsg.angle_min + i * scanMsg.angle_increment;
                float distance = (float)scanMsg.ranges[i];

                if (distance > scanMsg.range_min && distance < scanMsg.range_max)
                {
                    Vector3 point = new Vector3(
                        distance * Mathf.Cos(angle),
                        0,
                        distance * Mathf.Sin(angle)
                    );
                    lidarRenderer.SetPosition(i, point);
                }
            }
        }
    }

    void OnDestroy()
    {
        rosSocket?.Close();
    }
}
```

## Running the Complete Simulation

### 1. Start Gazebo with the Humanoid
```bash
# Launch the complete simulation
ros2 launch my_robot_package humanoid_simulation.launch.py
```

### 2. Start the Walking Controller
```bash
# In another terminal
ros2 run my_robot_package humanoid_walking_controller
```

### 3. Visualize in Rviz
```bash
# For basic visualization
ros2 run rviz2 rviz2
```

### 4. Unity Visualization (Optional)
If you have the Unity project set up with ROS#:
1. Start the ROS bridge server:
```bash
# Install and run rosbridge_server
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```
2. Run the Unity application configured to connect to the ROS bridge

## Key Features of This Example

### 1. Complete Humanoid Model
- 14+ degrees of freedom
- Proper inertial properties
- Sensor integration (IMU, camera, LiDAR)
- Actuator transmission definitions

### 2. Realistic Simulation Environment
- Physics-accurate world with obstacles
- Proper lighting and rendering
- Collision detection and response

### 3. Sensor Integration
- IMU for balance and orientation
- Camera for vision processing
- LiDAR for environment mapping

### 4. Control Architecture
- ROS 2 control framework
- Joint trajectory controllers
- Basic walking pattern implementation

### 5. Visualization Options
- Gazebo for physics simulation
- Rviz for sensor data visualization
- Unity for advanced visualization (optional)

## Challenges and Considerations

### 1. Balance and Stability
Real humanoid robots require sophisticated balance control:
- Center of Mass (CoM) management
- Zero Moment Point (ZMP) control
- Feedback from IMU sensors

### 2. Computational Complexity
Humanoid simulation is computationally intensive:
- Optimize physics parameters
- Use appropriate update rates
- Consider simplifying models for real-time performance

### 3. Sensor Fusion
Combine multiple sensors for robust perception:
- IMU for orientation
- LiDAR for obstacle detection
- Camera for visual recognition

## Learning Objectives

After completing this example, you should understand:
- How to create complete humanoid robot models with proper kinematics
- How to set up a simulation environment with obstacles
- How to integrate multiple sensors for robot perception
- How to implement basic control patterns for humanoid robots
- How to visualize the robot in both Gazebo and Unity

## Next Steps

Continue to learn about [Sensor Data Visualization](../examples/sensor-data-visualization) to understand how to effectively visualize and interpret sensor data from your humanoid robot simulation.