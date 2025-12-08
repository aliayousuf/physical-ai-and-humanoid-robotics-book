---
title: "ROS 2 Communication Patterns: Nodes, Topics, Services"
description: "Understanding the fundamental communication mechanisms in ROS 2"
---

# ROS 2 Communication Patterns: Nodes, Topics, Services

## Overview

ROS 2 provides three primary communication patterns that enable nodes to interact with each other: nodes, topics, and services. Understanding these patterns is crucial for designing effective robotic systems.

## Nodes

A node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system. Each node typically performs a specific task and communicates with other nodes through topics, services, or actions.

### Creating Nodes

Nodes are created using client libraries like `rclpy` for Python or `rclcpp` for C++. A node must be initialized with a name and can contain publishers, subscribers, services, clients, and parameters.

### Node Responsibilities
- Publishing data to topics
- Subscribing to topics to receive data
- Providing services
- Calling services
- Managing parameters

## Topics - Publish/Subscribe Pattern

Topics implement a publish/subscribe communication pattern where publishers send data and subscribers receive it. This is a many-to-many relationship where multiple publishers can send to the same topic and multiple subscribers can listen to the same topic.

### Key Characteristics
- **Asynchronous**: Publishers and subscribers don't need to be synchronized
- **Unidirectional**: Data flows from publisher to subscriber
- **Message-based**: Communication happens through structured messages
- **Topic-based**: Identified by a unique name (e.g., `/cmd_vel`, `/laser_scan`)

### Use Cases
- Sensor data distribution (camera images, laser scans)
- Robot state broadcasting (joint states, odometry)
- Command distribution (velocity commands)

## Services - Request/Response Pattern

Services implement a request/response communication pattern. A client sends a request to a service server, which processes the request and returns a response. This is a synchronous, one-to-one relationship.

### Key Characteristics
- **Synchronous**: Client waits for response from server
- **Bidirectional**: Request goes to server, response comes back to client
- **Request/Response**: Structured data exchange with defined interface
- **Service-based**: Identified by a unique name (e.g., `/add_two_ints`, `/get_map`)

### Use Cases
- Map loading/unloading
- Coordinate transformations
- Parameter updates
- Action triggering

## Message Types

### Topic Messages
Messages sent via topics are defined in `.msg` files and contain only data fields. They follow a structured format with typed fields.

### Service Messages
Services use two message types: request and response, defined in `.srv` files. The structure defines both input and output.

## Quality of Service (QoS)

ROS 2 provides Quality of Service settings to control communication behavior:
- **Reliability**: Best effort vs. reliable delivery
- **Durability**: Volatile vs. transient local
- **History**: Keep all vs. keep last N messages
- **Deadline**: Maximum time between messages

## Practical Example

Consider a mobile robot:
- **Navigation node** publishes velocity commands to `/cmd_vel` topic
- **Motor controller node** subscribes to `/cmd_vel` and controls the motors
- **Map server node** provides a service `/static_map` to retrieve the map
- **Path planner node** calls the `/static_map` service to get the map for planning

## Best Practices

1. **Use topics** for continuous data streams like sensor data
2. **Use services** for infrequent, request-response interactions
3. **Follow naming conventions** (lowercase with underscores)
4. **Define appropriate message types** for your application
5. **Consider QoS settings** based on your application requirements
6. **Keep message sizes reasonable** for network efficiency

## Learning Objectives

After completing this section, you should understand:
- The differences between nodes, topics, and services
- When to use each communication pattern
- How to structure communication in a robot system
- The role of Quality of Service settings

## Next Steps

Continue to learn about [ROS 2 Actions](./actions) to understand how to handle long-running tasks with feedback.