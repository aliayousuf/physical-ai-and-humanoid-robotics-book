---
title: "ROS 2 Actions"
description: "Understanding actions for long-running tasks with feedback"
---

# ROS 2 Actions

## Overview

Actions are a communication pattern in ROS 2 designed for long-running tasks that require feedback and the ability to cancel. Unlike services, which are synchronous and provide a single response, actions are asynchronous and can provide continuous feedback while the task is executing.

## When to Use Actions

Actions are appropriate for tasks that:
- Take a significant amount of time to complete
- Need to provide feedback during execution
- Might need to be canceled before completion
- Have intermediate results worth reporting

## Action Structure

An action consists of three message types:
1. **Goal**: Defines the request sent to the action server
2. **Feedback**: Provides ongoing status updates during execution
3. **Result**: Contains the final outcome when the goal completes

## Action Client-Server Model

### Action Server
- Receives goals from clients
- Executes the requested task
- Sends feedback during execution
- Returns results when complete
- Handles cancellation requests

### Action Client
- Sends goals to the action server
- Receives feedback during execution
- Can cancel goals if needed
- Receives results when complete

## State Machine

Actions follow a specific state machine:
- **PENDING**: Goal accepted but not started
- **ACTIVE**: Goal is being processed
- **PREEMPTED**: Goal was canceled
- **SUCCEEDED**: Goal completed successfully
- **ABORTED**: Goal failed
- **RECALLED**: Goal was canceled before execution started
- **REJECTED**: Goal was rejected by the server

## Practical Example: Navigation

A navigation action might work as follows:
1. Client sends a "navigate to goal" goal
2. Server starts navigation and sends feedback with progress
3. Server continues sending feedback as robot moves
4. Server sends final result when robot reaches goal
5. Client can cancel the navigation if needed

## Comparison with Other Patterns

| Pattern | Synchronous | Feedback | Cancellation | Use Case |
|---------|-------------|----------|--------------|----------|
| Topics | No | Continuous | No | Streaming data |
| Services | Yes | No | No | Simple requests |
| Actions | No | Continuous | Yes | Long-running tasks |

## Action Interfaces

### Goal Interface
- Defines the parameters for the action
- Sent once when the action is requested
- Contains all necessary information to start the task

### Feedback Interface
- Sent periodically during execution
- Provides status updates
- Allows clients to monitor progress

### Result Interface
- Sent once when the action completes
- Contains the final outcome
- May include summary information

## Quality of Service for Actions

Actions have specific QoS settings for each component:
- Goal QoS: For goal requests
- Result QoS: For result responses
- Feedback QoS: For feedback messages
- Status QoS: For action status updates

## Best Practices

1. **Use actions** for tasks that take more than a few seconds
2. **Provide meaningful feedback** during long operations
3. **Handle cancellation gracefully** and stop work when requested
4. **Design clear goal/result interfaces** with all necessary information
5. **Set appropriate timeouts** to prevent hanging operations
6. **Use appropriate QoS settings** for your application requirements

## Learning Objectives

After completing this section, you should understand:
- When to use actions versus topics or services
- The structure and components of ROS 2 actions
- How to implement action clients and servers
- The action state machine and lifecycle

## Next Steps

Continue to learn about [URDF (Unified Robot Description Format)](./urdf) to understand how to describe robot structure.