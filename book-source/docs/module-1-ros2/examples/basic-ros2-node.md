---
title: "Basic ROS 2 Node Example"
description: "Complete working example with publisher and subscriber"
---

# Basic ROS 2 Node Example

## Overview

This example demonstrates a complete ROS 2 system with a publisher node that sends messages and a subscriber node that receives them. This is the "Hello World" of ROS 2 and introduces the fundamental concepts of node creation, message publishing, and subscription.

## Complete Publisher Node

Create `my_robot_package/my_robot_package/publisher_member_function.py`:

```python
#!/usr/bin/env python3

"""
Publisher node example that sends messages at regular intervals.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalPublisher(Node):
    """
    A publisher node that sends messages with a counter.
    """
    def __init__(self):
        super().__init__('minimal_publisher')

        # Create a publisher for the 'topic' topic with String messages
        self.publisher = self.create_publisher(String, 'topic', 10)

        # Create a timer that triggers the timer_callback every 0.5 seconds
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter to track message number
        self.i = 0

        # Log that the publisher has started
        self.get_logger().info('Minimal publisher node started')

    def timer_callback(self):
        """
        Callback function executed by the timer at regular intervals.
        """
        # Create a new String message
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # Publish the message
        self.publisher.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        # Increment the counter
        self.i += 1


def main(args=None):
    """
    Main function to initialize and run the publisher node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the publisher node
    minimal_publisher = MinimalPublisher()

    try:
        # Keep the node running and processing callbacks
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_publisher.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        # Clean up the node
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Complete Subscriber Node

Create `my_robot_package/my_robot_package/subscriber_member_function.py`:

```python
#!/usr/bin/env python3

"""
Subscriber node example that receives and logs messages.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MinimalSubscriber(Node):
    """
    A subscriber node that receives and logs messages.
    """
    def __init__(self):
        super().__init__('minimal_subscriber')

        # Create a subscription to the 'topic' topic with String messages
        self.subscription = self.create_subscription(
            String,
            'topic',  # Topic name
            self.listener_callback,  # Callback function
            10)  # QoS history depth

        # Prevent unused variable warning
        self.subscription  # Only needed for Python style checkers

        # Log that the subscriber has started
        self.get_logger().info('Minimal subscriber node started')

    def listener_callback(self, msg):
        """
        Callback function executed when a message is received.
        """
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    """
    Main function to initialize and run the subscriber node.
    """
    # Initialize the ROS 2 client library
    rclpy.init(args=args)

    # Create the subscriber node
    minimal_subscriber = MinimalSubscriber()

    try:
        # Keep the node running and processing callbacks
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        minimal_subscriber.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        # Clean up the node
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Package Configuration

Update the `setup.py` file to include the new executables:

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
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Example ROS 2 package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = my_robot_package.publisher_member_function:main',
            'listener = my_robot_package.subscriber_member_function:main',
        ],
    },
)
```

Make the Python files executable:

```bash
chmod +x my_robot_package/my_robot_package/publisher_member_function.py
chmod +x my_robot_package/my_robot_package/subscriber_member_function.py
```

## Building the Package

After creating the nodes, rebuild your package:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
source install/setup.bash
```

## Running the Example

### Terminal 1 - Publisher:
```bash
ros2 run my_robot_package talker
```

### Terminal 2 - Subscriber:
```bash
ros2 run my_robot_package listener
```

You should see the publisher sending messages and the subscriber receiving them:

Publisher output:
```
[INFO] [1612345678.123456789] [minimal_publisher]: Publishing: "Hello World: 0"
[INFO] [1612345678.623456789] [minimal_publisher]: Publishing: "Hello World: 1"
[INFO] [1612345679.123456789] [minimal_publisher]: Publishing: "Hello World: 2"
```

Subscriber output:
```
[INFO] [1612345678.123456789] [minimal_subscriber]: I heard: "Hello World: 0"
[INFO] [1612345678.623456789] [minimal_subscriber]: I heard: "Hello World: 1"
[INFO] [1612345679.123456789] [minimal_subscriber]: I heard: "Hello World: 2"
```

## Alternative: Single Node with Both Publisher and Subscriber

You can also create a single node that both publishes and subscribes:

```python
#!/usr/bin/env python3

"""
Node that both publishes and subscribes to demonstrate bidirectional communication.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class TalkerListenerNode(Node):
    """
    A node that demonstrates both publishing and subscribing.
    """
    def __init__(self):
        super().__init__('talker_listener')

        # Create publisher
        self.publisher = self.create_publisher(String, 'chatter', 10)

        # Create subscriber
        self.subscription = self.create_subscription(
            String,
            'chatter',
            self.listener_callback,
            10)

        # Create timer for publishing
        timer_period = 1.0  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.i = 0

        self.get_logger().info('Talker-Listener node started')

    def timer_callback(self):
        """
        Publish a message at regular intervals.
        """
        msg = String()
        msg.data = f'Hello from talker_listener: {self.i}'

        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')

        self.i += 1

    def listener_callback(self, msg):
        """
        Handle received messages.
        """
        self.get_logger().info(f'Received: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = TalkerListenerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt, shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Understanding the Components

### Node Creation
- `rclpy.init()` initializes the ROS 2 client library
- `Node()` creates a node with a unique name
- `rclpy.spin()` keeps the node running and processes callbacks

### Publisher
- `create_publisher()` creates a publisher for a specific topic
- `publish()` sends messages to the topic
- The second parameter is the message type (e.g., `String`)

### Subscriber
- `create_subscription()` creates a subscription to a topic
- The callback function is executed when a message is received
- The third parameter is the QoS history depth

### Timer
- `create_timer()` creates a timer that executes a callback at regular intervals
- Useful for periodic publishing or processing

## Common Issues and Solutions

### Topic Not Connecting
- Ensure both nodes are on the same topic name
- Check that the message types match
- Verify that nodes are in the same ROS domain (if using multiple systems)

### Node Names Conflicts
- Use unique node names to avoid conflicts
- Consider using namespaces for multiple instances

### Build Issues
- Ensure all dependencies are listed in `package.xml`
- Check that `setup.py` has correct entry points
- Verify file permissions on executable scripts

## Extending the Example

### Adding Parameters
```python
# In the node constructor
self.declare_parameter('publish_rate', 0.5)
rate = self.get_parameter('publish_rate').value
self.timer = self.create_timer(rate, self.timer_callback)
```

### Using Custom Message Types
```python
# Instead of std_msgs.msg.String
from my_robot_package_interfaces.msg import CustomMessage
```

### Adding Services
```python
# In the node constructor
self.service = self.create_service(
    AddTwoInts,
    'add_two_ints',
    self.add_two_ints_callback
)
```

## Learning Objectives

After completing this example, you should be able to:
- Create complete ROS 2 nodes with publishers and subscribers
- Understand the structure and components of ROS 2 nodes
- Build and run ROS 2 nodes with proper executable configuration
- Debug common issues with node communication

## Next Steps

Continue to learn about [URDF Robot Definition Example](../examples/urdf-robot-definition) to see how to create a simple robot model using URDF.