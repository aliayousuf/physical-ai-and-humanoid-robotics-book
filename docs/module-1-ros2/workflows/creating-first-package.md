---
title: "Creating Your First ROS 2 Package"
description: "Step-by-step guide to creating and building your first ROS 2 package"
---

# Creating Your First ROS 2 Package

## Overview

A ROS 2 package is a reusable, self-contained unit of software that provides specific functionality. Packages contain nodes, libraries, data, and configuration files. This guide will walk you through creating your first ROS 2 package using the `colcon` build system.

## Prerequisites

Before creating your first package, ensure you have:
- ROS 2 installed (Humble Hawksbill or later)
- Properly sourced ROS 2 environment
- Basic knowledge of command line operations
- A workspace directory (e.g., `~/ros2_ws`)

## Setting Up Your Workspace

First, create and navigate to your workspace directory:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

## Creating a Package

Use the `ros2 pkg create` command to create a new package:

```bash
ros2 pkg create --build-type ament_python my_robot_package
```

For C++ packages, use:
```bash
ros2 pkg create --build-type ament_cmake my_robot_package
```

## Package Structure

After creation, your package will have this structure:

```
my_robot_package/
├── package.xml          # Package metadata
├── CMakeLists.txt       # Build configuration (for C++)
├── setup.py             # Python setup (for Python packages)
├── setup.cfg            # Installation configuration
├── my_robot_package/    # Python module directory
│   ├── __init__.py
│   └── my_node.py       # Example node file
└── test/                # Test files
    ├── test_copyright.py
    ├── test_flake8.py
    └── test_pep257.py
```

## Package.xml

The `package.xml` file contains metadata about your package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_package</name>
  <version>0.0.0</version>
  <description>Example ROS 2 package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

## Creating a Simple Node

Create a simple publisher node in `my_robot_package/my_robot_package/simple_publisher.py`:

```python
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class SimplePublisher(Node):
    def __init__(self):
        super().__init__('simple_publisher')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)
    simple_publisher = SimplePublisher()
    rclpy.spin(simple_publisher)
    simple_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Making the Node Executable

Make the Python file executable:

```bash
chmod +x my_robot_package/my_robot_package/simple_publisher.py
```

## Updating setup.py

Update the `setup.py` file to include your executable:

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
            'simple_publisher = my_robot_package.simple_publisher:main',
        ],
    },
)
```

## Building the Package

Navigate to your workspace root and build:

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_package
```

## Sourcing the Environment

After building, source the setup files:

```bash
source install/setup.bash
```

## Running the Node

Run your node:

```bash
ros2 run my_robot_package simple_publisher
```

## Verifying the Node

In another terminal, verify the node is publishing:

```bash
ros2 topic echo /chatter std_msgs/msg/String
```

## Best Practices

1. **Package naming**: Use lowercase with underscores (snake_case)
2. **Dependencies**: Only add necessary dependencies to `package.xml`
3. **Documentation**: Include README files and proper comments
4. **Testing**: Write tests for your nodes and functions
5. **Licensing**: Use appropriate open-source licenses
6. **Versioning**: Follow semantic versioning practices

## Common Issues and Solutions

### Build Issues
- Ensure all dependencies are listed in `package.xml`
- Check that `setup.py` has correct entry points
- Verify file permissions on executable scripts

### Runtime Issues
- Source the setup file after building
- Check that nodes are properly initialized
- Verify topic names and message types

## Learning Objectives

After completing this tutorial, you should be able to:
- Create a new ROS 2 package using the command line
- Understand the basic structure of a ROS 2 package
- Create and run a simple ROS 2 node
- Build and execute your package

## Next Steps

Continue to learn about [Launch Files Configuration](./launch-files) to learn how to start multiple nodes together.