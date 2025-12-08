---
title: "Python-ROS Integration with rclpy"
description: "Bridging Python agents to ROS controllers using rclpy"
---

# Python-ROS Integration with rclpy

## Overview

`rclpy` is the Python client library for ROS 2 that allows Python programs to interact with the ROS 2 system. It provides the necessary functionality to create nodes, publish and subscribe to topics, provide and call services, and manage parameters. This guide covers how to bridge Python agents to ROS controllers using rclpy.

## What is rclpy?

`rclpy` is part of the ROS 2 client library ecosystem. It provides:
- Python bindings to the ROS 2 client library (rcl)
- Node creation and management
- Publisher and subscriber functionality
- Service and action client/server interfaces
- Parameter management
- Time and duration utilities
- Logging capabilities

## Installation and Setup

`rclpy` is included with ROS 2 installations. To use it in your Python package, add it as a dependency in your `package.xml`:

```xml
<depend>rclpy</depend>
```

And in your Python code:

```python
import rclpy
from rclpy.node import Node
```

## Basic Node Structure

Every ROS 2 Python node follows this basic structure:

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        super().__init__('node_name')
        # Initialize publishers, subscribers, etc.

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Publishers

Publishers send messages to topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher = self.create_publisher(String, 'topic_name', 10)
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
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Subscribers

Subscribers receive messages from topics:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic_name',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Services

Services provide request/response communication:

### Service Server

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    minimal_service.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Actions

Actions provide feedback for long-running tasks:

### Action Server

```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Result: {result.sequence}')
        return result

def main(args=None):
    rclpy.init(args=args)
    fibonacci_action_server = FibonacciActionServer()
    rclpy.spin(fibonacci_action_server)
    fibonacci_action_server.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Action Client

```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

    def send_goal(self, order):
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

def main(args=None):
    rclpy.init(args=args)
    action_client = FibonacciActionClient()
    action_client.send_goal(10)
    rclpy.spin(action_client)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Parameters

Nodes can have parameters that can be configured at runtime:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('param1', 'default_value')
        self.declare_parameter('param2', 42)
        self.declare_parameter('param3', True)

        # Get parameter values
        param1_value = self.get_parameter('param1').value
        param2_value = self.get_parameter('param2').value
        param3_value = self.get_parameter('param3').value

        self.get_logger().info(f'param1: {param1_value}')
        self.get_logger().info(f'param2: {param2_value}')
        self.get_logger().info(f'param3: {param3_value}')

def main(args=None):
    rclpy.init(args=args)
    parameter_node = ParameterNode()
    rclpy.spin(parameter_node)
    parameter_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Advanced Features

### Quality of Service (QoS)

Customize publisher/subscriber behavior:

```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

# Create a custom QoS profile
qos_profile = QoSProfile(
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE
)

# Use it in publisher/subscriber
publisher = self.create_publisher(String, 'topic_name', qos_profile)
```

### Timers

Execute code at regular intervals:

```python
def __init__(self):
    super().__init__('timer_node')
    self.timer = self.create_timer(0.5, self.timer_callback)  # 0.5 seconds

def timer_callback(self):
    self.get_logger().info('Timer callback executed')
```

### Logging

Use different logging levels:

```python
self.get_logger().debug('Debug message')
self.get_logger().info('Info message')
self.get_logger().warn('Warning message')
self.get_logger().error('Error message')
self.get_logger().fatal('Fatal message')
```

## Best Practices for Python-ROS Integration

1. **Use type hints**: Make your code more readable and maintainable
2. **Handle exceptions**: Wrap ROS calls in try-catch blocks when appropriate
3. **Use proper shutdown**: Always call `destroy_node()` and `rclpy.shutdown()`
4. **Manage resources**: Properly clean up publishers, subscribers, and other resources
5. **Follow naming conventions**: Use snake_case for node names and topics
6. **Document interfaces**: Clearly document message types and expected behavior
7. **Use QoS appropriately**: Choose QoS settings based on your application needs
8. **Test thoroughly**: Verify your nodes work in different scenarios

## Common Integration Patterns

### Bridge Pattern
Create a node that translates between different message types or protocols:

```python
class MessageBridge(Node):
    def __init__(self):
        super().__init__('message_bridge')
        self.subscriber = self.create_subscription(
            OriginalMessage, 'input_topic', self.callback, 10)
        self.publisher = self.create_publisher(
            NewMessage, 'output_topic', 10)

    def callback(self, msg):
        # Convert message format
        new_msg = self.convert_message(msg)
        self.publisher.publish(new_msg)
```

### Agent Wrapper
Wrap a Python agent in a ROS node:

```python
class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_node')
        self.agent = MyPythonAgent()  # Your Python agent
        self.subscriber = self.create_subscription(
            AgentCommand, 'agent_command', self.command_callback, 10)
        self.publisher = self.create_publisher(
            AgentState, 'agent_state', 10)

    def command_callback(self, cmd):
        result = self.agent.process_command(cmd)
        state_msg = self.create_state_message(result)
        self.publisher.publish(state_msg)
```

## Learning Objectives

After completing this tutorial, you should be able to:
- Create ROS 2 nodes using rclpy
- Implement publishers, subscribers, services, and actions
- Manage parameters in ROS 2 nodes
- Use advanced features like QoS and timers
- Bridge Python agents to ROS controllers effectively

## Next Steps

Continue to learn about [Basic ROS 2 Node Example](../examples/basic-ros2-node) to see a complete working example with publisher and subscriber.