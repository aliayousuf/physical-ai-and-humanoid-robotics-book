---
title: "Voice-to-Action Example"
description: "Complete example demonstrating voice command processing to robot action execution"
---

# Voice-to-Action Example

## Overview

This example demonstrates the complete pipeline from voice command to robot action execution in a Vision-Language-Action system. We'll build a simple but complete system that can understand voice commands like "Move the red cup to the table" and execute the corresponding actions.

## Complete Implementation

### 1. Voice Command Processing System

```python
import speech_recognition as sr
import threading
import queue
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class VoiceCommand:
    """Represents a processed voice command"""
    text: str
    intent: str
    entities: Dict[str, str]
    timestamp: float

class VoiceCommandProcessor:
    """Processes voice commands and converts them to structured actions"""

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.command_queue = queue.Queue()

        # Configure recognizer
        self.recognizer.energy_threshold = 4000
        self.recognizer.dynamic_energy_threshold = True

        # Define command patterns
        self.intent_patterns = {
            'navigation': [
                r'go to (.+)',
                r'move to (.+)',
                r'go to the (.+)',
                r'walk to (.+)'
            ],
            'manipulation': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'grab (.+)',
                r'move (.+) to (.+)',
                r'place (.+) on (.+)'
            ],
            'inspection': [
                r'find (.+)',
                r'look at (.+)',
                r'scan (.+)'
            ]
        }

        # Common objects and locations
        self.objects = ['cup', 'bottle', 'book', 'box', 'ball', 'remote']
        self.locations = ['kitchen', 'living room', 'bedroom', 'office', 'table', 'shelf']

    def start_listening(self):
        """Start listening for voice commands in a background thread"""
        listener_thread = threading.Thread(target=self._listen_continuously, daemon=True)
        listener_thread.start()
        return listener_thread

    def _listen_continuously(self):
        """Continuously listen for voice commands"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Voice command processor ready. Listening...")

        while True:
            try:
                with self.microphone as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                print(f"Recognized: {text}")

                # Process the command
                command = self.process_command(text)
                if command:
                    self.command_queue.put(command)

            except sr.WaitTimeoutError:
                # No speech detected, continue listening
                continue
            except sr.UnknownValueError:
                print("Could not understand audio")
                continue
            except sr.RequestError as e:
                print(f"Speech recognition error: {e}")
                time.sleep(1)  # Brief pause before retrying
                continue

    def process_command(self, text: str) -> Optional[VoiceCommand]:
        """Process text command and extract intent and entities"""
        text_lower = text.lower()

        # Determine intent
        intent = self._classify_intent(text_lower)

        # Extract entities
        entities = self._extract_entities(text_lower, intent)

        if intent != 'unknown':
            return VoiceCommand(
                text=text,
                intent=intent,
                entities=entities,
                timestamp=time.time()
            )

        return None

    def _classify_intent(self, text: str) -> str:
        """Classify the intent of the command"""
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                import re
                if re.search(pattern, text):
                    return intent

        return 'unknown'

    def _extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        """Extract entities from the command"""
        entities = {}

        # Extract objects based on intent
        if intent in ['manipulation', 'inspection']:
            for obj in self.objects:
                if obj in text:
                    entities['object'] = obj
                    break

        # Extract locations
        for location in self.locations:
            if location in text:
                entities['location'] = location
                break

        # Special handling for move commands (object to location)
        if intent == 'manipulation' and 'move' in text:
            import re
            move_match = re.search(r'move (.+?) to (.+)', text)
            if move_match:
                entities['object'] = move_match.group(1).strip()
                entities['destination'] = move_match.group(2).strip()

        return entities

    def get_command(self, timeout: float = None) -> Optional[VoiceCommand]:
        """Get the next command from the queue"""
        try:
            return self.command_queue.get(timeout=timeout)
        except queue.Empty:
            return None
```

### 2. Action Planning System

```python
@dataclass
class RobotAction:
    """Represents a robot action to be executed"""
    action_type: str  # 'navigation', 'manipulation', 'perception'
    action_name: str  # 'navigate_to', 'grasp_object', 'detect_object'
    parameters: Dict[str, any]
    priority: int = 1

class ActionPlanner:
    """Plans sequences of actions based on voice commands"""

    def __init__(self):
        self.object_locations = {
            'cup': 'table',
            'bottle': 'kitchen',
            'book': 'shelf'
        }

    def plan_actions(self, command: VoiceCommand) -> List[RobotAction]:
        """Plan actions based on the voice command"""
        actions = []

        if command.intent == 'navigation':
            actions.extend(self._plan_navigation_actions(command))
        elif command.intent == 'manipulation':
            actions.extend(self._plan_manipulation_actions(command))
        elif command.intent == 'inspection':
            actions.extend(self._plan_inspection_actions(command))

        return actions

    def _plan_navigation_actions(self, command: VoiceCommand) -> List[RobotAction]:
        """Plan navigation actions"""
        actions = []

        if 'location' in command.entities:
            location = command.entities['location']
            actions.append(RobotAction(
                action_type='navigation',
                action_name='navigate_to',
                parameters={'destination': location}
            ))

        return actions

    def _plan_manipulation_actions(self, command: VoiceCommand) -> List[RobotAction]:
        """Plan manipulation actions"""
        actions = []

        if 'object' in command.entities:
            obj = command.entities['object']

            # First, navigate to the object if location is known
            current_location = self.object_locations.get(obj)
            if current_location:
                actions.append(RobotAction(
                    action_type='navigation',
                    action_name='navigate_to',
                    parameters={'destination': current_location}
                ))

            # Then grasp the object
            actions.append(RobotAction(
                action_type='manipulation',
                action_name='grasp_object',
                parameters={'object': obj}
            ))

            # If there's a destination, navigate and place
            if 'destination' in command.entities:
                destination = command.entities['destination']
                actions.append(RobotAction(
                    action_type='navigation',
                    action_name='navigate_to',
                    parameters={'destination': destination}
                ))
                actions.append(RobotAction(
                    action_type='manipulation',
                    action_name='place_object',
                    parameters={'object': obj, 'destination': destination}
                ))

        return actions

    def _plan_inspection_actions(self, command: VoiceCommand) -> List[RobotAction]:
        """Plan inspection actions"""
        actions = []

        if 'object' in command.entities:
            obj = command.entities['object']
            actions.append(RobotAction(
                action_type='perception',
                action_name='detect_object',
                parameters={'object': obj}
            ))

        return actions
```

### 3. Robot Execution System

```python
import time
from enum import Enum

class RobotState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"
    MANIPULATING = "manipulating"
    PERCEIVING = "perceiving"
    ERROR = "error"

class RobotExecutor:
    """Executes robot actions in the real world"""

    def __init__(self):
        self.state = RobotState.IDLE
        self.current_position = {'x': 0.0, 'y': 0.0}
        self.held_object = None
        self.simulated_environment = {
            'kitchen': {'x': 5.0, 'y': 0.0},
            'living_room': {'x': 0.0, 'y': 5.0},
            'bedroom': {'x': -5.0, 'y': 0.0},
            'office': {'x': 0.0, 'y': -5.0},
            'table': {'x': 1.0, 'y': 1.0},
            'shelf': {'x': -1.0, 'y': -1.0}
        }

    def execute_action(self, action: RobotAction) -> bool:
        """Execute a single robot action"""
        print(f"Executing action: {action.action_name} with params {action.parameters}")

        if action.action_type == 'navigation':
            return self._execute_navigation(action)
        elif action.action_type == 'manipulation':
            return self._execute_manipulation(action)
        elif action.action_type == 'perception':
            return self._execute_perception(action)

        return False

    def _execute_navigation(self, action: RobotAction) -> bool:
        """Execute navigation action"""
        self.state = RobotState.NAVIGATING
        destination = action.parameters['destination']

        if destination in self.simulated_environment:
            target_pos = self.simulated_environment[destination]
            print(f"Navigating to {destination} at ({target_pos['x']}, {target_pos['y']})")

            # Simulate navigation time
            time.sleep(2)  # 2 seconds to navigate

            # Update current position
            self.current_position = target_pos
            print(f"Arrived at {destination}")

            self.state = RobotState.IDLE
            return True
        else:
            print(f"Unknown destination: {destination}")
            self.state = RobotState.ERROR
            return False

    def _execute_manipulation(self, action: RobotAction) -> bool:
        """Execute manipulation action"""
        self.state = RobotState.MANIPULATING
        action_name = action.action_name

        if action_name == 'grasp_object':
            obj = action.parameters['object']
            print(f"Grasping {obj}")

            # Simulate grasping time
            time.sleep(1)

            self.held_object = obj
            print(f"Successfully grasped {obj}")

        elif action_name == 'place_object':
            obj = action.parameters['object']
            destination = action.parameters['destination']
            print(f"Placing {obj} at {destination}")

            # Simulate placement time
            time.sleep(1)

            self.held_object = None
            print(f"Successfully placed {obj}")

        self.state = RobotState.IDLE
        return True

    def _execute_perception(self, action: RobotAction) -> bool:
        """Execute perception action"""
        self.state = RobotState.PERCEIVING
        action_name = action.action_name

        if action_name == 'detect_object':
            obj = action.parameters['object']
            print(f"Detecting {obj}")

            # Simulate detection time
            time.sleep(1)

            # Simulate detection result
            detected = obj in ['cup', 'bottle', 'book']  # Simple simulation
            if detected:
                print(f"Successfully detected {obj}")
            else:
                print(f"Could not detect {obj}")

            self.state = RobotState.IDLE
            return detected

        return False

    def execute_action_sequence(self, actions: List[RobotAction]) -> bool:
        """Execute a sequence of actions"""
        for action in actions:
            success = self.execute_action(action)
            if not success:
                print(f"Action failed: {action}")
                return False
            time.sleep(0.5)  # Brief pause between actions

        return True
```

### 4. Complete Voice-to-Action System

```python
class VoiceToActionSystem:
    """Complete system that integrates voice processing, planning, and execution"""

    def __init__(self):
        self.voice_processor = VoiceCommandProcessor()
        self.action_planner = ActionPlanner()
        self.robot_executor = RobotExecutor()
        self.is_running = False

    def start_system(self):
        """Start the complete voice-to-action system"""
        print("Starting Voice-to-Action System...")

        # Start voice processing in background
        self.voice_thread = self.voice_processor.start_listening()
        self.is_running = True

        print("System ready! Listening for commands...")

        # Main processing loop
        while self.is_running:
            try:
                # Get command from voice processor
                command = self.voice_processor.get_command(timeout=0.1)

                if command:
                    print(f"\nProcessing command: {command.text}")
                    print(f"Intent: {command.intent}, Entities: {command.entities}")

                    # Plan actions
                    actions = self.action_planner.plan_actions(command)
                    print(f"Planned actions: {[a.action_name for a in actions]}")

                    # Execute actions
                    success = self.robot_executor.execute_action_sequence(actions)

                    if success:
                        print("Command executed successfully!")
                    else:
                        print("Command execution failed!")

            except KeyboardInterrupt:
                print("\nShutting down system...")
                self.is_running = False
                break

    def stop_system(self):
        """Stop the voice-to-action system"""
        self.is_running = False
        print("System stopped.")

# Example usage and testing
def main():
    """Main function to run the voice-to-action system"""
    system = VoiceToActionSystem()

    print("Voice-to-Action System Demo")
    print("Available commands:")
    print("- 'Go to the kitchen'")
    print("- 'Pick up the cup'")
    print("- 'Move the bottle to the table'")
    print("- 'Find the book'")
    print("\nSpeak clearly into your microphone when ready.")
    print("Press Ctrl+C to stop the system.\n")

    try:
        system.start_system()
    except KeyboardInterrupt:
        print("\nSystem interrupted by user.")
        system.stop_system()

if __name__ == "__main__":
    main()
```

## Running the Example

### 1. Prerequisites

Install the required packages:

```bash
pip install speechrecognition pyaudio
```

### 2. Basic Usage

```python
# Create and run the system
system = VoiceToActionSystem()
system.start_system()
```

### 3. Example Commands

The system can understand and execute commands like:

- "Go to the kitchen" → Navigate to kitchen
- "Pick up the cup" → Navigate to cup location and grasp it
- "Move the bottle to the table" → Grasp bottle, navigate to table, place bottle
- "Find the book" → Detect book in environment

## Integration with ROS 2

For integration with ROS 2, you can wrap the system in a ROS 2 node:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class VoiceToActionROSNode(Node):
    def __init__(self):
        super().__init__('voice_to_action_node')

        # Initialize the voice-to-action system
        self.vta_system = VoiceToActionSystem()

        # Publishers
        self.status_pub = self.create_publisher(String, 'voice_status', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.command_sub = self.create_subscription(
            String, 'voice_commands', self.command_callback, 10
        )

        # Timer for processing
        self.timer = self.create_timer(0.1, self.process_commands)

        self.get_logger().info('Voice-to-Action ROS Node initialized')

    def command_callback(self, msg):
        """Handle incoming voice commands"""
        # Process the command through the VTA system
        command = self.vta_system.voice_processor.process_command(msg.data)
        if command:
            actions = self.vta_system.action_planner.plan_actions(command)
            success = self.vta_system.robot_executor.execute_action_sequence(actions)

            # Publish status
            status_msg = String()
            status_msg.data = f"Command '{msg.data}' {'succeeded' if success else 'failed'}"
            self.status_pub.publish(status_msg)

    def process_commands(self):
        """Process any queued commands"""
        command = self.vta_system.voice_processor.get_command(timeout=0.0)
        if command:
            actions = self.vta_system.action_planner.plan_actions(command)
            success = self.vta_system.robot_executor.execute_action_sequence(actions)

            status_msg = String()
            status_msg.data = f"Processed: {command.text} - {'Success' if success else 'Failed'}"
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VoiceToActionROSNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Key Features

1. **Real-time Voice Processing**: Continuously listens for and processes voice commands
2. **Intent Classification**: Understands navigation, manipulation, and inspection intents
3. **Entity Extraction**: Identifies objects and locations in commands
4. **Action Planning**: Converts high-level commands to sequences of robot actions
5. **Simulated Execution**: Demonstrates action execution in a simulated environment
6. **ROS 2 Integration**: Example of how to integrate with ROS 2 robotics framework

## Learning Outcomes

After implementing this example, you should understand:

- How to process voice commands in real-time
- Techniques for natural language understanding in robotics
- Action planning for voice-driven robot behaviors
- Integration of voice processing with robot execution systems
- ROS 2 integration patterns for voice-controlled robots

## Next Steps

This example can be extended with:

- More sophisticated natural language understanding
- Integration with real robot hardware
- Advanced perception for object detection and localization
- Multi-modal interaction combining voice, vision, and touch
- Context-aware command processing with memory of previous interactions