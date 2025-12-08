---
title: "LLM Cognitive Planning"
description: "Using Large Language Models for high-level task planning and action sequence generation"
---

# LLM Cognitive Planning

## Overview

Large Language Models (LLMs) can serve as cognitive planners for robotic systems, translating high-level human commands into executable action sequences. This approach enables robots to understand and act on natural language instructions like "Clean the room" by decomposing them into specific, achievable tasks.

## The Cognitive Planning Problem

Traditional robotics approaches require pre-programmed behaviors for specific tasks. Cognitive planning with LLMs allows robots to:
- Interpret high-level, abstract commands
- Decompose complex tasks into sub-tasks
- Generate action sequences dynamically
- Adapt to novel situations and environments

### Example Transformation
```
Input: "Clean the room"
Output:
1. Navigate to the kitchen
2. Pick up the bottle on the counter
3. Place bottle in the cabinet
4. Navigate to the living room
5. Pick up the book on the table
6. Place book on the shelf
```

## LLM Integration Architecture

### 1. Prompt Engineering for Robotics

Effective LLM prompting for robotics requires structured inputs:

```python
# Example prompt structure for cognitive planning
def create_planning_prompt(command, environment_state, robot_capabilities):
    prompt = f"""
    You are a cognitive planner for a humanoid robot. Given the following information:

    CURRENT ENVIRONMENT STATE:
    {environment_state}

    ROBOT CAPABILITIES:
    {robot_capabilities}

    TASK REQUEST:
    {command}

    Please decompose this high-level task into a sequence of specific, executable actions.
    Each action should be something the robot can directly execute.
    Format your response as a numbered list with one action per line.
    Only include actions that are physically possible given the environment state.

    ACTION SEQUENCE:
    """
    return prompt
```

### 2. Environment State Representation

For effective planning, LLMs need accurate environment state information:

```python
def format_environment_state(robot_pos, objects, obstacles, surfaces):
    """Format environment state for LLM consumption"""
    state_desc = f"""
    ROBOT POSITION: {robot_pos}
    DETECTED OBJECTS: {', '.join([f'{obj.name} at {obj.location}' for obj in objects])}
    NAVIGATION OBSTACLES: {', '.join([obs.name for obs in obstacles])}
    AVAILABLE SURFACES: {', '.join([surf.name for surf in surfaces])}
    """
    return state_desc.strip()
```

### 3. Robot Capability Description

LLMs need to understand what the robot can and cannot do:

```python
def format_robot_capabilities(mobility, manipulation, sensors):
    """Format robot capabilities for LLM planning"""
    capabilities = f"""
    MOBILITY: Can navigate to any accessible location in the environment
    MANIPULATION:
    - Can grasp objects within reach ({manipulation.reach_range}m)
    - Can lift objects up to {manipulation.weight_limit}kg
    - Can place objects on surfaces
    SENSORS: {sensors.description}
    """
    return capabilities.strip()
```

## Action Sequence Generation

### 1. Primitive Actions

Define a vocabulary of primitive actions the LLM can plan with:

```python
class RobotActionPrimitives:
    """Primitive actions for robot planning"""

    # Navigation actions
    NAVIGATE_TO = "navigate_to(location)"
    EXPLORE_AREA = "explore_area(area_name)"
    GO_HOME = "go_home()"

    # Manipulation actions
    GRASP_OBJECT = "grasp_object(object_name, location)"
    PLACE_OBJECT = "place_object(object_name, destination)"
    PUSH_OBJECT = "push_object(object_name, direction)"

    # Perception actions
    SCAN_ENVIRONMENT = "scan_environment()"
    LOCATE_OBJECT = "locate_object(object_name)"
    ASSESS_SURFACE = "assess_surface(surface_name)"

    # Compound actions
    PICK_AND_PLACE = "pick_and_place(object_name, source, destination)"
    CLEAN_AREA = "clean_area(area_name)"
    ORGANIZE_SPACE = "organize_space(space_name)"
```

### 2. Planning Validation

Validate LLM-generated plans for feasibility:

```python
class PlanValidator:
    def __init__(self, robot_model, environment):
        self.robot_model = robot_model
        self.environment = environment

    def validate_plan(self, action_sequence, initial_state):
        """Validate if the plan is feasible"""
        current_state = initial_state.copy()

        for i, action in enumerate(action_sequence):
            if not self.is_action_feasible(action, current_state):
                return False, f"Action {i+1} is not feasible: {action}"

            # Update state after action
            current_state = self.update_state(action, current_state)

        return True, "Plan is feasible"

    def is_action_feasible(self, action, state):
        """Check if action is physically possible in current state"""
        # Parse action and check feasibility
        action_name, params = self.parse_action(action)

        if action_name == "navigate_to":
            return self.can_navigate_to(params['location'], state)
        elif action_name == "grasp_object":
            return self.can_grasp_object(params['object_name'], params['location'], state)
        elif action_name == "place_object":
            return self.can_place_object(params['object_name'], params['destination'], state)

        return True  # Assume other actions are feasible for now

    def can_navigate_to(self, location, state):
        """Check if navigation to location is possible"""
        # Check if path exists and is not blocked
        path = self.robot_model.find_path(state['robot_position'], location)
        return path is not None and self.environment.is_path_clear(path)

    def can_grasp_object(self, object_name, location, state):
        """Check if object can be grasped from current position"""
        # Check reachability and object properties
        object_pos = state.get('objects', {}).get(object_name, {}).get('position')
        if not object_pos:
            return False

        distance = self.calculate_distance(state['robot_position'], object_pos)
        return distance <= self.robot_model.reach_distance

    def can_place_object(self, object_name, destination, state):
        """Check if object can be placed at destination"""
        # Check if destination is a valid surface and reachable
        if destination not in state.get('surfaces', {}):
            return False

        # Check if robot is holding the object
        holding = state.get('holding', None)
        return holding == object_name
```

## LLM Selection and Configuration

### 1. Model Considerations

For robotics cognitive planning, consider these LLM characteristics:

- **Reasoning Capability**: Ability to decompose complex tasks
- **Context Window**: Large enough to handle environment descriptions
- **Instruction Following**: Ability to follow structured prompts
- **Consistency**: Reliable output formatting for parsing

### 2. Configuration Parameters

```python
class LLMConfiguration:
    def __init__(self):
        self.temperature = 0.1  # Low temperature for consistent outputs
        self.max_tokens = 512   # Enough for detailed action sequences
        self.top_p = 0.9        # Balance creativity and consistency
        self.frequency_penalty = 0.5  # Reduce repetitive actions
        self.presence_penalty = 0.5   # Encourage diverse action selection
```

### 3. API Integration

```python
import openai
import json

class LLMPlanner:
    def __init__(self, api_key, model="gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def plan_task(self, command, environment_state, robot_capabilities):
        """Generate action sequence from natural language command"""
        prompt = create_planning_prompt(command, environment_state, robot_capabilities)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive planner for a humanoid robot. Generate executable action sequences."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )

            action_sequence = self.parse_llm_response(response.choices[0].message.content)
            return action_sequence

        except Exception as e:
            print(f"Error in LLM planning: {e}")
            return []

    def parse_llm_response(self, response_text):
        """Parse LLM response into structured action sequence"""
        lines = response_text.strip().split('\n')
        actions = []

        for line in lines:
            # Extract numbered actions (e.g., "1. Navigate to kitchen")
            if line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                action_text = line.split('.', 1)[1].strip()
                action = self.convert_to_primitive_action(action_text)
                if action:
                    actions.append(action)

        return actions

    def convert_to_primitive_action(self, action_text):
        """Convert natural language action to primitive robot action"""
        # This would use more sophisticated NLP/semantic matching
        # For now, simple keyword matching
        action_text_lower = action_text.lower()

        if 'navigate' in action_text_lower or 'go to' in action_text_lower:
            # Extract location using NLP
            location = self.extract_location(action_text)
            return f"navigate_to(location='{location}')"

        elif 'grasp' in action_text_lower or 'pick up' in action_text_lower:
            obj = self.extract_object(action_text)
            return f"grasp_object(object_name='{obj}')"

        elif 'place' in action_text_lower or 'put' in action_text_lower:
            obj = self.extract_object(action_text)
            dest = self.extract_destination(action_text)
            return f"place_object(object_name='{obj}', destination='{dest}')"

        # Add more action mappings as needed
        return action_text  # Return original if no mapping found

    def extract_location(self, action_text):
        """Extract location from action text using NLP"""
        # In practice, this would use more sophisticated NLP
        # For now, simple keyword extraction
        keywords = ['kitchen', 'living room', 'bedroom', 'office', 'hallway', 'dining room']
        for keyword in keywords:
            if keyword in action_text.lower():
                return keyword
        return "unknown_location"

    def extract_object(self, action_text):
        """Extract object from action text using NLP"""
        # In practice, this would use more sophisticated NLP
        # For now, simple keyword extraction
        object_keywords = ['bottle', 'cup', 'book', 'box', 'chair', 'table']
        for keyword in object_keywords:
            if keyword in action_text.lower():
                return keyword
        return "unknown_object"

    def extract_destination(self, action_text):
        """Extract destination from action text using NLP"""
        # In practice, this would use more sophisticated NLP
        # For now, simple keyword extraction
        dest_keywords = ['shelf', 'cabinet', 'counter', 'table', 'floor']
        for keyword in dest_keywords:
            if keyword in action_text.lower():
                return keyword
        return "unknown_destination"
```

## Safety and Error Handling

### 1. Plan Safety Checking

```python
class PlanSafetyChecker:
    def __init__(self):
        self.forbidden_actions = [
            "touch_person", "enter_restricted_area", "handle_dangerous_object"
        ]
        self.risky_conditions = [
            "low_battery", "unstable_surface", "crowded_area"
        ]

    def check_plan_safety(self, action_sequence, environment_state):
        """Check if plan contains unsafe actions"""
        issues = []

        for action in action_sequence:
            if self.is_forbidden_action(action):
                issues.append(f"Forbidden action: {action}")

            if self.has_safety_concern(action, environment_state):
                issues.append(f"Safety concern with action: {action}")

        return len(issues) == 0, issues

    def is_forbidden_action(self, action):
        """Check if action is forbidden"""
        for forbidden in self.forbidden_actions:
            if forbidden in action.lower():
                return True
        return False

    def has_safety_concern(self, action, state):
        """Check if action has safety concerns given environment state"""
        # Check for risky conditions in environment state
        for condition in self.risky_conditions:
            if condition in state.lower():
                # Check if action is risky under this condition
                if self.is_risky_under_condition(action, condition):
                    return True
        return False

    def is_risky_under_condition(self, action, condition):
        """Check if action is risky under specific condition"""
        if condition == "low_battery" and "navigate" in action.lower():
            return True  # Risky to navigate with low battery
        if condition == "unstable_surface" and "place" in action.lower():
            return True  # Risky to place objects on unstable surface
        return False
```

## Integration with Robot Control

### 1. Planning-Execution Interface

```python
class PlanningExecutionInterface:
    def __init__(self, llm_planner, robot_controller, environment_monitor):
        self.llm_planner = llm_planner
        self.robot_controller = robot_controller
        self.environment_monitor = environment_monitor
        self.plan_validator = PlanValidator(robot_controller.model, environment_monitor)
        self.safety_checker = PlanSafetyChecker()

    def execute_high_level_command(self, command):
        """Execute a high-level natural language command"""
        # Get current environment state
        env_state = self.environment_monitor.get_state()

        # Get robot capabilities
        capabilities = self.robot_controller.get_capabilities()

        # Generate plan using LLM
        action_sequence = self.llm_planner.plan_task(command, env_state, capabilities)

        # Validate plan
        is_valid, validation_msg = self.plan_validator.validate_plan(action_sequence, env_state)
        if not is_valid:
            print(f"Plan validation failed: {validation_msg}")
            return False

        # Check plan safety
        is_safe, safety_issues = self.safety_checker.check_plan_safety(action_sequence, env_state)
        if not is_safe:
            print(f"Plan safety issues: {safety_issues}")
            return False

        # Execute plan step by step
        for i, action in enumerate(action_sequence):
            print(f"Executing action {i+1}/{len(action_sequence)}: {action}")

            success = self.robot_controller.execute_action(action)
            if not success:
                print(f"Action failed: {action}")
                return False

            # Update environment state after each action
            env_state = self.environment_monitor.get_state()

        print("Command completed successfully!")
        return True
```

## Performance Optimization

### 1. Caching and Learning

```python
class CachedPlanner:
    def __init__(self, base_planner):
        self.base_planner = base_planner
        self.cache = {}
        self.learning_enabled = True

    def plan_task(self, command, environment_state, robot_capabilities):
        """Plan task with caching for repeated commands"""
        # Create cache key based on command and simplified environment state
        cache_key = self.create_cache_key(command, environment_state, robot_capabilities)

        if cache_key in self.cache:
            print("Using cached plan")
            return self.cache[cache_key]

        # Generate new plan
        plan = self.base_planner.plan_task(command, environment_state, robot_capabilities)

        # Store in cache if learning is enabled
        if self.learning_enabled:
            self.cache[cache_key] = plan

        return plan

    def create_cache_key(self, command, environment_state, robot_capabilities):
        """Create cache key that ignores minor environmental changes"""
        # Simplify environment state to focus on relevant aspects
        simplified_env = self.simplify_environment_state(environment_state)
        return f"{command}_{simplified_env}_{robot_capabilities}"

    def simplify_environment_state(self, env_state):
        """Simplify environment state for caching purposes"""
        # Only include major environmental features for caching
        # This allows reuse of plans when only minor details change
        return env_state  # In practice, would extract key features
```

## Learning Objectives

After completing this section, you should understand:
- How to structure prompts for effective LLM-based planning
- How to represent environment state for LLM consumption
- How to validate and check safety of generated action sequences
- How to integrate LLM planning with robot execution systems
- The considerations for using LLMs in robotics applications

## Next Steps

Continue to learn about [Multi-Modal Fusion](./multi-modal-fusion) to understand how to combine vision, language, and action modalities in a unified system.