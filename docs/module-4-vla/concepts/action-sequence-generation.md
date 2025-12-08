---
title: "Action Sequence Generation"
description: "Generating executable action sequences from high-level commands"
---

# Action Sequence Generation

## Overview

Action sequence generation is the process of converting high-level commands and plans into specific, executable robot actions. This involves breaking down complex tasks into primitive operations that the robot can perform sequentially. In the context of Vision-Language-Action systems, this process bridges the gap between natural language understanding and physical execution.

## The Action Generation Pipeline

The action sequence generation process follows this flow:

```
High-Level Command → Semantic Parsing → Task Decomposition → Action Primitives → Execution Plan → Robot Control
```

### 1. Semantic Parsing

Convert natural language commands into structured representations:

```python
import re
import spacy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ActionPrimitive:
    """Represents a primitive robot action"""
    name: str
    parameters: Dict[str, Any]
    duration: float  # Estimated execution time in seconds
    preconditions: List[str]  # Conditions that must be met before execution
    effects: List[str]      # Effects on the world state after execution

@dataclass
class TaskDecomposition:
    """Represents a decomposed task with subtasks"""
    task_name: str
    subtasks: List['TaskDecomposition']
    primitive_actions: List[ActionPrimitive]
    dependencies: List[str]  # Other tasks this task depends on

class SemanticParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm") if spacy.util.is_package("en_core_web_sm") else None

        # Define action templates and patterns
        self.action_templates = {
            'navigation': {
                'patterns': [
                    r'(go|move|navigate|walk|travel) to (.+)',
                    r'move to the (.+)',
                    r'go to the (.+) room',
                    r'travel to (.+)'
                ],
                'action_type': 'navigate_to'
            },
            'manipulation': {
                'patterns': [
                    r'(pick up|grasp|grab|lift) (.+)',
                    r'(place|put|set) (.+) (on|in|at) (.+)',
                    r'bring (.+) to (.+)',
                    r'move (.+) from (.+) to (.+)'
                ],
                'action_type': 'manipulate_object'
            },
            'inspection': {
                'patterns': [
                    r'(look at|examine|inspect|check|scan) (.+)',
                    r'find (.+)',
                    r'locate (.+)',
                    r'identify (.+)'
                ],
                'action_type': 'inspect_object'
            }
        }

    def parse_command(self, command: str) -> Dict[str, Any]:
        """Parse a natural language command into structured representation"""
        command_lower = command.lower().strip()

        # Try pattern matching first
        for action_type, config in self.action_templates.items():
            for pattern in config['patterns']:
                match = re.search(pattern, command_lower)
                if match:
                    return {
                        'action_type': config['action_type'],
                        'arguments': match.groups(),
                        'confidence': 0.9  # High confidence for pattern match
                    }

        # If no pattern matches, use NLP parsing
        return self._nlp_parse(command)

    def _nlp_parse(self, command: str) -> Dict[str, Any]:
        """Use NLP to parse command when pattern matching fails"""
        if not self.nlp:
            return {'action_type': 'unknown', 'arguments': [], 'confidence': 0.0}

        doc = self.nlp(command)

        # Extract main verb (action)
        verbs = [token for token in doc if token.pos_ == 'VERB']
        if not verbs:
            return {'action_type': 'unknown', 'arguments': [], 'confidence': 0.0}

        main_verb = verbs[0]
        action_type = self._verb_to_action(main_verb.lemma_)

        # Extract arguments (objects, locations, etc.)
        arguments = []
        for token in doc:
            if token.dep_ in ['dobj', 'pobj', 'attr']:  # Direct object, prepositional object, attribute
                arguments.append(token.text)

        # Extract prepositional phrases for locations
        prep_phrases = []
        for token in doc:
            if token.pos_ == 'ADP':  # Preposition
                if token.head.pos_ == 'VERB':
                    # Get the object of the preposition
                    for child in token.children:
                        if child.dep_ == 'pobj':
                            prep_phrases.append(f"{token.text} {child.text}")

        return {
            'action_type': action_type,
            'arguments': arguments,
            'prep_phrases': prep_phrases,
            'confidence': 0.7  # Lower confidence for NLP parsing
        }

    def _verb_to_action(self, verb: str) -> str:
        """Convert verb to action type"""
        verb_to_action_map = {
            'go': 'navigate_to',
            'move': 'navigate_to',
            'navigate': 'navigate_to',
            'walk': 'navigate_to',
            'travel': 'navigate_to',
            'pick': 'grasp_object',
            'grasp': 'grasp_object',
            'grab': 'grasp_object',
            'lift': 'grasp_object',
            'place': 'place_object',
            'put': 'place_object',
            'set': 'place_object',
            'look': 'inspect_object',
            'examine': 'inspect_object',
            'inspect': 'inspect_object',
            'check': 'inspect_object',
            'scan': 'inspect_object',
            'find': 'locate_object',
            'locate': 'locate_object',
            'identify': 'identify_object'
        }

        return verb_to_action_map.get(verb, 'unknown')
```

### 2. Task Decomposition

Break down complex tasks into manageable subtasks:

```python
class TaskDecomposer:
    def __init__(self):
        self.knowledge_base = self._initialize_knowledge_base()
        self.robot_capabilities = self._load_robot_capabilities()

    def _initialize_knowledge_base(self) -> Dict[str, Any]:
        """Initialize knowledge base with task decompositions"""
        return {
            'clean_room': {
                'subtasks': [
                    'survey_room',
                    'identify_dirty_items',
                    'navigate_to_item',
                    'pick_up_item',
                    'place_item_properly',
                    'repeat_until_clean'
                ]
            },
            'set_table': {
                'subtasks': [
                    'identify_table_location',
                    'identify_items_to_place',
                    'navigate_to_storage',
                    'pick_up_item',
                    'navigate_to_table',
                    'place_item_on_table',
                    'repeat_for_all_items'
                ]
            },
            'find_object': {
                'subtasks': [
                    'identify_object',
                    'check_known_locations',
                    'navigate_to_search_area',
                    'scan_for_object',
                    'confirm_detection',
                    'approach_object'
                ]
            }
        }

    def _load_robot_capabilities(self) -> Dict[str, Any]:
        """Load robot capabilities for constraint checking"""
        return {
            'navigation': {
                'reachable_areas': ['kitchen', 'living_room', 'bedroom', 'office', 'hallway'],
                'max_speed': 1.0,  # m/s
                'turn_speed': 0.5  # rad/s
            },
            'manipulation': {
                'max_reach': 1.0,  # meters
                'weight_limit': 5.0,  # kg
                'grasp_types': ['pinch', 'power', 'precision']
            },
            'sensing': {
                'camera_range': 10.0,  # meters
                'lidar_range': 20.0,   # meters
                'object_detection': ['person', 'chair', 'table', 'bottle', 'cup', 'book']
            }
        }

    def decompose_task(self, command: str, parsed_command: Dict[str, Any]) -> TaskDecomposition:
        """Decompose high-level task into subtasks and primitives"""
        # First, try to match the command to a known task template
        task_template = self._match_task_template(command)

        if task_template:
            return self._decompose_known_task(task_template, parsed_command)
        else:
            return self._decompose_generic_task(parsed_command)

    def _match_task_template(self, command: str) -> Optional[str]:
        """Match command to known task templates"""
        command_lower = command.lower()

        for template_name in self.knowledge_base.keys():
            if template_name in command_lower:
                return template_name

        return None

    def _decompose_known_task(self, template_name: str, parsed_command: Dict[str, Any]) -> TaskDecomposition:
        """Decompose a known task using template"""
        template = self.knowledge_base[template_name]
        subtasks = []

        for subtask_name in template['subtasks']:
            # Create subtask decomposition
            subtask = TaskDecomposition(
                task_name=subtask_name,
                subtasks=[],
                primitive_actions=self._create_primitives_for_subtask(subtask_name, parsed_command),
                dependencies=[]
            )
            subtasks.append(subtask)

        return TaskDecomposition(
            task_name=template_name,
            subtasks=subtasks,
            primitive_actions=[],
            dependencies=[]
        )

    def _create_primitives_for_subtask(self, subtask_name: str, parsed_command: Dict[str, Any]) -> List[ActionPrimitive]:
        """Create primitive actions for a subtask"""
        if subtask_name == 'navigate_to_item':
            return [ActionPrimitive(
                name='navigate_to',
                parameters={'location': parsed_command.get('arguments', [''])[0]},
                duration=10.0,
                preconditions=['robot_is_idle', 'path_is_clear'],
                effects=['robot_position_changed']
            )]
        elif subtask_name == 'pick_up_item':
            return [ActionPrimitive(
                name='grasp_object',
                parameters={'object': parsed_command.get('arguments', [''])[0]},
                duration=5.0,
                preconditions=['object_in_reach', 'hand_is_free'],
                effects=['object_grasped', 'hand_is_occupied']
            )]
        elif subtask_name == 'place_item_properly':
            return [ActionPrimitive(
                name='place_object',
                parameters={'object': parsed_command.get('arguments', [''])[0], 'location': parsed_command.get('prep_phrases', [''])[0]},
                duration=3.0,
                preconditions=['object_grasped', 'destination_reachable'],
                effects=['object_placed', 'hand_is_free']
            )]

        # Default return empty list
        return []
```

## Planning and Scheduling

### 1. Sequential Planning

Create sequential execution plans:

```python
class SequentialPlanner:
    def __init__(self):
        self.action_library = self._load_action_library()
        self.constraint_solver = ConstraintSolver()

    def _load_action_library(self) -> Dict[str, Dict[str, Any]]:
        """Load library of available actions with their properties"""
        return {
            'navigate_to': {
                'parameters': ['location'],
                'duration': 10.0,
                'preconditions': ['robot_is_operational'],
                'effects': ['robot_at_location'],
                'cost': 1.0
            },
            'grasp_object': {
                'parameters': ['object'],
                'duration': 5.0,
                'preconditions': ['object_visible', 'object_reachable', 'gripper_free'],
                'effects': ['object_grasped', 'gripper_occupied'],
                'cost': 2.0
            },
            'place_object': {
                'parameters': ['object', 'destination'],
                'duration': 3.0,
                'preconditions': ['object_grasped', 'destination_reachable'],
                'effects': ['object_placed', 'gripper_free'],
                'cost': 1.5
            },
            'inspect_object': {
                'parameters': ['object'],
                'duration': 2.0,
                'preconditions': ['object_visible'],
                'effects': ['object_properties_known'],
                'cost': 0.5
            }
        }

    def create_execution_plan(self, task_decomposition: TaskDecomposition) -> List[ActionPrimitive]:
        """Create sequential execution plan from task decomposition"""
        plan = []

        # Flatten the task decomposition into a sequence of primitive actions
        for subtask in task_decomposition.subtasks:
            plan.extend(subtask.primitive_actions)

        # Add any top-level primitive actions
        plan.extend(task_decomposition.primitive_actions)

        # Validate the plan
        is_valid, errors = self.validate_plan(plan)
        if not is_valid:
            raise ValueError(f"Invalid plan: {errors}")

        return plan

    def validate_plan(self, plan: List[ActionPrimitive]) -> tuple[bool, List[str]]:
        """Validate that the plan is executable"""
        errors = []

        for i, action in enumerate(plan):
            # Check if action is in the library
            if action.name not in self.action_library:
                errors.append(f"Action '{action.name}' not in action library")

            # Check preconditions can be satisfied
            if i > 0:
                previous_effects = []
                for prev_action in plan[:i]:
                    prev_effects = self.action_library[prev_action.name]['effects']
                    previous_effects.extend(prev_effects)

                # Check if preconditions are met by previous actions
                for precondition in action.preconditions:
                    if precondition not in previous_effects:
                        errors.append(f"Precondition '{precondition}' not satisfied for action {i+1}")

        return len(errors) == 0, errors

    def optimize_plan(self, plan: List[ActionPrimitive]) -> List[ActionPrimitive]:
        """Optimize the execution plan for efficiency"""
        # Basic optimization: remove redundant actions
        optimized_plan = []
        executed_effects = set()

        for action in plan:
            # Check if action's effects are already achieved
            if not any(effect in executed_effects for effect in action.effects):
                optimized_plan.append(action)
                executed_effects.update(action.effects)

        return optimized_plan
```

### 2. Constraint-Based Planning

Implement constraint-based planning for more complex scenarios:

```python
class ConstraintSolver:
    def __init__(self):
        self.constraints = []
        self.variables = set()
        self.domains = {}

    def add_constraint(self, constraint_func, variables):
        """Add a constraint to the problem"""
        self.constraints.append({
            'func': constraint_func,
            'variables': variables
        })

    def solve(self, variables_domains):
        """Solve the constraint satisfaction problem"""
        # For simplicity, we'll use a basic backtracking approach
        # In practice, more sophisticated CSP solvers would be used

        assignment = {}
        return self._backtrack(assignment, variables_domains)

    def _backtrack(self, assignment, variables_domains):
        """Backtracking search for constraint satisfaction"""
        # Check if complete assignment
        unassigned = [var for var in variables_domains if var not in assignment]
        if not unassigned:
            return assignment if self._is_consistent(assignment) else None

        # Select unassigned variable
        var = unassigned[0]

        # Try each value in domain
        for value in variables_domains[var]:
            assignment[var] = value

            if self._is_consistent(assignment):
                result = self._backtrack(assignment, variables_domains)
                if result is not None:
                    return result

            del assignment[var]

        return None

    def _is_consistent(self, assignment):
        """Check if assignment satisfies all constraints"""
        for constraint in self.constraints:
            vars_in_assignment = {var: assignment.get(var) for var in constraint['variables']}
            if None not in vars_in_assignment.values():
                if not constraint['func'](vars_in_assignment):
                    return False
        return True
```

## Integration with LLM Planning

### 1. LLM-Driven Action Generation

Connect the LLM cognitive planner with action sequence generation:

```python
import json
import re
from typing import List, Dict, Tuple

class LLMActionGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.semantic_parser = SemanticParser()
        self.task_decomposer = TaskDecomposer()
        self.sequential_planner = SequentialPlanner()

    def generate_action_sequence(self, command: str, context: Dict[str, Any]) -> List[ActionPrimitive]:
        """Generate action sequence from natural language command using LLM"""
        # Create detailed prompt for LLM
        prompt = self._create_generation_prompt(command, context)

        try:
            response = self.llm_client.generate(prompt)
            action_sequence = self._parse_llm_response(response)

            # Validate and optimize the sequence
            is_valid, errors = self.sequential_planner.validate_plan(action_sequence)
            if not is_valid:
                # Try to fix common issues
                action_sequence = self._repair_plan(action_sequence, errors)

            # Optimize the plan
            optimized_sequence = self.sequential_planner.optimize_plan(action_sequence)

            return optimized_sequence

        except Exception as e:
            print(f"Error generating action sequence: {e}")
            return self._fallback_action_sequence(command)

    def _create_generation_prompt(self, command: str, context: Dict[str, Any]) -> str:
        """Create prompt for LLM action sequence generation"""
        prompt = f"""
        You are an action sequence generator for a humanoid robot. Given a natural language command and context, generate a sequence of executable actions.

        CONTEXT:
        - Robot capabilities: {json.dumps(context.get('capabilities', {}), indent=2)}
        - Current environment: {json.dumps(context.get('environment', {}), indent=2)}
        - Available objects: {json.dumps(context.get('objects', []), indent=2)}
        - Known locations: {json.dumps(context.get('locations', []), indent=2)}

        COMMAND: {command}

        Please generate a sequence of actions that the robot can execute to fulfill the command. Each action should be in the format:
        ACTION_NAME(PARAMETER1=value1, PARAMETER2=value2)

        Example actions:
        - navigate_to(location="kitchen")
        - inspect_object(object="red cup")
        - grasp_object(object="red cup", location="table")
        - place_object(object="red cup", destination="shelf")
        - speak(text="I have picked up the red cup")

        RESPONSE FORMAT:
        1. [Action 1]
        2. [Action 2]
        3. [Action 3]
        ...
        """
        return prompt

    def _parse_llm_response(self, response: str) -> List[ActionPrimitive]:
        """Parse LLM response into action primitives"""
        actions = []

        # Extract numbered actions
        lines = response.strip().split('\n')
        for line in lines:
            # Look for patterns like "1. action_name(params)" or "ACTION_NAME(PARAMS)"
            match = re.search(r'\d+\.\s*([A-Z_]+)\((.*)\)', line) or re.search(r'([A-Z_]+)\((.*)\)', line)

            if match:
                action_name = match.group(1).lower()
                params_str = match.group(2)

                # Parse parameters
                params = self._parse_parameters(params_str)

                # Create action primitive
                action = ActionPrimitive(
                    name=action_name,
                    parameters=params,
                    duration=self._estimate_duration(action_name),
                    preconditions=self._get_preconditions(action_name, params),
                    effects=self._get_effects(action_name, params)
                )

                actions.append(action)

        return actions

    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary"""
        params = {}

        # Split by comma but handle nested parentheses/quotes
        param_pairs = re.findall(r'(\w+)=(?:([^,()]+|\([^)]+\)|"[^"]*"|\'[^\']*\'))', params_str)

        for key, value in param_pairs:
            # Clean up the value
            value = value.strip().strip('"\'')

            # Try to convert to appropriate type
            if value.isdigit():
                params[key] = int(value)
            elif value.replace('.', '', 1).isdigit():
                params[key] = float(value)
            elif value.lower() in ['true', 'false']:
                params[key] = value.lower() == 'true'
            else:
                params[key] = value

        return params

    def _estimate_duration(self, action_name: str) -> float:
        """Estimate duration for an action"""
        duration_map = {
            'navigate_to': 10.0,
            'grasp_object': 5.0,
            'place_object': 3.0,
            'inspect_object': 2.0,
            'speak': 1.0,
            'wait': 1.0
        }
        return duration_map.get(action_name, 2.0)

    def _get_preconditions(self, action_name: str, params: Dict[str, Any]) -> List[str]:
        """Get preconditions for an action"""
        preconditions_map = {
            'navigate_to': ['robot_is_operational'],
            'grasp_object': ['object_visible', 'object_reachable', 'gripper_free'],
            'place_object': ['object_grasped', 'destination_reachable'],
            'inspect_object': ['object_visible', 'robot_stationary']
        }
        return preconditions_map.get(action_name, [])

    def _get_effects(self, action_name: str, params: Dict[str, Any]) -> List[str]:
        """Get effects of an action"""
        effects_map = {
            'navigate_to': ['robot_at_destination'],
            'grasp_object': ['object_grasped', 'gripper_occupied'],
            'place_object': ['object_placed', 'gripper_free'],
            'inspect_object': ['object_properties_known'],
            'speak': ['command_acknowledged']
        }
        return effects_map.get(action_name, [])

    def _repair_plan(self, plan: List[ActionPrimitive], errors: List[str]) -> List[ActionPrimitive]:
        """Repair common plan issues"""
        # This is a simplified repair function
        # In practice, this would be more sophisticated
        repaired_plan = []

        for action in plan:
            # Skip actions with known issues
            if not any(action.name in error for error in errors):
                repaired_plan.append(action)

        return repaired_plan

    def _fallback_action_sequence(self, command: str) -> List[ActionPrimitive]:
        """Generate fallback action sequence when LLM fails"""
        # Parse command using semantic parser as fallback
        parsed = self.semantic_parser.parse_command(command)

        if parsed['action_type'] == 'navigate_to':
            return [ActionPrimitive(
                name='navigate_to',
                parameters={'location': parsed['arguments'][0] if parsed['arguments'] else 'unknown'},
                duration=10.0,
                preconditions=['robot_is_operational'],
                effects=['robot_at_location']
            )]
        elif parsed['action_type'] == 'manipulate_object':
            return [
                ActionPrimitive(
                    name='inspect_object',
                    parameters={'object': parsed['arguments'][0] if len(parsed['arguments']) > 0 else 'unknown'},
                    duration=2.0,
                    preconditions=['robot_stationary'],
                    effects=['object_properties_known']
                ),
                ActionPrimitive(
                    name='grasp_object',
                    parameters={'object': parsed['arguments'][0] if len(parsed['arguments']) > 0 else 'unknown'},
                    duration=5.0,
                    preconditions=['object_visible', 'object_reachable', 'gripper_free'],
                    effects=['object_grasped', 'gripper_occupied']
                )
            ]

        # Default: return empty plan
        return []
```

## ROS 2 Integration

### 1. Action Execution Node

Create a ROS 2 node to execute the generated action sequences:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from action_msgs.msg import GoalStatus
from move_base_msgs.action import MoveBase
from std_srvs.srv import Trigger
import time
import threading
from queue import Queue

class ActionExecutionNode(Node):
    def __init__(self):
        super().__init__('action_execution_node')

        # Initialize action sequence executor
        self.action_queue = Queue()
        self.is_executing = False
        self.current_action = None
        self.execution_lock = threading.Lock()

        # Publishers and subscribers
        self.status_pub = self.create_publisher(String, '/action_status', 10)
        self.feedback_pub = self.create_publisher(String, '/action_feedback', 10)
        self.result_pub = self.create_publisher(String, '/action_result', 10)

        # Subscribers for command input
        self.command_sub = self.create_subscription(
            String,
            '/action_sequence',
            self.action_sequence_callback,
            10
        )

        # Action clients for navigation
        self.nav_client = ActionClient(self, MoveBase, 'move_base')

        # Service clients for manipulation
        self.grasp_service = self.create_client(Trigger, '/grasp_object')
        self.place_service = self.create_client(Trigger, '/place_object')

        # Start execution thread
        self.execution_thread = threading.Thread(target=self.execution_worker, daemon=True)
        self.execution_thread.start()

        self.get_logger().info('Action execution node initialized')

    def action_sequence_callback(self, msg):
        """Receive action sequence from planning system"""
        try:
            # Parse the action sequence from the message
            action_sequence = self.parse_action_sequence(msg.data)

            # Add actions to execution queue
            for action in action_sequence:
                self.action_queue.put(action)

            # Start execution if not already running
            if not self.is_executing:
                self.start_execution()

        except Exception as e:
            self.get_logger().error(f'Error processing action sequence: {e}')

    def parse_action_sequence(self, action_sequence_str: str) -> List[ActionPrimitive]:
        """Parse action sequence string into action primitives"""
        # This would parse the action sequence from the LLM or planner
        # For now, we'll implement a simple parsing
        actions = []

        # Split by lines and parse each action
        lines = action_sequence_str.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- ') or line.startswith('* '):
                action_str = line[2:]  # Remove bullet point
                if '(' in action_str and ')' in action_str:
                    action_name = action_str.split('(')[0].strip()
                    params_str = action_str.split('(')[1].rstrip(')').strip()

                    # Parse parameters
                    params = {}
                    if params_str:
                        for param_pair in params_str.split(','):
                            if '=' in param_pair:
                                key, value = param_pair.split('=', 1)
                                params[key.strip()] = value.strip().strip('"\'')

                    action = ActionPrimitive(
                        name=action_name,
                        parameters=params,
                        duration=2.0,  # Default duration
                        preconditions=[],
                        effects=[]
                    )
                    actions.append(action)

        return actions

    def execution_worker(self):
        """Worker thread for executing actions"""
        while rclpy.ok():
            if not self.action_queue.empty() and not self.is_executing:
                with self.execution_lock:
                    if not self.action_queue.empty():
                        action = self.action_queue.get()
                        self.execute_action(action)

            time.sleep(0.1)  # Brief sleep to avoid busy waiting

    def execute_action(self, action: ActionPrimitive):
        """Execute a single action primitive"""
        self.is_executing = True
        self.current_action = action

        # Update status
        status_msg = String()
        status_msg.data = f"Executing: {action.name} with params {action.parameters}"
        self.status_pub.publish(status_msg)

        success = False
        try:
            if action.name == 'navigate_to':
                success = self.execute_navigation_action(action)
            elif action.name == 'grasp_object':
                success = self.execute_grasp_action(action)
            elif action.name == 'place_object':
                success = self.execute_place_action(action)
            elif action.name == 'inspect_object':
                success = self.execute_inspection_action(action)
            elif action.name == 'speak':
                success = self.execute_speak_action(action)
            else:
                self.get_logger().warning(f"Unknown action type: {action.name}")
                success = False

        except Exception as e:
            self.get_logger().error(f'Error executing action {action.name}: {e}')
            success = False

        # Update status based on result
        result_msg = String()
        if success:
            result_msg.data = f"SUCCESS: {action.name}"
        else:
            result_msg.data = f"FAILED: {action.name}"

        self.result_pub.publish(result_msg)

        # Reset execution state
        self.is_executing = False
        self.current_action = None

    def execute_navigation_action(self, action: ActionPrimitive) -> bool:
        """Execute navigation action"""
        try:
            # Wait for navigation action server
            if not self.nav_client.wait_for_server(timeout_sec=5.0):
                self.get_logger().error('Navigation server not available')
                return False

            # Create navigation goal
            goal_msg = MoveBase.Goal()
            goal_msg.target_pose.header.frame_id = 'map'
            goal_msg.target_pose.header.stamp = self.get_clock().now().to_msg()

            # Parse location from parameters and convert to coordinates
            location = action.parameters.get('location', 'unknown')

            # In a real system, you would have a location map
            # For this example, we'll use a simple mapping
            location_coords = self.get_location_coordinates(location)

            if location_coords:
                goal_msg.target_pose.pose.position.x = location_coords[0]
                goal_msg.target_pose.pose.position.y = location_coords[1]
                goal_msg.target_pose.pose.position.z = 0.0
                # Set orientation to face forward
                goal_msg.target_pose.pose.orientation.w = 1.0

                # Send navigation goal
                future = self.nav_client.send_goal_async(goal_msg)
                rclpy.spin_until_future_complete(self, future)

                # Wait for result
                result = future.result()
                return result.status == GoalStatus.STATUS_SUCCEEDED
            else:
                self.get_logger().error(f'Unknown location: {location}')
                return False

        except Exception as e:
            self.get_logger().error(f'Navigation execution error: {e}')
            return False

    def execute_grasp_action(self, action: ActionPrimitive) -> bool:
        """Execute grasping action"""
        try:
            # Wait for grasp service
            if not self.grasp_service.wait_for_service(timeout_sec=2.0):
                self.get_logger().error('Grasp service not available')
                return False

            # Call grasp service
            request = Trigger.Request()
            future = self.grasp_service.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()
            return response.success

        except Exception as e:
            self.get_logger().error(f'Grasp execution error: {e}')
            return False

    def execute_place_action(self, action: ActionPrimitive) -> bool:
        """Execute placing action"""
        try:
            # Wait for place service
            if not self.place_service.wait_for_service(timeout_sec=2.0):
                self.get_logger().error('Place service not available')
                return False

            # Call place service
            request = Trigger.Request()
            future = self.place_service.call_async(request)
            rclpy.spin_until_future_complete(self, future)

            response = future.result()
            return response.success

        except Exception as e:
            self.get_logger().error(f'Place execution error: {e}')
            return False

    def get_location_coordinates(self, location_name: str) -> Optional[List[float]]:
        """Get coordinates for a named location"""
        # This would typically come from a map or localization system
        # For this example, we'll use a simple lookup table
        location_map = {
            'kitchen': [5.0, 0.0],
            'living_room': [0.0, 5.0],
            'bedroom': [-5.0, 0.0],
            'office': [0.0, -5.0],
            'hallway': [0.0, 0.0],
            'dining_room': [3.0, 3.0]
        }
        return location_map.get(location_name.lower())

    def start_execution(self):
        """Start executing action sequence"""
        self.get_logger().info('Starting action sequence execution')

    def stop_execution(self):
        """Stop current execution"""
        self.get_logger().info('Stopping action sequence execution')
        with self.execution_lock:
            self.is_executing = False
            # Clear queue
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except:
                    break