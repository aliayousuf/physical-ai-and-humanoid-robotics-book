---
title: "AI Planning Workflow"
description: "Creating sophisticated planning systems that integrate vision, language, and action capabilities"
---

# AI Planning Workflow

## Overview

AI planning in robotics involves creating sophisticated systems that can interpret high-level goals, decompose them into executable actions, and adapt to changing conditions. This workflow covers the complete process of building AI planning systems that integrate vision, language, and action capabilities to enable autonomous robot behavior.

## Planning System Architecture

### 1. Hierarchical Planning Structure

The AI planning system follows a hierarchical approach:

```
Task Level: "Clean the room"
    ↓
Goal Level: "Pick up all objects on floor"
    ↓
Action Level: "Navigate to object location → Grasp object → Place in bin"
    ↓
Motion Level: "Path planning → Arm trajectory → Gripper control"
    ↓
Control Level: "Joint position commands"
```

### 2. Planning Components

```python
class AIPlanningSystem:
    def __init__(self):
        # High-level task planner
        self.task_planner = TaskPlanner()

        # Mid-level goal decomposer
        self.goal_decomposer = GoalDecomposer()

        # Low-level action sequencer
        self.action_sequencer = ActionSequencer()

        # Motion planner
        self.motion_planner = MotionPlanner()

        # Execution monitor
        self.execution_monitor = ExecutionMonitor()

        # Recovery system
        self.recovery_system = RecoverySystem()

    def plan_and_execute(self, high_level_goal, environment_state):
        """Execute complete planning and execution pipeline"""
        # Step 1: Task planning
        task_plan = self.task_planner.create_plan(high_level_goal, environment_state)

        # Step 2: Goal decomposition
        goal_sequence = self.goal_decomposer.decompose_tasks(task_plan)

        # Step 3: Action sequencing
        action_sequence = self.action_sequencer.sequence_actions(goal_sequence)

        # Step 4: Motion planning for each action
        motion_sequences = []
        for action in action_sequence:
            motion_seq = self.motion_planner.plan_motion_for_action(action, environment_state)
            motion_sequences.append(motion_seq)

        # Step 5: Execute with monitoring
        execution_result = self.execute_with_monitoring(motion_sequences)

        return execution_result
```

### 3. Vision-Guided Planning

Integrating visual information into planning decisions:

```python
class VisionGuidedPlanner:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.spatial_reasoner = SpatialReasoner()

    def plan_with_vision(self, goal, current_image, environment_state):
        """Plan with visual input guiding the process"""
        # Analyze current visual scene
        scene_description = self.analyze_scene(current_image)

        # Update environment state with visual information
        updated_state = self.update_environment_state(
            environment_state, scene_description
        )

        # Plan based on updated state
        plan = self.create_plan_with_visual_context(goal, updated_state)

        # Validate plan against visual constraints
        validated_plan = self.validate_plan_against_scene(plan, scene_description)

        return validated_plan

    def analyze_scene(self, image):
        """Analyze scene for planning-relevant information"""
        # Detect objects and their properties
        objects = self.object_detector.detect(image)

        # Analyze spatial relationships
        spatial_relations = self.spatial_reasoner.analyze_relationships(objects)

        # Identify navigable areas
        navigable_areas = self.spatial_reasoner.identify_navigable_spaces(image)

        # Detect obstacles and hazards
        obstacles = self.spatial_reasoner.detect_obstacles(image)

        return {
            'objects': objects,
            'spatial_relations': spatial_relations,
            'navigable_areas': navigable_areas,
            'obstacles': obstacles
        }

    def create_plan_with_visual_context(self, goal, environment_state):
        """Create plan considering visual information"""
        # Parse the goal to understand what needs to be done
        goal_requirements = self.parse_goal_requirements(goal)

        # Find relevant objects in environment
        relevant_objects = self.find_relevant_objects(
            goal_requirements, environment_state
        )

        # Create plan considering object locations and accessibility
        plan = self.generate_plan_for_objects(goal, relevant_objects, environment_state)

        return plan

    def validate_plan_against_scene(self, plan, scene_description):
        """Validate plan against current scene constraints"""
        validated_plan = []

        for action in plan:
            # Check if action is feasible given current scene
            if self.is_action_feasible(action, scene_description):
                validated_plan.append(action)
            else:
                # Try to modify action to fit scene constraints
                modified_action = self.modify_action_for_scene(action, scene_description)
                if modified_action:
                    validated_plan.append(modified_action)
                else:
                    # Action is not feasible, need alternative approach
                    alternative_action = self.find_alternative_action(
                        action, scene_description
                    )
                    if alternative_action:
                        validated_plan.append(alternative_action)
                    else:
                        raise PlanningException(f"No feasible way to execute action: {action}")

        return validated_plan
```

## Language-Guided Planning

### 1. Natural Language Understanding for Planning

Processing natural language commands into executable plans:

```python
class LanguageGuidedPlanner:
    def __init__(self):
        self.parser = NaturalLanguageParser()
        self.semantic_mapper = SemanticMapper()
        self.primitive_mapper = PrimitiveActionMapper()

    def plan_from_language(self, command, environment_state):
        """Create plan from natural language command"""
        # Parse the command into semantic structure
        semantic_structure = self.parser.parse_command(command)

        # Map semantics to planning concepts
        planning_concepts = self.semantic_mapper.map_to_concepts(semantic_structure)

        # Convert to primitive actions
        primitive_actions = self.primitive_mapper.map_to_primitives(
            planning_concepts, environment_state
        )

        # Validate and optimize action sequence
        validated_sequence = self.validate_action_sequence(
            primitive_actions, environment_state
        )

        return validated_sequence

    def parse_command(self, command):
        """Parse natural language command into structured representation"""
        # Use dependency parsing to understand command structure
        doc = self.nlp(command)

        # Extract main action
        main_verb = self.extract_main_verb(doc)

        # Extract objects and locations
        objects = self.extract_objects(doc)
        locations = self.extract_locations(doc)

        # Extract modifiers and constraints
        modifiers = self.extract_modifiers(doc)

        return {
            'main_action': main_verb,
            'objects': objects,
            'locations': locations,
            'modifiers': modifiers,
            'raw_command': command
        }

    def map_to_primitives(self, planning_concepts, environment_state):
        """Map planning concepts to primitive robot actions"""
        primitive_actions = []

        for concept in planning_concepts:
            if concept['type'] == 'navigation':
                primitive = self.create_navigation_primitive(concept, environment_state)
            elif concept['type'] == 'manipulation':
                primitive = self.create_manipulation_primitive(concept, environment_state)
            elif concept['type'] == 'inspection':
                primitive = self.create_inspection_primitive(concept, environment_state)
            elif concept['type'] == 'communication':
                primitive = self.create_communication_primitive(concept, environment_state)
            else:
                # Unknown concept type, skip or raise exception
                continue

            primitive_actions.append(primitive)

        return primitive_actions

    def create_navigation_primitive(self, concept, environment_state):
        """Create navigation primitive from concept"""
        destination = concept.get('destination', 'unknown')

        # Find actual location in environment
        actual_destination = self.resolve_location(destination, environment_state)

        return {
            'type': 'navigation',
            'action': 'navigate_to',
            'parameters': {
                'destination': actual_destination,
                'path_constraints': concept.get('constraints', [])
            }
        }

    def create_manipulation_primitive(self, concept, environment_state):
        """Create manipulation primitive from concept"""
        object_name = concept.get('object', 'unknown')
        action_type = concept.get('manipulation_action', 'grasp')

        # Find actual object in environment
        actual_object = self.resolve_object(object_name, environment_state)

        if action_type == 'grasp':
            return {
                'type': 'manipulation',
                'action': 'grasp_object',
                'parameters': {
                    'object': actual_object,
                    'approach_strategy': concept.get('approach', 'default')
                }
            }
        elif action_type == 'place':
            destination = concept.get('destination', 'default')
            return {
                'type': 'manipulation',
                'action': 'place_object',
                'parameters': {
                    'object': actual_object,
                    'destination': destination,
                    'placement_strategy': concept.get('placement', 'default')
                }
            }

        # Add more manipulation types as needed
        return None
```

## Multi-Modal Planning Integration

### 1. Fusing Vision and Language for Planning

Creating plans that leverage both visual and linguistic information:

```python
class MultiModalPlanner:
    def __init__(self):
        self.vision_planner = VisionGuidedPlanner()
        self.language_planner = LanguageGuidedPlanner()
        self.fusion_engine = FusionEngine()
        self.conflict_resolver = ConflictResolver()

    def plan_multimodal(self, command, current_image, environment_state):
        """Create plan using both vision and language information"""
        # Plan separately with each modality
        vision_plan = self.vision_planner.plan_with_vision(
            command, current_image, environment_state
        )

        language_plan = self.language_planner.plan_from_language(
            command, environment_state
        )

        # Fuse the plans considering confidence in each modality
        fused_plan = self.fusion_engine.fuse_plans(
            vision_plan, language_plan, self.assess_confidence(command, current_image)
        )

        # Resolve any conflicts in the fused plan
        resolved_plan = self.conflict_resolver.resolve_conflicts(fused_plan)

        # Validate the final plan
        validated_plan = self.validate_multimodal_plan(resolved_plan, environment_state)

        return validated_plan

    def assess_confidence(self, command, image):
        """Assess confidence in each modality for the given inputs"""
        # Analyze command complexity for language processing
        language_confidence = self.assess_language_confidence(command)

        # Analyze image quality for vision processing
        vision_confidence = self.assess_vision_confidence(image)

        return {
            'language_confidence': language_confidence,
            'vision_confidence': vision_confidence
        }

    def assess_language_confidence(self, command):
        """Assess confidence in language understanding"""
        # Simple assessment based on command structure
        # In practice, this would use more sophisticated metrics

        # Longer, more structured commands tend to be clearer
        if len(command.split()) >= 3:
            return 0.8
        else:
            return 0.5

    def assess_vision_confidence(self, image):
        """Assess confidence in visual understanding"""
        # Analyze image quality, object visibility, etc.
        # For now, return a default value
        return 0.9  # Assume high confidence in vision

    def validate_multimodal_plan(self, plan, environment_state):
        """Validate plan considering both modalities"""
        validated_plan = []

        for action in plan:
            # Validate action feasibility
            if self.is_action_feasible(action, environment_state):
                # Check if action aligns with command intent
                if self.action_aligns_with_command(action, plan['original_command']):
                    validated_plan.append(action)
                else:
                    # Modify action to better align with command
                    aligned_action = self.align_action_with_command(action, plan['original_command'])
                    if aligned_action:
                        validated_plan.append(aligned_action)
            else:
                # Action not feasible, find alternative
                alternative = self.find_alternative_action(action, environment_state)
                if alternative:
                    validated_plan.append(alternative)

        return validated_plan
```

### 2. Planning with Uncertainty

Handling uncertainty in both vision and language inputs:

```python
class UncertaintyAwarePlanner:
    def __init__(self):
        self.belief_propagator = BeliefPropagator()
        self.probabilistic_planner = ProbabilisticPlanner()

    def plan_with_uncertainty(self, command, image, environment_state):
        """Plan considering uncertainty in inputs"""
        # Create belief state from uncertain inputs
        belief_state = self.create_belief_state(command, image, environment_state)

        # Plan considering uncertainty distributions
        plan = self.probabilistic_planner.plan_with_beliefs(belief_state)

        # Include contingency plans for uncertain situations
        contingency_plans = self.generate_contingency_plans(belief_state)

        return {
            'primary_plan': plan,
            'contingency_plans': contingency_plans,
            'belief_state': belief_state
        }

    def create_belief_state(self, command, image, environment_state):
        """Create belief state from uncertain inputs"""
        belief_state = {
            'objects': self.estimate_object_probabilities(image),
            'locations': self.estimate_location_probabilities(command, image),
            'actions': self.estimate_action_probabilities(command),
            'environment': environment_state
        }

        return belief_state

    def estimate_object_probabilities(self, image):
        """Estimate probabilities for different object hypotheses"""
        # Use object detection with confidence scores
        detections = self.object_detector.detect_with_confidence(image)

        objects_with_probabilities = []
        for detection in detections:
            objects_with_probabilities.append({
                'name': detection['class'],
                'probability': detection['confidence'],
                'location': detection['bbox'],
                'properties': detection.get('properties', {})
            })

        return objects_with_probabilities

    def estimate_location_probabilities(self, command, image):
        """Estimate probabilities for different location hypotheses"""
        # Use both command and visual information to estimate locations
        command_locations = self.extract_location_hypotheses_from_command(command)
        visual_locations = self.extract_location_hypotheses_from_image(image)

        # Combine information from both modalities
        combined_locations = self.combine_location_hypotheses(
            command_locations, visual_locations
        )

        return combined_locations

    def plan_with_beliefs(self, belief_state):
        """Plan considering belief distributions"""
        # Use probabilistic planning techniques
        # This would typically involve POMDP or similar approaches

        # For each possible world state, create a plan
        possible_worlds = self.generate_possible_worlds(belief_state)

        plans_for_worlds = []
        for world in possible_worlds:
            plan = self.create_plan_for_world(world)
            plans_for_worlds.append(plan)

        # Select best plan based on expected utility
        best_plan = self.select_best_plan(plans_for_worlds, belief_state)

        return best_plan
```

## Execution Monitoring and Adaptation

### 1. Plan Execution with Monitoring

Monitor plan execution and adapt when necessary:

```python
class PlanExecutionMonitor:
    def __init__(self):
        self.executor = ActionExecutor()
        self.monitor = ExecutionMonitor()
        self.adaptor = PlanAdaptor()

    def execute_with_monitoring(self, plan, environment_state):
        """Execute plan with continuous monitoring and adaptation"""
        execution_context = {
            'plan': plan,
            'current_step': 0,
            'execution_history': [],
            'environment_state': environment_state,
            'failure_count': 0
        }

        while execution_context['current_step'] < len(plan):
            current_action = plan[execution_context['current_step']]

            # Execute action
            result = self.executor.execute_action(current_action, environment_state)

            # Monitor execution
            monitoring_result = self.monitor.evaluate_execution(
                current_action, result, environment_state
            )

            # Update execution context
            execution_context['execution_history'].append({
                'action': current_action,
                'result': result,
                'monitoring': monitoring_result
            })

            if monitoring_result['success']:
                # Action succeeded, move to next
                execution_context['current_step'] += 1
            else:
                # Action failed, handle failure
                execution_context['failure_count'] += 1

                # Check if we should abandon or adapt
                if execution_context['failure_count'] > MAX_FAILURES:
                    return self.handle_abandonment(plan, execution_context)

                # Try to adapt the plan
                adapted_plan = self.adaptor.adapt_plan(
                    plan, execution_context, monitoring_result['failure_reason']
                )

                if adapted_plan:
                    plan = adapted_plan
                    # Reset execution context for new plan
                    execution_context['current_step'] = 0
                    execution_context['execution_history'] = []
                else:
                    # No adaptation possible, abandon
                    return self.handle_abandonment(plan, execution_context)

        return {'status': 'success', 'execution_history': execution_context['execution_history']}

    def handle_abandonment(self, original_plan, execution_context):
        """Handle plan abandonment and provide alternatives"""
        failure_analysis = self.analyze_failures(execution_context['execution_history'])

        alternative_goals = self.generate_alternative_goals(
            original_plan, failure_analysis, execution_context['environment_state']
        )

        return {
            'status': 'failed',
            'original_plan': original_plan,
            'execution_history': execution_context['execution_history'],
            'failure_analysis': failure_analysis,
            'alternatives': alternative_goals
        }

    def analyze_failures(self, execution_history):
        """Analyze execution failures to understand causes"""
        failure_patterns = []

        for execution in execution_history:
            if not execution['monitoring']['success']:
                failure_patterns.append({
                    'action': execution['action'],
                    'failure_type': execution['monitoring']['failure_type'],
                    'failure_reason': execution['monitoring']['failure_reason'],
                    'environment_state': execution['monitoring']['environment_state']
                })

        return failure_patterns
```

### 2. Plan Adaptation Strategies

Implement strategies for adapting plans when execution encounters problems:

```python
class PlanAdaptor:
    def __init__(self):
        self.knowledge_base = PlanKnowledgeBase()
        self.recovery_strategies = RecoveryStrategies()

    def adapt_plan(self, original_plan, execution_context, failure_reason):
        """Adapt plan based on execution failure"""
        # Identify the type of failure
        failure_type = self.classify_failure(failure_reason)

        if failure_type == 'object_not_found':
            return self.adapt_for_missing_object(original_plan, execution_context)
        elif failure_type == 'path_blocked':
            return self.adapt_for_blocked_path(original_plan, execution_context)
        elif failure_type == 'grasp_failed':
            return self.adapt_for_grasp_failure(original_plan, execution_context)
        elif failure_type == 'environment_changed':
            return self.adapt_for_environment_change(original_plan, execution_context)
        else:
            # Unknown failure type, try general adaptation
            return self.general_adaptation(original_plan, execution_context)

    def adapt_for_missing_object(self, plan, execution_context):
        """Adapt plan when expected object is not found"""
        # Look for alternative objects that could satisfy the goal
        current_action = plan[execution_context['current_step']]
        expected_object = current_action['parameters'].get('object', 'unknown')

        # Search for similar objects in environment
        alternative_objects = self.find_alternative_objects(
            expected_object, execution_context['environment_state']
        )

        if alternative_objects:
            # Create new plan with alternative objects
            new_plan = self.modify_plan_with_alternatives(
                plan, execution_context['current_step'], alternative_objects
            )
            return new_plan

        # No alternatives found, try to skip or modify goal
        return self.modify_goal_for_missing_object(plan, execution_context)

    def adapt_for_blocked_path(self, plan, execution_context):
        """Adapt plan when navigation path is blocked"""
        # Recalculate path considering current obstacles
        current_action = plan[execution_context['current_step']]

        if current_action['action'] == 'navigate_to':
            new_destination = current_action['parameters']['destination']

            # Get updated environment state with new obstacles
            updated_env = self.update_environment_with_new_obstacles(
                execution_context['environment_state'],
                execution_context['execution_history']
            )

            # Find alternative path
            alternative_path = self.find_alternative_path(
                updated_env, new_destination
            )

            if alternative_path:
                # Update navigation action with new path
                modified_action = current_action.copy()
                modified_action['parameters']['alternative_path'] = alternative_path

                new_plan = plan.copy()
                new_plan[execution_context['current_step']] = modified_action

                return new_plan

        return None  # No adaptation possible for non-navigation blocking

    def adapt_for_environment_change(self, plan, execution_context):
        """Adapt plan when environment has changed significantly"""
        # Re-assess the environment state
        new_environment_state = self.reassess_environment(
            execution_context['environment_state']
        )

        # Check if original goal is still achievable
        if self.is_goal_achievable(plan[0], new_environment_state):
            # Goal still achievable, just replan from current state
            remaining_actions = plan[execution_context['current_step']:]
            return self.replan_remaining_actions(
                remaining_actions, new_environment_state
            )
        else:
            # Goal no longer achievable, need to modify
            return self.modify_goal_for_environment_change(
                plan, execution_context, new_environment_state
            )

    def classify_failure(self, failure_reason):
        """Classify failure reason into adaptation strategy"""
        failure_lower = failure_reason.lower()

        if 'object' in failure_lower and ('not found' in failure_lower or 'missing' in failure_lower):
            return 'object_not_found'
        elif 'path' in failure_lower and ('blocked' in failure_lower or 'obstacle' in failure_lower):
            return 'path_blocked'
        elif 'grasp' in failure_lower or 'manipulation' in failure_lower:
            return 'grasp_failed'
        elif 'environment' in failure_lower or 'changed' in failure_lower:
            return 'environment_changed'
        else:
            return 'unknown'
```

## Integration with ROS 2

### 1. Planning Service Node

Create a ROS 2 service node for AI planning:

```python
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.qos import QoSProfile
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import threading
import json

class AIPlanningNode(Node):
    def __init__(self):
        super().__init__('ai_planning_node')

        # Initialize planning system components
        self.planning_system = AIPlanningSystem()
        self.vision_planner = VisionGuidedPlanner()
        self.language_planner = LanguageGuidedPlanner()
        self.multimodal_planner = MultiModalPlanner()

        # Publishers for plan status and visualization
        self.plan_status_pub = self.create_publisher(
            String, '/plan_status', 10
        )

        self.plan_visualization_pub = self.create_publisher(
            String, '/plan_visualization', 10
        )

        # Subscribers for environment state
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10
        )

        # Service for planning requests
        self.plan_service = self.create_service(
            PlanRequest, '/plan_request', self.plan_request_callback
        )

        # Store latest environment information
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.environment_state = self.initialize_environment_state()

        self.get_logger().info('AI Planning node initialized')

    def image_callback(self, msg):
        """Update latest image for planning"""
        with self.image_lock:
            self.latest_image = msg

    def command_callback(self, msg):
        """Process natural language command and generate plan"""
        try:
            # Get current environment state
            with self.image_lock:
                current_image = self.latest_image

            if current_image is not None:
                # Plan using multimodal approach
                plan = self.multimodal_planner.plan_multimodal(
                    msg.data, current_image, self.environment_state
                )
            else:
                # Plan using language only
                plan = self.language_planner.plan_from_language(
                    msg.data, self.environment_state
                )

            # Publish plan for visualization
            plan_msg = String()
            plan_msg.data = json.dumps({
                'command': msg.data,
                'plan': plan,
                'timestamp': self.get_clock().now().to_msg()
            })
            self.plan_visualization_pub.publish(plan_msg)

            # Execute the plan
            execution_result = self.execute_plan(plan)

            # Publish status
            status_msg = String()
            status_msg.data = f"Command '{msg.data}' execution: {execution_result['status']}"
            self.plan_status_pub.publish(status_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')

    def plan_request_callback(self, request, response):
        """Handle planning service requests"""
        try:
            # Get current environment state
            with self.image_lock:
                current_image = self.latest_image

            # Plan based on request type
            if request.type == 'multimodal':
                plan = self.multimodal_planner.plan_multimodal(
                    request.command, current_image, self.environment_state
                )
            elif request.type == 'language_only':
                plan = self.language_planner.plan_from_language(
                    request.command, self.environment_state
                )
            elif request.type == 'vision_guided':
                plan = self.vision_planner.plan_with_vision(
                    request.command, current_image, self.environment_state
                )
            else:
                response.success = False
                response.message = f"Unknown plan type: {request.type}"
                return response

            # Serialize plan
            response.plan = json.dumps(plan)
            response.success = True
            response.message = "Plan generated successfully"

        except Exception as e:
            self.get_logger().error(f'Error in plan request: {e}')
            response.success = False
            response.message = f"Planning failed: {str(e)}"

        return response

    def execute_plan(self, plan):
        """Execute the generated plan with monitoring"""
        execution_monitor = PlanExecutionMonitor()
        result = execution_monitor.execute_with_monitoring(
            plan, self.environment_state
        )
        return result

    def initialize_environment_state(self):
        """Initialize the environment state"""
        return {
            'robot_position': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'theta': 0.0},
            'robot_state': 'idle',
            'known_objects': {},
            'navigable_areas': [],
            'obstacles': [],
            'timestamp': self.get_clock().now().to_msg()
        }
```

## Performance Considerations

### 1. Planning Efficiency

Optimize planning performance for real-time applications:

```python
class EfficientPlanner:
    def __init__(self):
        self.cache = PlanningCache()
        self.early_terminator = EarlyTerminationCriteria()
        self.parallel_planner = ParallelPlanning()

    def plan_efficiently(self, goal, environment_state, timeout=5.0):
        """Plan efficiently with performance optimizations"""
        # Check cache first
        cached_plan = self.cache.get_cached_plan(goal, environment_state)
        if cached_plan:
            return cached_plan

        # Plan with timeout and early termination
        start_time = time.time()

        # Use hierarchical planning to quickly find approximate solutions
        coarse_plan = self.plan_coarse(goal, environment_state)

        if time.time() - start_time > timeout * 0.3:  # Use 30% of timeout for coarse planning
            # Return coarse plan if time is running out
            self.cache.store_plan(goal, environment_state, coarse_plan)
            return coarse_plan

        # Refine plan if sufficient time remains
        refined_plan = self.refine_plan(coarse_plan, environment_state)

        if time.time() - start_time > timeout * 0.8:  # Use 80% of timeout for refinement
            # Return partially refined plan if time is running out
            self.cache.store_plan(goal, environment_state, refined_plan)
            return refined_plan

        # Optimize plan if time permits
        optimized_plan = self.optimize_plan(refined_plan, environment_state)

        # Store in cache for future use
        self.cache.store_plan(goal, environment_state, optimized_plan)

        return optimized_plan

    def plan_coarse(self, goal, environment_state):
        """Create coarse plan quickly"""
        # Use simplified models and heuristics for fast planning
        # This would implement faster planning algorithms like A* with relaxed constraints
        pass

    def refine_plan(self, coarse_plan, environment_state):
        """Refine coarse plan with detailed considerations"""
        # Add detailed constraints and validations to coarse plan
        # This would implement more precise planning algorithms
        pass

    def optimize_plan(self, refined_plan, environment_state):
        """Optimize plan for efficiency and safety"""
        # Optimize action sequences for better performance
        # This might include parallelization, smoothing, etc.
        pass

class PlanningCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}

    def get_cached_plan(self, goal, environment_state):
        """Get plan from cache if available"""
        cache_key = self.generate_cache_key(goal, environment_state)

        if cache_key in self.cache:
            self.access_times[cache_key] = time.time()
            return self.cache[cache_key]

        return None

    def store_plan(self, goal, environment_state, plan):
        """Store plan in cache"""
        cache_key = self.generate_cache_key(goal, environment_state)

        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]

        self.cache[cache_key] = plan
        self.access_times[cache_key] = time.time()

    def generate_cache_key(self, goal, environment_state):
        """Generate cache key from goal and environment state"""
        # Create a hashable key from goal and relevant environment features
        # This should be tolerant of minor environmental changes
        env_features = self.extract_relevant_features(environment_state)
        return f"{hash(goal)}_{hash(str(env_features))}"

    def extract_relevant_features(self, environment_state):
        """Extract features that affect planning significantly"""
        # Extract only features that significantly impact planning
        # Ignore minor changes that shouldn't invalidate cached plans
        return {
            'major_obstacles': environment_state.get('major_obstacles', []),
            'navigable_areas': environment_state.get('navigable_areas', []),
            'robot_state': environment_state.get('robot_state', 'idle')
        }
```

## Learning Objectives

After completing this workflow, you should understand:
- How to create hierarchical planning systems for robotics applications
- Techniques for integrating vision and language information in planning
- Methods for handling uncertainty in planning systems
- Strategies for plan execution monitoring and adaptation
- Implementation of ROS 2 services for AI planning
- Performance optimization techniques for real-time planning

## Next Steps

Continue to learn about [Capstone Implementation](./capstone-implementation) to understand how to build complete integrated systems that combine all the VLA capabilities into a functional autonomous humanoid agent.