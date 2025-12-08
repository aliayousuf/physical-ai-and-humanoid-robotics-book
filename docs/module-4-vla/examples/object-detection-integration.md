---
title: "Object Detection Integration Example"
description: "Complete example demonstrating object detection integration with VLA systems"
---

# Object Detection Integration Example

## Overview

This example demonstrates how to integrate object detection capabilities into Vision-Language-Action (VLA) systems. We'll build a complete pipeline that can detect objects in real-time camera feeds, associate them with voice commands, and execute appropriate actions based on the detected objects.

## Complete Implementation

### 1. Object Detection System

```python
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import time

@dataclass
class DetectedObject:
    """Represents a detected object with its properties"""
    name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]          # (x, y) center coordinates
    area: float

class ObjectDetector:
    """Object detection system using PyTorch pretrained models"""

    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # COCO dataset class names
        self.coco_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
            'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
            'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
            'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Transformation for input images
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects in an image"""
        # Convert image to tensor
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process predictions
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        detected_objects = []
        for i in range(len(boxes)):
            if scores[i] > self.confidence_threshold:
                box = boxes[i]
                label = self.coco_names[labels[i]]
                score = scores[i]

                # Calculate center and area
                x1, y1, x2, y2 = box
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                area = (x2 - x1) * (y2 - y1)

                detected_obj = DetectedObject(
                    name=label,
                    confidence=score,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    center=(center_x, center_y),
                    area=area
                )
                detected_objects.append(detected_obj)

        return detected_objects

    def draw_detections(self, image: np.ndarray, objects: List[DetectedObject]) -> np.ndarray:
        """Draw detection boxes on image"""
        output_image = image.copy()

        for obj in objects:
            x1, y1, x2, y2 = obj.bbox
            center_x, center_y = obj.center

            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label and confidence
            label = f"{obj.name}: {obj.confidence:.2f}"
            cv2.putText(output_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw center point
            cv2.circle(output_image, (center_x, center_y), 3, (255, 0, 0), -1)

        return output_image
```

### 2. Object-Command Association System

```python
from typing import Union

class ObjectCommandAssociator:
    """Associates detected objects with voice commands"""

    def __init__(self):
        # Define object categories for different commands
        self.graspable_objects = [
            'bottle', 'cup', 'book', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'hot dog', 'pizza', 'donut', 'cake',
            'backpack', 'handbag', 'frisbee', 'sports ball', 'bicycle', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'wine glass', 'cell phone', 'remote', 'teddy bear'
        ]

        self.navigable_objects = [
            'person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe'
        ]

        self.interactable_objects = [
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'keyboard', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'clock', 'vase', 'scissors'
        ]

    def find_relevant_objects(self, command_entities: Dict[str, str],
                            detected_objects: List[DetectedObject]) -> List[DetectedObject]:
        """Find objects relevant to the command"""
        relevant_objects = []

        # Look for specific object mentioned in command
        if 'object' in command_entities:
            target_object = command_entities['object'].lower()

            for obj in detected_objects:
                if target_object in obj.name.lower() or obj.name.lower() in target_object:
                    relevant_objects.append(obj)

        # If no specific object, return all graspable objects
        if not relevant_objects:
            for obj in detected_objects:
                if obj.name in self.graspable_objects:
                    relevant_objects.append(obj)

        return relevant_objects

    def find_closest_object(self, target_object: str,
                          detected_objects: List[DetectedObject],
                          image_center: Tuple[int, int] = None) -> Optional[DetectedObject]:
        """Find the closest instance of a target object to the image center or robot"""
        if image_center is None:
            # Use center of image as reference
            image_center = (320, 240)  # Assuming 640x480 image

        target_objects = [obj for obj in detected_objects
                         if target_object.lower() in obj.name.lower()]

        if not target_objects:
            return None

        # Find closest object to center
        closest_obj = min(target_objects,
                         key=lambda obj: self._distance_to_point(obj.center, image_center))

        return closest_obj

    def _distance_to_point(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def is_object_graspable(self, obj: DetectedObject) -> bool:
        """Check if an object is suitable for grasping"""
        # Check if object type is graspable
        if obj.name not in self.graspable_objects:
            return False

        # Check size constraints (too small or too large)
        if obj.area < 1000 or obj.area > 100000:  # Adjust thresholds as needed
            return False

        return True

    def get_object_location_relative_to_robot(self, obj: DetectedObject,
                                            image_shape: Tuple[int, int]) -> str:
        """Get object location relative to robot (for navigation)"""
        img_width, img_height = image_shape
        center_x, center_y = obj.center

        # Determine horizontal position
        if center_x < img_width * 0.33:
            horizontal_pos = "left"
        elif center_x > img_width * 0.67:
            horizontal_pos = "right"
        else:
            horizontal_pos = "center"

        # Determine vertical position
        if center_y < img_height * 0.33:
            vertical_pos = "top"
        elif center_y > img_height * 0.67:
            vertical_pos = "bottom"
        else:
            vertical_pos = "middle"

        return f"{horizontal_pos} {vertical_pos}"
```

### 3. Vision-Language Integration System

```python
@dataclass
class ObjectBasedCommand:
    """Command that incorporates object detection results"""
    original_command: str
    intent: str
    detected_objects: List[DetectedObject]
    target_object: Optional[DetectedObject]
    action_parameters: Dict[str, any]

class VisionLanguageIntegrator:
    """Integrates vision and language processing for object-based commands"""

    def __init__(self):
        self.object_detector = ObjectDetector()
        self.object_associator = ObjectCommandAssociator()

    def process_command_with_vision(self, command_text: str, image: np.ndarray) -> ObjectBasedCommand:
        """Process command using both language and vision information"""
        # Detect objects in the image
        detected_objects = self.object_detector.detect_objects(image)

        # Parse the command to extract intent and entities
        intent, entities = self._parse_command(command_text)

        # Find relevant objects for the command
        target_object = None
        if 'object' in entities:
            target_object = self.object_associator.find_closest_object(
                entities['object'], detected_objects
            )

        # If no target object found but command requires one, use the most confident graspable object
        if not target_object and intent in ['manipulation', 'inspection']:
            graspable_objects = [obj for obj in detected_objects
                               if self.object_associator.is_object_graspable(obj)]
            if graspable_objects:
                target_object = max(graspable_objects, key=lambda obj: obj.confidence)

        # Generate action parameters based on detected objects and command
        action_params = self._generate_action_parameters(
            intent, entities, target_object, detected_objects, image.shape
        )

        return ObjectBasedCommand(
            original_command=command_text,
            intent=intent,
            detected_objects=detected_objects,
            target_object=target_object,
            action_parameters=action_params
        )

    def _parse_command(self, command: str) -> Tuple[str, Dict[str, str]]:
        """Parse command to extract intent and entities"""
        command_lower = command.lower()
        entities = {}

        # Extract intent
        if any(word in command_lower for word in ['go to', 'navigate', 'move to', 'walk to']):
            intent = 'navigation'
        elif any(word in command_lower for word in ['pick up', 'grasp', 'grab', 'lift', 'move', 'place']):
            intent = 'manipulation'
        elif any(word in command_lower for word in ['find', 'look at', 'see', 'show', 'locate']):
            intent = 'inspection'
        else:
            intent = 'unknown'

        # Extract object entities
        import re
        # Look for common object names in the command
        for obj_name in ['bottle', 'cup', 'book', 'ball', 'remote', 'phone', 'laptop', 'chair', 'table']:
            if obj_name in command_lower:
                entities['object'] = obj_name
                break

        # Extract location entities
        for loc_name in ['kitchen', 'living room', 'bedroom', 'office', 'table', 'shelf']:
            if loc_name in command_lower:
                entities['location'] = loc_name
                break

        return intent, entities

    def _generate_action_parameters(self, intent: str, entities: Dict[str, str],
                                  target_object: Optional[DetectedObject],
                                  all_objects: List[DetectedObject],
                                  image_shape: Tuple[int, int]) -> Dict[str, any]:
        """Generate action parameters based on vision and language"""
        params = {}

        if intent == 'navigation':
            # Navigation parameters based on object location
            if target_object:
                params['target_location'] = self.object_associator.get_object_location_relative_to_robot(
                    target_object, (image_shape[1], image_shape[0])  # width, height
                )
                params['target_coordinates'] = target_object.center

        elif intent == 'manipulation':
            # Manipulation parameters based on target object
            if target_object:
                params['object_name'] = target_object.name
                params['object_bbox'] = target_object.bbox
                params['object_center'] = target_object.center
                params['graspable'] = self.object_associator.is_object_graspable(target_object)

        elif intent == 'inspection':
            # Inspection parameters
            params['detected_objects'] = [obj.name for obj in all_objects]
            if target_object:
                params['focused_object'] = target_object.name
                params['object_details'] = {
                    'confidence': target_object.confidence,
                    'area': target_object.area,
                    'position': target_object.center
                }

        return params
```

### 4. Real-time Object Detection and Processing

```python
import threading
import queue
from collections import deque

class RealTimeObjectProcessor:
    """Real-time processor for continuous object detection and command integration"""

    def __init__(self):
        self.vision_language_integrator = VisionLanguageIntegrator()
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_running = False
        self.fps_counter = deque(maxlen=30)  # For FPS calculation

        # Video capture
        self.cap = None

    def start_camera_processing(self, camera_index: int = 0):
        """Start real-time processing from camera"""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        self.is_running = True

        # Start processing thread
        processing_thread = threading.Thread(target=self._process_camera_frames, daemon=True)
        processing_thread.start()

        return processing_thread

    def _process_camera_frames(self):
        """Process camera frames in real-time"""
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            start_time = time.time()

            # Check for pending commands
            try:
                command = self.command_queue.get_nowait()
                # Process command with current frame
                result = self.vision_language_integrator.process_command_with_vision(command, frame)
                self.result_queue.put(result)
            except queue.Empty:
                # Just detect objects for visualization
                detected_objects = self.vision_language_integrator.object_detector.detect_objects(frame)
                annotated_frame = self.vision_language_integrator.object_detector.draw_detections(
                    frame, detected_objects
                )
                # Display frame (in a real system, you might send it somewhere else)
                cv2.imshow('Object Detection', annotated_frame)

            # Calculate and store FPS
            processing_time = time.time() - start_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.fps_counter.append(fps)

            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def submit_command(self, command: str):
        """Submit a command for processing"""
        self.command_queue.put(command)

    def get_result(self, timeout: float = None) -> Optional[ObjectBasedCommand]:
        """Get the result of command processing"""
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_current_fps(self) -> float:
        """Get current processing FPS"""
        if self.fps_counter:
            return sum(self.fps_counter) / len(self.fps_counter)
        return 0.0

    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
```

### 5. Complete Integration Example

```python
class ObjectDetectionIntegrationSystem:
    """Complete system integrating object detection with voice commands and robot actions"""

    def __init__(self):
        self.vision_language_integrator = VisionLanguageIntegrator()
        self.object_detector = ObjectDetector()
        self.object_associator = ObjectCommandAssociator()
        self.real_time_processor = RealTimeObjectProcessor()
        self.robot_executor = None  # Would be connected to actual robot

    def process_voice_command_with_objects(self, command: str, image: np.ndarray):
        """Process a voice command using object detection results"""
        print(f"Processing command: {command}")

        # Integrate vision and language
        integrated_result = self.vision_language_integrator.process_command_with_vision(
            command, image
        )

        print(f"Intent: {integrated_result.intent}")
        print(f"Detected {len(integrated_result.detected_objects)} objects")

        if integrated_result.target_object:
            print(f"Target object: {integrated_result.target_object.name} "
                  f"with confidence {integrated_result.target_object.confidence:.2f}")

        # Generate robot actions based on integration result
        robot_actions = self._generate_robot_actions(integrated_result)

        return integrated_result, robot_actions

    def _generate_robot_actions(self, integrated_result: ObjectBasedCommand):
        """Generate robot actions based on integrated vision-language result"""
        actions = []

        if integrated_result.intent == 'navigation':
            if 'target_coordinates' in integrated_result.action_parameters:
                x, y = integrated_result.action_parameters['target_coordinates']
                actions.append({
                    'action': 'navigate_to_pixel',
                    'parameters': {'x': x, 'y': y}
                })

        elif integrated_result.intent == 'manipulation':
            if 'object_name' in integrated_result.action_parameters:
                obj_name = integrated_result.action_parameters['object_name']
                graspable = integrated_result.action_parameters.get('graspable', False)

                if graspable:
                    actions.extend([
                        {
                            'action': 'navigate_to_object',
                            'parameters': {'object_name': obj_name}
                        },
                        {
                            'action': 'grasp_object',
                            'parameters': {'object_name': obj_name}
                        }
                    ])
                else:
                    print(f"Object {obj_name} is not suitable for grasping")

        elif integrated_result.intent == 'inspection':
            detected_objs = integrated_result.action_parameters.get('detected_objects', [])
            print(f"Detected objects: {detected_objs}")

        return actions

    def run_demo(self):
        """Run a demonstration of object detection integration"""
        print("Object Detection Integration Demo")
        print("=" * 40)

        # Example image (in practice, this would come from camera)
        # For demo, we'll create a synthetic image
        demo_image = np.zeros((480, 640, 3), dtype=np.uint8)
        demo_image[:] = [200, 200, 200]  # Light gray background

        # Add some colored rectangles to simulate objects
        cv2.rectangle(demo_image, (100, 100), (200, 200), (0, 255, 0), -1)  # Green (could be bottle)
        cv2.rectangle(demo_image, (300, 300), (400, 400), (0, 0, 255), -1)  # Red (could be cup)
        cv2.putText(demo_image, "bottle", (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(demo_image, "cup", (300, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Example commands
        commands = [
            "Find the bottle",
            "Pick up the red cup",
            "Look at the objects"
        ]

        for command in commands:
            print(f"\nCommand: {command}")
            integrated_result, actions = self.process_voice_command_with_objects(command, demo_image)

            print(f"Generated actions: {actions}")

        print("\nDemo completed!")

    def start_real_time_demo(self):
        """Start real-time demo with camera"""
        print("Starting real-time object detection demo...")
        print("Say commands like 'Find the bottle' or 'Pick up the cup'")
        print("Press 'q' in the video window to quit")

        # Start camera processing
        processing_thread = self.real_time_processor.start_camera_processing()

        try:
            while True:
                # In a real system, you would get voice commands here
                # For this demo, we'll just keep processing
                result = self.real_time_processor.get_result(timeout=0.1)
                if result:
                    print(f"Processed: {result.original_command}")
                    print(f"Target object: {result.target_object.name if result.target_object else 'None'}")

                # Check if user wants to quit
                import sys
                import select
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    if sys.stdin.readline().strip().lower() == 'quit':
                        break

        except KeyboardInterrupt:
            print("\nStopping real-time demo...")
        finally:
            self.real_time_processor.stop_processing()

# Example usage
def main():
    """Main function to demonstrate object detection integration"""
    system = ObjectDetectionIntegrationSystem()

    print("Object Detection Integration System")
    print("1. Run demo with synthetic image")
    print("2. Run real-time demo with camera (requires camera access)")

    choice = input("Choose option (1 or 2): ").strip()

    if choice == "1":
        system.run_demo()
    elif choice == "2":
        system.start_real_time_demo()
    else:
        print("Invalid choice. Running demo with synthetic image...")
        system.run_demo()

if __name__ == "__main__":
    main()
```

## ROS 2 Integration Example

### 1. Object Detection ROS 2 Node

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectDetectionROSNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Initialize object detection system
        self.object_detector = ObjectDetector()
        self.vision_language_integrator = VisionLanguageIntegrator()
        self.cv_bridge = CvBridge()

        # Publishers
        self.detection_pub = self.create_publisher(String, 'object_detections', 10)
        self.image_pub = self.create_publisher(Image, 'annotated_image', 10)

        # Subscribers
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10
        )

        # Timer for processing
        self.process_timer = self.create_timer(0.1, self.process_pending_commands)

        self.get_logger().info('Object Detection ROS Node initialized')

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Detect objects
            detected_objects = self.object_detector.detect_objects(cv_image)

            # Draw detections
            annotated_image = self.object_detector.draw_detections(cv_image, detected_objects)

            # Publish detections
            detection_msg = String()
            detection_msg.data = self.format_detections(detected_objects)
            self.detection_pub.publish(detection_msg)

            # Publish annotated image
            annotated_msg = self.cv_bridge.cv2_to_imgmsg(annotated_image, 'bgr8')
            self.image_pub.publish(annotated_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def format_detections(self, detected_objects):
        """Format detections for message publishing"""
        detection_data = {
            'timestamp': self.get_clock().now().to_msg(),
            'objects': [
                {
                    'name': obj.name,
                    'confidence': obj.confidence,
                    'bbox': obj.bbox,
                    'center': obj.center
                }
                for obj in detected_objects
            ]
        }
        import json
        return json.dumps(detection_data)

    def process_pending_commands(self):
        """Process any pending commands that need object detection"""
        # This would integrate with voice command processing
        pass

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionROSNode()

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

## Performance Optimization

### 1. Efficient Object Detection Pipeline

```python
class OptimizedObjectDetector:
    """Optimized object detection for real-time performance"""

    def __init__(self):
        self.model = None
        self.transform = None
        self.warmup_model()

    def warmup_model(self):
        """Warm up the model to reduce first-inference latency"""
        # Load and run a dummy inference to initialize GPU/CPU
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _ = self.detect_objects(dummy_image)

    def detect_objects_optimized(self, image: np.ndarray) -> List[DetectedObject]:
        """Optimized object detection with performance considerations"""
        # Resize image to standard size for consistent processing time
        resized_image = cv2.resize(image, (640, 480))

        # Convert to tensor
        image_tensor = self.transform(resized_image).unsqueeze(0)

        # Run inference
        with torch.no_grad():
            predictions = self.model(image_tensor)

        # Process results
        return self._process_predictions_optimized(predictions)

    def _process_predictions_optimized(self, predictions):
        """Optimized prediction processing"""
        # Vectorized operations for better performance
        boxes = predictions[0]['boxes'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()

        # Filter by confidence threshold using numpy operations
        valid_indices = scores > self.confidence_threshold
        boxes = boxes[valid_indices]
        labels = labels[valid_indices]
        scores = scores[valid_indices]

        # Create DetectedObject instances
        detected_objects = []
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.astype(int)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            area = (x2 - x1) * (y2 - y1)

            detected_obj = DetectedObject(
                name=self.coco_names[label],
                confidence=score,
                bbox=(x1, y1, x2, y2),
                center=(center_x, center_y),
                area=area
            )
            detected_objects.append(detected_obj)

        return detected_objects
```

## Key Features

1. **Real-time Object Detection**: Detects objects in camera feeds with bounding boxes
2. **Vision-Language Integration**: Associates detected objects with voice commands
3. **Object Command Association**: Matches objects to specific command intents
4. **Action Parameter Generation**: Creates robot action parameters from visual information
5. **ROS 2 Integration**: Example of integration with ROS 2 robotics framework
6. **Performance Optimization**: Techniques for real-time processing

## Learning Outcomes

After implementing this example, you should understand:

- How to integrate object detection with voice command processing
- Techniques for associating visual information with language commands
- Methods for generating robot actions from detected objects
- Real-time processing considerations for vision-language systems
- ROS 2 integration patterns for object detection systems

## Next Steps

This example can be extended with:

- Custom object detection models trained on specific robot environments
- 3D object detection and pose estimation
- Multi-camera object tracking
- Integration with manipulation planning for grasping
- Advanced scene understanding beyond object detection