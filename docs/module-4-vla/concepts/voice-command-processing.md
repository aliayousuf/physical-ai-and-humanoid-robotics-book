---
title: "Voice Command Processing"
description: "Processing natural language voice commands for robot control"
---

# Voice Command Processing

## Overview

Voice command processing enables robots to understand and respond to spoken natural language commands. This capability allows for more intuitive human-robot interaction, moving beyond traditional button-based or gesture-based interfaces to enable communication using natural language.

## Voice Processing Pipeline

The voice processing pipeline typically follows this sequence:

```
Audio Input → Speech Recognition → Natural Language Processing → Intent Classification → Action Mapping → Robot Execution
```

### 1. Audio Input and Preprocessing

Capturing and preparing audio for processing:

```python
import pyaudio
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy import signal
import librosa

class AudioPreprocessor:
    def __init__(self, sample_rate=16000, frame_duration_ms=30):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)

        # Voice activity detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2

        # Audio parameters
        self.chunk_size = 1024
        self.silence_threshold = 0.01  # Silence threshold
        self.energy_threshold = 0.05   # Minimum energy for speech

    def capture_audio(self, duration=5.0):
        """Capture audio from microphone with voice activity detection"""
        p = pyaudio.PyAudio()

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )

        frames = []
        silence_frames = 0
        max_silence_frames = 50  # Stop after 50 frames of silence
        speech_detected = False

        for _ in range(0, int(self.sample_rate / self.chunk_size * duration)):
            data = stream.read(self.chunk_size)
            audio_data = np.frombuffer(data, dtype=np.float32)

            # Calculate energy
            energy = np.sqrt(np.mean(audio_data**2))

            # Check for voice activity
            if energy > self.energy_threshold:
                frames.append(audio_data)
                speech_detected = True
                silence_frames = 0
            elif speech_detected:  # Only count silence after speech begins
                frames.append(audio_data)
                silence_frames += 1

                # Stop early if too much silence after speech starts
                if silence_frames > max_silence_frames:
                    break

        stream.stop_stream()
        stream.close()
        p.terminate()

        # Combine all frames
        audio_signal = np.concatenate(frames)
        return audio_signal

    def preprocess_audio(self, audio_signal):
        """Preprocess audio for speech recognition"""
        # Normalize audio
        audio_normalized = audio_signal / np.max(np.abs(audio_signal))

        # Apply pre-emphasis filter
        pre_emphasis = 0.97
        audio_preemphasized = np.append(
            audio_normalized[0],
            audio_normalized[1:] - pre_emphasis * audio_normalized[:-1]
        )

        # Remove DC offset
        audio_filtered = audio_preemphasized - np.mean(audio_preemphasized)

        # Apply noise reduction if needed
        audio_cleaned = self.reduce_noise(audio_filtered)

        return audio_cleaned

    def reduce_noise(self, audio_signal):
        """Simple noise reduction using spectral gating"""
        # This is a simplified noise reduction approach
        # In practice, more sophisticated methods would be used
        return audio_signal  # Placeholder for now

    def detect_voice_activity(self, audio_frame):
        """Detect voice activity in audio frame using WebRTC VAD"""
        # Convert to 16-bit for WebRTC VAD
        audio_16bit = (audio_frame * 32767).astype(np.int16)

        # Check if frame size is valid for VAD (10, 20, or 30 ms)
        frame_size_samples = len(audio_16bit)
        expected_sizes = [int(self.sample_rate * t) for t in [0.01, 0.02, 0.03]]  # 10, 20, 30ms

        if frame_size_samples not in expected_sizes:
            # Pad or trim to nearest valid size
            valid_size = min(expected_sizes, key=lambda x: abs(x - frame_size_samples))
            if len(audio_16bit) > valid_size:
                audio_16bit = audio_16bit[:valid_size]
            else:
                pad_width = valid_size - len(audio_16bit)
                audio_16bit = np.pad(audio_16bit, (0, pad_width), mode='constant')

        try:
            return self.vad.is_speech(audio_16bit.tobytes(), self.sample_rate)
        except:
            return False  # If VAD fails, assume no speech
```

### 2. Speech Recognition

Converting speech to text using various approaches:

```python
import speech_recognition as sr
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import whisper  # OpenAI Whisper

class SpeechRecognizer:
    def __init__(self, model_type="whisper"):
        self.model_type = model_type

        if model_type == "whisper":
            # Initialize Whisper model for offline recognition
            self.whisper_model = whisper.load_model("base")
        elif model_type == "vosk":
            # Initialize Vosk model for offline recognition
            from vosk import Model
            self.vosk_model = Model("model")  # Path to vosk model
            self.recognizer = KaldiRecognizer(self.vosk_model, 16000)
        else:
            # Use speech_recognition library for online recognition
            self.recognizer = sr.Recognizer()

    def recognize_speech(self, audio_data):
        """Recognize speech from audio data"""
        if self.model_type == "whisper":
            return self.recognize_with_whisper(audio_data)
        elif self.model_type == "vosk":
            return self.recognize_with_vosk(audio_data)
        else:
            return self.recognize_with_google(audio_data)

    def recognize_with_whisper(self, audio_data):
        """Offline speech recognition using OpenAI Whisper"""
        # Convert audio to the format expected by Whisper
        audio_tensor = torch.from_numpy(audio_data).float()

        # Process with Whisper
        result = self.whisper_model.transcribe(audio_tensor.numpy())
        return result["text"].strip()

    def recognize_with_vosk(self, audio_data):
        """Offline speech recognition using Vosk"""
        import json
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self.vosk_model, 16000)
        rec.AcceptWaveform(audio_data.tobytes())
        result = rec.Result()

        return json.loads(result)["text"]

    def recognize_with_google(self, audio_data):
        """Online speech recognition using Google Speech API"""
        # For this to work, we'd need to convert numpy array to AudioData format
        # This is a simplified example
        try:
            # In practice, you'd need to convert the numpy array to the proper format
            # This is a placeholder implementation
            return "placeholder recognition result"
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return ""
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return ""
```

## Natural Language Processing for Robotics

### 1. Intent Classification

Identifying the user's intent from spoken commands:

```python
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import numpy as np

class IntentClassifier:
    def __init__(self):
        # Load spaCy model for English
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Define robot command patterns and intents
        self.intent_patterns = {
            'navigation': [
                r'go to (.+)',
                r'move to (.+)',
                r'go to the (.+)',
                r'walk to (.+)',
                r'travel to (.+)',
                r'navigate to (.+)',
                r'go to (.+)',
                r'move towards (.+)',
                r'head to (.+)'
            ],
            'manipulation': [
                r'pick up (.+)',
                r'grasp (.+)',
                r'grab (.+)',
                r'lift (.+)',
                r'place (.+) (on|in|at) (.+)',
                r'put (.+) (on|in|at) (.+)',
                r'move (.+) (from|to) (.+)',
                r'collect (.+)',
                r'get (.+)'
            ],
            'inspection': [
                r'look at (.+)',
                r'find (.+)',
                r'locate (.+)',
                r'scan (.+)',
                r'examine (.+)',
                r'inspect (.+)',
                r'search for (.+)',
                r'where is (.+)'
            ],
            'communication': [
                r'tell me about (.+)',
                r'describe (.+)',
                r'what do you see',
                r'analyze (.+)',
                r'report on (.+)',
                r'give me status',
                r'how are you doing'
            ],
            'cleanup': [
                r'clean (.+)',
                r'clean up (.+)',
                r'tidy (.+)',
                r'organize (.+)',
                r'clear (.+)',
                r'pickup (.+)',
                r'collect trash',
                r'gather objects'
            ],
            'stop': [
                r'stop',
                r'abort',
                r'emergency stop',
                r'halt',
                r'pause',
                r'freeze',
                r'cease'
            ],
            'help': [
                r'help',
                r'what can you do',
                r'how do i',
                r'commands',
                r'options',
                r'assist me'
            ]
        }

        # Initialize classifier
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.is_trained = False

    def classify_intent(self, text):
        """Classify intent using pattern matching and ML classification"""
        text_lower = text.lower().strip()

        # First try pattern matching
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Extract entities using the matched pattern
                    match = re.search(pattern, text_lower)
                    entities = match.groups() if match else ()
                    return intent, entities, 0.9  # High confidence for pattern match

        # If no pattern matches, use ML classification
        if self.is_trained:
            return self.ml_classify_intent(text)
        else:
            # Use NLP-based approach without training
            return self.nlp_based_classification(text)

    def ml_classify_intent(self, text):
        """Classify intent using trained ML model"""
        text_vector = self.vectorizer.transform([text])
        prediction = self.classifier.predict(text_vector)[0]
        confidence = max(self.classifier.predict_proba(text_vector)[0])

        # Extract entities based on the predicted intent
        entities = self.extract_entities(text, prediction)
        return prediction, entities, confidence

    def nlp_based_classification(self, text):
        """Use NLP to classify intent without training"""
        if not self.nlp:
            return 'unknown', [], 0.0

        doc = self.nlp(text)

        # Extract verbs and objects to determine intent
        verbs = [token.lemma_ for token in doc if token.pos_ == 'VERB']
        entities = [ent.text for ent in doc.ents]

        # Simple rule-based classification based on verb patterns
        navigation_verbs = ['go', 'move', 'navigate', 'walk', 'travel', 'head', 'drive']
        manipulation_verbs = ['pick', 'grasp', 'grab', 'lift', 'place', 'put', 'move', 'collect', 'get']
        inspection_verbs = ['look', 'find', 'locate', 'scan', 'examine', 'inspect', 'search']
        communication_verbs = ['tell', 'describe', 'report', 'analyze']
        cleanup_verbs = ['clean', 'tidy', 'organize', 'clear', 'pickup']

        if any(verb in navigation_verbs for verb in verbs):
            return 'navigation', entities, 0.7
        elif any(verb in manipulation_verbs for verb in verbs):
            return 'manipulation', entities, 0.7
        elif any(verb in inspection_verbs for verb in verbs):
            return 'inspection', entities, 0.7
        elif any(verb in communication_verbs for verb in verbs):
            return 'communication', entities, 0.7
        elif any(verb in cleanup_verbs for verb in verbs):
            return 'cleanup', entities, 0.7
        elif any(word in ['stop', 'halt', 'abort', 'pause'] for word in text.lower().split()):
            return 'stop', [], 0.8
        elif any(word in ['help', 'what', 'how', 'options'] for word in text.lower().split()):
            return 'help', [], 0.8
        else:
            return 'unknown', entities, 0.3

    def extract_entities(self, text, intent):
        """Extract entities from text based on intent"""
        doc = self.nlp(text) if self.nlp else None

        entities = {}

        # Named entity recognition
        if doc:
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'OBJECT']:
                    entities[ent.label_.lower()] = ent.text

        # Extract objects and locations from text
        text_lower = text.lower()

        # Look for common object patterns
        object_keywords = [
            'bottle', 'cup', 'book', 'box', 'chair', 'table', 'ball',
            'toy', 'phone', 'computer', 'laptop', 'remote', 'keys',
            'kitchen', 'living room', 'bedroom', 'office', 'bathroom',
            'hallway', 'door', 'window', 'couch', 'sofa', 'desk'
        ]

        found_objects = []
        found_locations = []

        for keyword in object_keywords:
            if keyword in text_lower:
                if keyword in ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'hallway']:
                    found_locations.append(keyword)
                else:
                    found_objects.append(keyword)

        if found_objects:
            entities['objects'] = found_objects
        if found_locations:
            entities['locations'] = found_locations

        return entities
```

## Integration with Robot Control

### 1. Voice Command Handler

Creating a handler that processes voice commands and converts them to robot actions:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Duration
import threading
import queue
import time

class VoiceCommandHandler(Node):
    def __init__(self):
        super().__init__('voice_command_handler')

        # Initialize components
        self.audio_preprocessor = AudioPreprocessor()
        self.speech_recognizer = SpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = ActionMapper()

        # ROS publishers and subscribers
        self.voice_cmd_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_command_callback,
            10
        )

        self.robot_cmd_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.status_pub = self.create_publisher(
            String,
            '/voice_status',
            10
        )

        # Command queue for processing
        self.command_queue = queue.Queue()
        self.is_processing = False

        # Processing thread
        self.processing_thread = threading.Thread(target=self.process_commands, daemon=True)
        self.processing_thread.start()

        self.get_logger().info('Voice command handler initialized')

    def voice_command_callback(self, msg):
        """Process incoming voice command"""
        try:
            # Classify intent and extract entities
            intent, entities, confidence = self.intent_classifier.classify_intent(msg.data)

            if confidence > 0.5:  # Only process if confidence is high enough
                # Map intent to robot action
                robot_action = self.action_mapper.map_to_action(intent, entities, msg.data)

                # Add to processing queue
                command_data = {
                    'intent': intent,
                    'entities': entities,
                    'action': robot_action,
                    'original_command': msg.data,
                    'confidence': confidence,
                    'timestamp': self.get_clock().now()
                }

                self.command_queue.put(command_data)

                # Publish status
                status_msg = String()
                status_msg.data = f"Recognized: {intent} with confidence {confidence:.2f}"
                self.status_pub.publish(status_msg)

                self.get_logger().info(f'Processed command: {msg.data} -> {intent}')
            else:
                self.get_logger().warning(f'Low confidence ({confidence:.2f}) for command: {msg.data}')

        except Exception as e:
            self.get_logger().error(f'Error processing voice command: {e}')

    def process_commands(self):
        """Process commands from the queue in a separate thread"""
        while rclpy.ok():
            try:
                if not self.command_queue.empty():
                    command_data = self.command_queue.get_nowait()

                    with self.processing_lock:
                        self.is_processing = True
                        self.execute_action(command_data)
                        self.is_processing = False

            except queue.Empty:
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting
            except Exception as e:
                self.get_logger().error(f'Error in command processing thread: {e}')

    def execute_action(self, command_data):
        """Execute the mapped robot action"""
        action = command_data['action']
        intent = command_data['intent']

        try:
            if intent == 'navigation':
                self.execute_navigation_action(action)
            elif intent == 'manipulation':
                self.execute_manipulation_action(action)
            elif intent == 'inspection':
                self.execute_inspection_action(action)
            elif intent == 'cleanup':
                self.execute_cleanup_action(action)
            elif intent == 'stop':
                self.execute_stop_action()
            else:
                self.get_logger().warning(f'Unknown intent: {intent}')

        except Exception as e:
            self.get_logger().error(f'Error executing action: {e}')

    def execute_navigation_action(self, action):
        """Execute navigation action"""
        # Extract destination from action
        destination = action.get('destination', 'unknown')

        # Create navigation command
        cmd = Twist()

        # In a real implementation, this would send a navigation goal
        # For now, we'll use a simple movement approach
        if destination == 'kitchen':
            cmd.linear.x = 1.0  # Move forward
            cmd.angular.z = 0.2  # Turn right slightly
        elif destination == 'living room':
            cmd.linear.x = 1.0
            cmd.angular.z = -0.2  # Turn left slightly
        elif destination == 'bedroom':
            cmd.linear.x = 0.5
            cmd.angular.z = 0.5  # Turn left more
        else:
            # Default forward movement for unknown destinations
            cmd.linear.x = 0.5

        # Publish command
        self.robot_cmd_pub.publish(cmd)

    def execute_manipulation_action(self, action):
        """Execute manipulation action"""
        # This would typically involve more complex manipulation planning
        # For now, we'll log the intended action
        obj = action.get('object', 'unknown')
        self.get_logger().info(f'Intended manipulation action for object: {obj}')

    def execute_inspection_action(self, action):
        """Execute inspection action"""
        # This would involve directing robot to look at specific objects
        target = action.get('target', 'unknown')
        self.get_logger().info(f'Intended inspection action for target: {target}')

    def execute_cleanup_action(self, action):
        """Execute cleanup action"""
        # This would involve a sequence of navigation and manipulation actions
        self.get_logger().info('Executing cleanup action sequence')

    def execute_stop_action(self):
        """Execute stop action"""
        cmd = Twist()  # Zero velocities
        self.robot_cmd_pub.publish(cmd)
        self.get_logger().info('Robot stopped by voice command')

    def get_processing_status(self):
        """Get current processing status"""
        return {
            'is_processing': self.is_processing,
            'queue_size': self.command_queue.qsize(),
            'last_command': getattr(self, '_last_command', None)
        }

class ActionMapper:
    """Maps intents and entities to robot actions"""

    def __init__(self):
        self.action_templates = {
            'navigation': {
                'template': 'navigate_to(location="{location}")',
                'required_entities': ['location']
            },
            'manipulation': {
                'template': 'manipulate_object(object="{object}", destination="{destination}")',
                'required_entities': ['object']
            },
            'inspection': {
                'template': 'inspect_target(target="{target}")',
                'required_entities': ['target']
            }
        }

    def map_to_action(self, intent, entities, original_command):
        """Map intent and entities to robot action"""
        if intent == 'navigation':
            return self.map_navigation_action(entities, original_command)
        elif intent == 'manipulation':
            return self.map_manipulation_action(entities, original_command)
        elif intent == 'inspection':
            return self.map_inspection_action(entities, original_command)
        elif intent == 'stop':
            return {'action': 'stop_robot', 'params': {}}
        else:
            return {'action': 'unknown', 'params': {'original_command': original_command}}

    def map_navigation_action(self, entities, original_command):
        """Map navigation intent to specific navigation action"""
        # Extract destination from entities or command
        destination = None

        if 'locations' in entities and entities['locations']:
            destination = entities['locations'][0]
        elif 'loc' in entities:
            destination = entities['loc']
        else:
            # Try to extract location from original command
            destination = self.extract_location_from_command(original_command)

        return {
            'action': 'navigate_to',
            'destination': destination,
            'original_command': original_command
        }

    def map_manipulation_action(self, entities, original_command):
        """Map manipulation intent to specific manipulation action"""
        obj = None
        destination = None

        if 'objects' in entities and entities['objects']:
            obj = entities['objects'][0]
        elif 'object' in entities:
            obj = entities['object']
        else:
            obj = self.extract_object_from_command(original_command)

        # Extract destination if specified
        destination = self.extract_destination_from_command(original_command)

        return {
            'action': 'manipulate_object',
            'object': obj,
            'destination': destination,
            'original_command': original_command
        }

    def map_inspection_action(self, entities, original_command):
        """Map inspection intent to specific inspection action"""
        target = None

        if 'objects' in entities and entities['objects']:
            target = entities['objects'][0]
        elif 'target' in entities:
            target = entities['target']
        else:
            target = self.extract_target_from_command(original_command)

        return {
            'action': 'inspect_target',
            'target': target,
            'original_command': original_command
        }

    def extract_location_from_command(self, command):
        """Extract location from command using pattern matching"""
        command_lower = command.lower()

        # Common location patterns
        location_patterns = [
            r'to the (\w+)',
            r'to (\w+)',
            r'go to (\w+)',
            r'move to (\w+)',
            r'in the (\w+)',
            r'toward the (\w+)'
        ]

        for pattern in location_patterns:
            match = re.search(pattern, command_lower)
            if match:
                location = match.group(1)

                # Validate that it's actually a location
                if location in ['kitchen', 'living room', 'bedroom', 'office', 'bathroom', 'hallway', 'dining room']:
                    return location

        return 'unknown_location'

    def extract_object_from_command(self, command):
        """Extract object from command using pattern matching"""
        command_lower = command.lower()

        # Common object patterns
        object_patterns = [
            r'pick up the (\w+)',
            r'grasp the (\w+)',
            r'grab the (\w+)',
            r'the (\w+)',
            r'(\w+) on the table',
            r'(\w+) on the counter'
        ]

        for pattern in object_patterns:
            match = re.search(pattern, command_lower)
            if match:
                obj = match.group(1)
                return obj

        return 'unknown_object'

    def extract_destination_from_command(self, command):
        """Extract destination from command using pattern matching"""
        command_lower = command.lower()

        # Common destination patterns
        destination_patterns = [
            r'put it in the (\w+)',
            r'place it on the (\w+)',
            r'put it on the (\w+)',
            r'place it in the (\w+)'
        ]

        for pattern in destination_patterns:
            match = re.search(pattern, command_lower)
            if match:
                destination = match.group(1)
                return destination

        return 'default_destination'
```

## Voice Command Processing with Context Awareness

### 1. Contextual Understanding

Adding context awareness to voice processing:

```python
class ContextAwareVoiceProcessor:
    def __init__(self):
        self.context_memory = ContextMemory()
        self.dialogue_manager = DialogueManager()
        self.intent_classifier = IntentClassifier()
        self.action_mapper = ActionMapper()

    def process_command_with_context(self, command, current_context=None):
        """Process command considering current context"""
        # Get current context if not provided
        if current_context is None:
            current_context = self.context_memory.get_current_context()

        # Classify intent with context
        intent, entities, confidence = self.intent_classifier.classify_intent(command)

        # Resolve ambiguous references using context
        resolved_entities = self.resolve_entities_with_context(entities, current_context)

        # Update context with new information
        self.context_memory.update_context(command, intent, resolved_entities)

        # Map to action
        action = self.action_mapper.map_to_action(intent, resolved_entities, command)

        # Handle dialogue flow
        response = self.dialogue_manager.generate_response(intent, resolved_entities, current_context)

        return {
            'intent': intent,
            'entities': resolved_entities,
            'action': action,
            'confidence': confidence,
            'response': response,
            'context': current_context
        }

    def resolve_entities_with_context(self, entities, context):
        """Resolve ambiguous entities using context"""
        resolved_entities = entities.copy()

        # Handle pronouns and references
        for key, value in entities.items():
            if isinstance(value, str) and value.lower() in ['it', 'that', 'them', 'there']:
                # Resolve based on context
                if key == 'object' and value.lower() in ['it', 'that']:
                    # Get most recently mentioned object
                    last_object = self.context_memory.get_recent_object()
                    if last_object:
                        resolved_entities[key] = last_object
                elif key == 'location' and value.lower() == 'there':
                    # Get most recently mentioned location
                    last_location = self.context_memory.get_recent_location()
                    if last_location:
                        resolved_entities[key] = last_location

        return resolved_entities

class ContextMemory:
    def __init__(self):
        self.memory = {
            'recent_objects': [],
            'recent_locations': [],
            'recent_intents': [],
            'robot_state': {},
            'environment_state': {}
        }
        self.max_memory_size = 10

    def update_context(self, command, intent, entities):
        """Update context memory with new information"""
        # Update recent objects
        if 'objects' in entities:
            for obj in entities['objects']:
                self.memory['recent_objects'].append(obj)
                if len(self.memory['recent_objects']) > self.max_memory_size:
                    self.memory['recent_objects'].pop(0)

        # Update recent locations
        if 'locations' in entities:
            for loc in entities['locations']:
                self.memory['recent_locations'].append(loc)
                if len(self.memory['recent_locations']) > self.max_memory_size:
                    self.memory['recent_locations'].pop(0)

        # Update recent intents
        self.memory['recent_intents'].append(intent)
        if len(self.memory['recent_intents']) > self.max_memory_size:
            self.memory['recent_intents'].pop(0)

    def get_current_context(self):
        """Get current context for processing"""
        return {
            'recent_objects': self.memory['recent_objects'][-3:],  # Last 3 objects
            'recent_locations': self.memory['recent_locations'][-3:],  # Last 3 locations
            'recent_intents': self.memory['recent_intents'][-3:],  # Last 3 intents
            'robot_state': self.memory['robot_state'],
            'environment_state': self.memory['environment_state']
        }

    def get_recent_object(self):
        """Get most recently mentioned object"""
        if self.memory['recent_objects']:
            return self.memory['recent_objects'][-1]
        return None

    def get_recent_location(self):
        """Get most recently mentioned location"""
        if self.memory['recent_locations']:
            return self.memory['recent_locations'][-1]
        return None

class DialogueManager:
    def __init__(self):
        self.conversation_history = []

    def generate_response(self, intent, entities, context):
        """Generate appropriate response based on intent and context"""
        if intent == 'navigation':
            destination = entities.get('destination', 'unknown')
            return f"Navigating to {destination}."
        elif intent == 'manipulation':
            obj = entities.get('object', 'unknown')
            return f"Attempting to manipulate {obj}."
        elif intent == 'inspection':
            target = entities.get('target', 'unknown')
            return f"Inspecting {target}."
        elif intent == 'stop':
            return "Stopping robot operation."
        elif intent == 'help':
            return self.generate_help_response(context)
        else:
            return f"Processing command with intent: {intent}."

    def generate_help_response(self, context):
        """Generate help response based on context"""
        return "I can help with navigation, manipulation, inspection, and cleanup tasks. For example, you can say 'Go to the kitchen' or 'Pick up the bottle'."
```

## Performance Optimization

### 1. Efficient Processing Pipeline

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class EfficientVoiceProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.audio_preprocessor = AudioPreprocessor()
        self.speech_recognizer = SpeechRecognizer()
        self.intent_classifier = IntentClassifier()
        self.pipeline_cache = {}  # Cache for frequent commands

    async def process_voice_command_async(self, audio_data):
        """Process voice command asynchronously for better performance"""
        loop = asyncio.get_event_loop()

        # Preprocess audio in background
        processed_audio = await loop.run_in_executor(
            self.executor,
            self.audio_preprocessor.preprocess_audio,
            audio_data
        )

        # Recognize speech
        text = await loop.run_in_executor(
            self.executor,
            self.speech_recognizer.recognize_speech,
            processed_audio
        )

        if text and len(text.strip()) > 0:
            # Classify intent
            intent, entities, confidence = await loop.run_in_executor(
                self.executor,
                self.intent_classifier.classify_intent,
                text
            )

            return {
                'text': text,
                'intent': intent,
                'entities': entities,
                'confidence': confidence
            }

        return None

    def warm_models(self):
        """Warm up models to reduce first-request latency"""
        # Process a dummy request to load models into memory
        dummy_audio = np.zeros(16000)  # 1 second of silence
        try:
            result = self.process_voice_command(dummy_audio)
            print("Models warmed up successfully")
        except Exception as e:
            print(f"Model warming failed: {e}")

    def cache_frequent_commands(self):
        """Cache processing results for frequently used commands"""
        # This would identify and cache common command patterns
        pass
```

## Learning Objectives

After completing this section, you should understand:
- The complete voice processing pipeline from audio input to robot action
- How to implement speech recognition with both online and offline approaches
- Techniques for natural language understanding and intent classification
- How to map voice commands to robot actions with context awareness
- Performance optimization techniques for real-time voice processing
- Error handling and validation for voice command systems

## Next Steps

Continue to learn about [Action Sequence Generation](./action-sequence-generation) to understand how to convert high-level voice commands into specific sequences of robot actions.