---
title: "Unity Robot Control Example"
description: "Controlling robots through Unity interface with teleoperation and visualization"
---

# Unity Robot Control Example

## Overview

This example demonstrates how to create Unity-based interfaces for controlling robots, particularly for teleoperation scenarios. We'll cover creating intuitive control interfaces, implementing various control schemes, and visualizing robot state in real-time within the Unity environment.

## Prerequisites

Before implementing Unity robot control, ensure you have:
- Unity Editor (2021.3 LTS or later)
- ROS# package or similar for ROS communication
- Basic understanding of Unity UI and input systems
- Knowledge of robot kinematics and control principles
- Working robot model with ROS control interfaces

## Setting Up Unity Project for Robot Control

### 1. Project Configuration

Create a new Unity project and configure it for robotics applications:

1. Create a new 3D project in Unity Hub
2. Install necessary packages via Package Manager:
   - XR Interaction Toolkit (for VR support)
   - Input System (for advanced input handling)
   - ProBuilder (for quick environment prototyping)

3. Install ROS# or similar ROS communication package:
   - Download from Unity Asset Store or GitHub
   - Import into your project
   - Configure for your ROS distribution

### 2. Basic Scene Setup

Create the basic scene structure for robot control:

```csharp
using UnityEngine;

public class RobotControlScene : MonoBehaviour
{
    [Header("Robot Configuration")]
    public GameObject robotModel;
    public Transform robotSpawnPoint;

    [Header("Camera Configuration")]
    public Camera mainCamera;
    public Camera robotCamera;  // First-person view from robot
    public float followDistance = 5f;
    public float followHeight = 2f;

    [Header("UI Configuration")]
    public Canvas controlCanvas;
    public GameObject teleopPanel;

    void Start()
    {
        InitializeScene();
    }

    void InitializeScene()
    {
        // Spawn robot at designated location
        if (robotModel != null && robotSpawnPoint != null)
        {
            Instantiate(robotModel, robotSpawnPoint.position, robotSpawnPoint.rotation);
        }

        // Configure cameras
        SetupCameras();

        // Initialize UI
        SetupUI();
    }

    void SetupCameras()
    {
        if (mainCamera != null)
        {
            // Configure main camera properties
            mainCamera.fieldOfView = 60f;
            mainCamera.nearClipPlane = 0.1f;
            mainCamera.farClipPlane = 1000f;
        }
    }

    void SetupUI()
    {
        if (teleopPanel != null)
        {
            teleopPanel.SetActive(true);
        }
    }

    void LateUpdate()
    {
        // Update camera following if needed
        UpdateCameraFollow();
    }

    void UpdateCameraFollow()
    {
        if (robotModel != null && mainCamera != null)
        {
            // Simple follow implementation
            Vector3 targetPosition = robotModel.transform.position +
                                   Vector3.up * followHeight -
                                   robotModel.transform.forward * followDistance;
            mainCamera.transform.position = targetPosition;
            mainCamera.transform.LookAt(robotModel.transform);
        }
    }
}
```

## Basic Teleoperation Interface

### 1. Movement Controls

Create a basic interface for robot movement control:

```csharp
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;

public class RobotMovementController : MonoBehaviour
{
    [Header("Movement Configuration")]
    public float moveSpeed = 1.0f;
    public float turnSpeed = 1.0f;
    public float maxVelocity = 2.0f;

    [Header("Input Axes")]
    public string horizontalAxis = "Horizontal";
    public string verticalAxis = "Vertical";
    public string turnAxis = "Mouse X";

    [Header("UI Elements")]
    public Slider speedSlider;
    public Text velocityDisplay;
    public Button forwardButton;
    public Button backwardButton;
    public Button leftButton;
    public Button rightButton;

    [Header("ROS Integration")]
    public bool useROS = true;
    private RosSharp.RosBridgeClient.RosSocket rosSocket;

    private Rigidbody robotRigidbody;
    private float currentSpeed = 1.0f;
    private Vector3 targetVelocity;

    void Start()
    {
        InitializeController();
    }

    void InitializeController()
    {
        // Get robot rigidbody if available
        robotRigidbody = GetComponent<Rigidbody>();

        // Setup UI event listeners
        if (speedSlider != null)
        {
            speedSlider.onValueChanged.AddListener(OnSpeedChanged);
            speedSlider.value = currentSpeed;
        }

        SetupButtonListeners();
    }

    void SetupButtonListeners()
    {
        if (forwardButton != null)
            forwardButton.onClick.AddListener(() => MoveRobot(Vector3.forward));

        if (backwardButton != null)
            backwardButton.onClick.AddListener(() => MoveRobot(Vector3.back));

        if (leftButton != null)
            leftButton.onClick.AddListener(() => MoveRobot(Vector3.left));

        if (rightButton != null)
            rightButton.onClick.AddListener(() => MoveRobot(Vector3.right));
    }

    void Update()
    {
        HandleInput();
        UpdateUI();
    }

    void HandleInput()
    {
        // Get input from axes
        float horizontal = Input.GetAxis(horizontalAxis);
        float vertical = Input.GetAxis(verticalAxis);
        float turn = Input.GetAxis(turnAxis);

        // Calculate movement direction
        Vector3 movement = new Vector3(horizontal, 0, vertical).normalized;

        // Apply movement
        if (movement.magnitude > 0.1f)
        {
            MoveRobot(movement);
        }

        // Apply turning
        if (Mathf.Abs(turn) > 0.1f)
        {
            TurnRobot(turn);
        }
    }

    public void MoveRobot(Vector3 direction)
    {
        if (robotRigidbody != null)
        {
            // Apply movement using physics
            Vector3 worldDirection = transform.TransformDirection(direction);
            Vector3 targetVelocity = worldDirection * moveSpeed * currentSpeed;

            // Limit velocity
            targetVelocity = Vector3.ClampMagnitude(targetVelocity, maxVelocity * currentSpeed);

            // Apply force to rigidbody
            robotRigidbody.velocity = targetVelocity;
        }
        else
        {
            // Fallback: direct transform manipulation
            transform.Translate(direction * moveSpeed * currentSpeed * Time.deltaTime, Space.Self);
        }

        // Send command to ROS if connected
        if (useROS)
        {
            SendVelocityCommand(direction);
        }
    }

    public void TurnRobot(float turnAmount)
    {
        // Apply rotation
        transform.Rotate(Vector3.up, turnAmount * turnSpeed * currentSpeed * Time.deltaTime);

        // Send rotation command to ROS if connected
        if (useROS)
        {
            SendTurnCommand(turnAmount);
        }
    }

    void OnSpeedChanged(float newValue)
    {
        currentSpeed = newValue;
    }

    void UpdateUI()
    {
        if (velocityDisplay != null && robotRigidbody != null)
        {
            float currentVel = robotRigidbody.velocity.magnitude;
            velocityDisplay.text = $"Velocity: {currentVel:F2} m/s";
        }
    }

    // ROS Communication Methods
    void SendVelocityCommand(Vector3 direction)
    {
        // Implementation would send Twist message to ROS
        // This is a placeholder - actual implementation depends on ROS# setup
    }

    void SendTurnCommand(float turnAmount)
    {
        // Implementation would send rotational command to ROS
    }

    // Public methods for external control
    public void SetMoveSpeed(float speed)
    {
        moveSpeed = speed;
    }

    public void SetTurnSpeed(float speed)
    {
        turnSpeed = speed;
    }
}
```

### 2. Advanced Control Panel

Create a more sophisticated control panel with multiple input methods:

```csharp
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using System.Collections.Generic;

public class AdvancedRobotControlPanel : MonoBehaviour
{
    [Header("Control Modes")]
    public ControlMode currentMode = ControlMode.Teleoperation;
    public enum ControlMode { Teleoperation, Autonomous, Waypoint, Simulation }

    [Header("Teleoperation Controls")]
    public Joystick virtualJoystick;
    public Slider linearSpeedSlider;
    public Slider angularSpeedSlider;
    public Toggle velocityModeToggle;
    public Toggle positionModeToggle;

    [Header("Autonomous Controls")]
    public Button startAutonomousButton;
    public Button stopAutonomousButton;
    public Dropdown missionTypeDropdown;
    public Slider explorationRadiusSlider;

    [Header("Waypoint Controls")]
    public Button addWaypointButton;
    public Button clearWaypointsButton;
    public Button executeWaypointsButton;
    public Toggle showWaypointsToggle;

    [Header("Visualization")]
    public GameObject waypointMarkerPrefab;
    public LineRenderer pathRenderer;
    public List<Vector3> waypoints = new List<Vector3>();

    [Header("Status Display")]
    public Text statusText;
    public Text batteryLevelText;
    public Text connectionStatusText;

    private bool autonomousModeActive = false;
    private bool waypointsVisible = false;

    void Start()
    {
        InitializeControlPanel();
    }

    void InitializeControlPanel()
    {
        SetupTeleoperationControls();
        SetupAutonomousControls();
        SetupWaypointControls();
        UpdateStatusDisplay();
    }

    void SetupTeleoperationControls()
    {
        if (linearSpeedSlider != null)
        {
            linearSpeedSlider.minValue = 0.1f;
            linearSpeedSlider.maxValue = 5.0f;
            linearSpeedSlider.value = 1.0f;
        }

        if (angularSpeedSlider != null)
        {
            angularSpeedSlider.minValue = 0.1f;
            angularSpeedSlider.maxValue = 3.0f;
            angularSpeedSlider.value = 1.0f;
        }

        if (velocityModeToggle != null)
        {
            velocityModeToggle.isOn = true;
        }
    }

    void SetupAutonomousControls()
    {
        if (startAutonomousButton != null)
        {
            startAutonomousButton.onClick.AddListener(StartAutonomousMode);
        }

        if (stopAutonomousButton != null)
        {
            stopAutonomousButton.onClick.AddListener(StopAutonomousMode);
        }

        if (missionTypeDropdown != null)
        {
            missionTypeDropdown.options.Clear();
            missionTypeDropdown.AddOptions(new List<string>
            {
                "Patrol",
                "Exploration",
                "Mapping",
                "Following"
            });
            missionTypeDropdown.value = 0;
        }

        if (explorationRadiusSlider != null)
        {
            explorationRadiusSlider.minValue = 1.0f;
            explorationRadiusSlider.maxValue = 50.0f;
            explorationRadiusSlider.value = 10.0f;
        }
    }

    void SetupWaypointControls()
    {
        if (addWaypointButton != null)
        {
            addWaypointButton.onClick.AddListener(AddCurrentPositionAsWaypoint);
        }

        if (clearWaypointsButton != null)
        {
            clearWaypointsButton.onClick.AddListener(ClearAllWaypoints);
        }

        if (executeWaypointsButton != null)
        {
            executeWaypointsButton.onClick.AddListener(ExecuteWaypointMission);
        }

        if (showWaypointsToggle != null)
        {
            showWaypointsToggle.onValueChanged.AddListener(OnShowWaypointsChanged);
        }
    }

    public void ChangeControlMode(ControlMode newMode)
    {
        currentMode = newMode;

        // Disable all control groups
        SetTeleoperationControlsEnabled(false);
        SetAutonomousControlsEnabled(false);
        SetWaypointControlsEnabled(false);

        // Enable appropriate control group
        switch (newMode)
        {
            case ControlMode.Teleoperation:
                SetTeleoperationControlsEnabled(true);
                break;
            case ControlMode.Autonomous:
                SetAutonomousControlsEnabled(true);
                break;
            case ControlMode.Waypoint:
                SetWaypointControlsEnabled(true);
                break;
            case ControlMode.Simulation:
                // Simulation mode might disable direct robot control
                break;
        }

        UpdateStatusDisplay();
    }

    void SetTeleoperationControlsEnabled(bool enabled)
    {
        if (virtualJoystick != null) virtualJoystick.enabled = enabled;
        if (linearSpeedSlider != null) linearSpeedSlider.interactable = enabled;
        if (angularSpeedSlider != null) angularSpeedSlider.interactable = enabled;
        if (velocityModeToggle != null) velocityModeToggle.interactable = enabled;
        if (positionModeToggle != null) positionModeToggle.interactable = enabled;
    }

    void SetAutonomousControlsEnabled(bool enabled)
    {
        if (startAutonomousButton != null) startAutonomousButton.interactable = enabled;
        if (stopAutonomousButton != null) stopAutonomousButton.interactable = enabled;
        if (missionTypeDropdown != null) missionTypeDropdown.interactable = enabled;
        if (explorationRadiusSlider != null) explorationRadiusSlider.interactable = enabled;
    }

    void SetWaypointControlsEnabled(bool enabled)
    {
        if (addWaypointButton != null) addWaypointButton.interactable = enabled;
        if (clearWaypointsButton != null) clearWaypointsButton.interactable = enabled;
        if (executeWaypointsButton != null) executeWaypointsButton.interactable = enabled;
        if (showWaypointsToggle != null) showWaypointsToggle.interactable = enabled;
    }

    public void AddCurrentPositionAsWaypoint()
    {
        if (Camera.main != null)
        {
            // In a real scenario, this would get the robot's current position
            // For demo, we'll use the camera position as an example
            Vector3 waypoint = Camera.main.transform.position + Camera.main.transform.forward * 5f;
            waypoints.Add(waypoint);

            // Create visual marker
            if (waypointMarkerPrefab != null)
            {
                GameObject marker = Instantiate(waypointMarkerPrefab, waypoint, Quaternion.identity);
                marker.name = $"Waypoint_{waypoints.Count}";
            }

            UpdateWaypointVisualization();
        }
    }

    public void ClearAllWaypoints()
    {
        waypoints.Clear();

        // Remove all waypoint markers
        GameObject[] markers = GameObject.FindGameObjectsWithTag("WaypointMarker");
        foreach (GameObject marker in markers)
        {
            Destroy(marker);
        }

        if (pathRenderer != null)
        {
            pathRenderer.positionCount = 0;
        }
    }

    public void ExecuteWaypointMission()
    {
        if (waypoints.Count > 0)
        {
            // Send waypoints to ROS navigation system
            // This is a placeholder implementation
            Debug.Log($"Executing mission with {waypoints.Count} waypoints");

            // In real implementation, this would send waypoints to ROS navigation stack
            SendWaypointsToNavigation();
        }
        else
        {
            Debug.LogWarning("No waypoints to execute!");
        }
    }

    void UpdateWaypointVisualization()
    {
        if (pathRenderer != null && waypoints.Count > 1)
        {
            pathRenderer.positionCount = waypoints.Count;
            pathRenderer.SetPositions(waypoints.ToArray());
        }
    }

    void OnShowWaypointsChanged(bool show)
    {
        waypointsVisible = show;

        GameObject[] markers = GameObject.FindGameObjectsWithTag("WaypointMarker");
        foreach (GameObject marker in markers)
        {
            marker.SetActive(show);
        }

        if (pathRenderer != null)
        {
            pathRenderer.enabled = show && waypoints.Count > 1;
        }
    }

    void StartAutonomousMode()
    {
        autonomousModeActive = true;

        // Send autonomous mode command to ROS
        SendAutonomousCommand(true);

        UpdateStatusDisplay();
    }

    void StopAutonomousMode()
    {
        autonomousModeActive = false;

        // Send stop command to ROS
        SendAutonomousCommand(false);

        UpdateStatusDisplay();
    }

    void UpdateStatusDisplay()
    {
        if (statusText != null)
        {
            string modeText = currentMode.ToString();
            string autoText = autonomousModeActive ? " [AUTO]" : "";
            statusText.text = $"Mode: {modeText}{autoText}";
        }

        if (connectionStatusText != null)
        {
            // This would check actual ROS connection status
            connectionStatusText.text = "Connected: Yes"; // Placeholder
        }
    }

    // Placeholder methods for ROS communication
    void SendWaypointsToNavigation()
    {
        // Send waypoints to ROS navigation system
        // Implementation would convert waypoints to ROS Path message
    }

    void SendAutonomousCommand(bool start)
    {
        // Send autonomous mode command to ROS
        // Implementation would send appropriate ROS message
    }

    void Update()
    {
        // Update status display periodically
        if (batteryLevelText != null)
        {
            // In real implementation, this would come from robot's battery sensor
            // For demo, we'll simulate battery level
            float batteryLevel = Mathf.PerlinNoise(Time.time * 0.1f) * 100f;
            batteryLevelText.text = $"Battery: {batteryLevel:F1}%";
        }
    }
}
```

## ROS Integration for Robot Control

### 1. ROS Communication Setup

Set up ROS communication for sending control commands:

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;
using RosSharp.Messages.Geometry_msgs;
using RosSharp.Messages.Twist_msgs;
using RosSharp.Messages.Nav_msgs;
using System.Collections;

public class UnityRobotController : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeUrl = "ws://localhost:9090";
    public float connectionTimeout = 10f;

    [Header("Control Topics")]
    public string cmdVelTopic = "/cmd_vel";
    public string jointCmdTopic = "/joint_group_position_controller/command";
    public string navigationTopic = "/move_base_simple/goal";

    [Header("Robot Configuration")]
    public string robotName = "mobile_robot";
    public Transform robotTransform;

    private RosSocket rosSocket;
    private bool isConnected = false;
    private Coroutine connectionCoroutine;

    void Start()
    {
        ConnectToROS();
    }

    void ConnectToROS()
    {
        connectionCoroutine = StartCoroutine(AttemptConnection());
    }

    IEnumerator AttemptConnection()
    {
        float startTime = Time.time;

        while (!isConnected && Time.time - startTime < connectionTimeout)
        {
            try
            {
                rosSocket = new RosSocket(new WebSocketSharpClient(rosBridgeUrl));

                // Test connection by publishing a simple message
                rosSocket.Advertise<Twist>(cmdVelTopic);

                isConnected = true;
                Debug.Log("Successfully connected to ROS Bridge");

                yield break;
            }
            catch (System.Exception e)
            {
                Debug.LogWarning($"Connection attempt failed: {e.Message}");
                yield return new WaitForSeconds(2f);
            }
        }

        if (!isConnected)
        {
            Debug.LogError($"Failed to connect to ROS Bridge after {connectionTimeout} seconds");
        }
    }

    public void SendVelocityCommand(float linearX, float angularZ)
    {
        if (!isConnected) return;

        try
        {
            Twist twist = new Twist();
            twist.linear = new Vector3Msg(linearX, 0, 0);
            twist.angular = new Vector3Msg(0, 0, angularZ);

            rosSocket.Publish(cmdVelTopic, twist);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending velocity command: {e.Message}");
        }
    }

    public void SendJointCommands(Dictionary<string, float> jointPositions)
    {
        if (!isConnected) return;

        try
        {
            // For joint position control
            // This is a simplified example - actual implementation depends on your joint controller
            foreach (var joint in jointPositions)
            {
                // Send individual joint commands
                // Implementation would depend on your specific joint controller setup
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending joint commands: {e.Message}");
        }
    }

    public void SendNavigationGoal(Vector3 position, Vector3 orientation)
    {
        if (!isConnected) return;

        try
        {
            PoseStamped goal = new PoseStamped();
            goal.header.frame_id = "map";
            goal.header.stamp = new RosSharp.Messages.Standard_Msgs.Time();

            goal.pose.position = new PointMsg(position.x, position.y, position.z);
            goal.pose.orientation = new QuaternionMsg(orientation.x, orientation.y, orientation.z, 1.0f);

            rosSocket.Publish(navigationTopic, goal);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error sending navigation goal: {e.Message}");
        }
    }

    public void SendEmergencyStop()
    {
        if (!isConnected) return;

        // Send zero velocity to stop robot immediately
        SendVelocityCommand(0, 0);
    }

    void OnDestroy()
    {
        if (connectionCoroutine != null)
        {
            StopCoroutine(connectionCoroutine);
        }

        rosSocket?.Close();
    }
}
```

### 2. Control Mapping and Input Processing

Create a system for mapping Unity inputs to robot commands:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ControlMapper : MonoBehaviour
{
    [Header("Input Configuration")]
    public InputType inputType = InputType.Gamepad;
    public enum InputType { Gamepad, Keyboard, Touch, VR }

    [Header("Control Mapping")]
    public ControlMapping[] controlMappings;

    [Header("Deadzone Configuration")]
    public float joystickDeadzone = 0.2f;
    public float triggerDeadzone = 0.1f;

    [Header("Smoothing")]
    public bool enableSmoothing = true;
    public float smoothingFactor = 0.1f;

    private Dictionary<string, float> previousValues = new Dictionary<string, float>();
    private UnityRobotController robotController;

    [System.Serializable]
    public class ControlMapping
    {
        public string actionName;
        public InputAxis inputAxis;
        public RobotCommand robotCommand;
        public float scaleFactor = 1.0f;
        public float maxValue = 1.0f;
        public string unityInputName;  // For keyboard/gamepad input
    }

    public enum InputAxis { X, Y, Z, Pitch, Yaw, Roll }
    public enum RobotCommand { LinearVelocity, AngularVelocity, JointPosition, GripperControl }

    void Start()
    {
        robotController = GetComponent<UnityRobotController>();
        InitializePreviousValues();
    }

    void InitializePreviousValues()
    {
        foreach (var mapping in controlMappings)
        {
            previousValues[mapping.actionName] = 0f;
        }
    }

    void Update()
    {
        ProcessInput();
    }

    void ProcessInput()
    {
        foreach (var mapping in controlMappings)
        {
            float inputValue = GetInputValue(mapping);

            // Apply deadzone
            if (Mathf.Abs(inputValue) < GetDeadzoneForInput(mapping.inputAxis))
            {
                inputValue = 0f;
            }
            else
            {
                // Apply deadzone mapping
                inputValue = ApplyDeadzone(inputValue, GetDeadzoneForInput(mapping.inputAxis));
            }

            // Apply smoothing if enabled
            if (enableSmoothing)
            {
                inputValue = Mathf.Lerp(previousValues[mapping.actionName], inputValue, smoothingFactor);
                previousValues[mapping.actionName] = inputValue;
            }

            // Scale input
            inputValue *= mapping.scaleFactor;
            inputValue = Mathf.Clamp(inputValue, -mapping.maxValue, mapping.maxValue);

            // Execute robot command
            ExecuteRobotCommand(mapping.robotCommand, inputValue);
        }
    }

    float GetInputValue(ControlMapping mapping)
    {
        switch (inputType)
        {
            case InputType.Keyboard:
                return Input.GetAxis(mapping.unityInputName);
            case InputType.Gamepad:
                return Input.GetAxis(mapping.unityInputName);
            case InputType.Touch:
                return GetTouchInputValue(mapping);
            case InputType.VR:
                return GetVRInputValue(mapping);
            default:
                return 0f;
        }
    }

    float GetTouchInputValue(ControlMapping mapping)
    {
        // Implementation for touch-based input
        // This would typically involve virtual joysticks or touch gestures
        return 0f;
    }

    float GetVRInputValue(ControlMapping mapping)
    {
        // Implementation for VR controller input
        // This would involve VR controller axes and buttons
        return 0f;
    }

    float ApplyDeadzone(float value, float deadzone)
    {
        // Remap input from deadzone to full range
        if (value >= deadzone)
        {
            return Mathf.InverseLerp(deadzone, 1f, value);
        }
        else if (value <= -deadzone)
        {
            return Mathf.InverseLerp(-deadzone, -1f, value);
        }
        else
        {
            return 0f;
        }
    }

    float GetDeadzoneForInput(InputAxis axis)
    {
        switch (axis)
        {
            case InputAxis.X:
            case InputAxis.Y:
            case InputAxis.Z:
                return joystickDeadzone;
            default:
                return joystickDeadzone;
        }
    }

    void ExecuteRobotCommand(RobotCommand command, float value)
    {
        if (robotController == null) return;

        switch (command)
        {
            case RobotCommand.LinearVelocity:
                // Combine with angular velocity for differential drive
                // This would need to be coordinated with angular command
                break;
            case RobotCommand.AngularVelocity:
                // Combine with linear velocity for differential drive
                // This would need to be coordinated with linear command
                break;
            case RobotCommand.JointPosition:
                // Send joint position command
                break;
            case RobotCommand.GripperControl:
                // Control gripper or other end effector
                break;
        }
    }

    // Public methods for external control
    public void SetInputType(InputType newType)
    {
        inputType = newType;
    }

    public void UpdateControlMapping(string actionName, float newScaleFactor)
    {
        for (int i = 0; i < controlMappings.Length; i++)
        {
            if (controlMappings[i].actionName == actionName)
            {
                controlMappings[i].scaleFactor = newScaleFactor;
                break;
            }
        }
    }
}
```

## VR/AR Integration for Immersive Control

### 1. VR Teleoperation Interface

Create an immersive VR interface for robot control:

```csharp
#if UNITY_XR_MANAGEMENT_ENABLED
using UnityEngine.XR;
using UnityEngine.InputSystem;
#endif
using UnityEngine;

public class VRTeleoperationController : MonoBehaviour
{
    [Header("VR Device Configuration")]
    public XRNode leftControllerNode = XRNode.LeftHand;
    public XRNode rightControllerNode = XRNode.RightHand;

    [Header("Controller References")]
    public Transform leftControllerTransform;
    public Transform rightControllerTransform;

    [Header("Control Configuration")]
    public float teleportDistance = 5f;
    public float maxTeleportAngle = 30f;
    public bool useHeadDirection = true;

    [Header("Visualization")]
    public LineRenderer teleportArc;
    public GameObject teleportTargetIndicator;

    private InputDevice leftController;
    private InputDevice rightController;
    private bool teleportReady = false;

    void Start()
    {
        InitializeVRControllers();
    }

    void InitializeVRControllers()
    {
#if UNITY_XR_MANAGEMENT_ENABLED
        var devices = new List<InputDevice>();
        InputDevices.GetDevicesAtXRNode(leftControllerNode, devices);
        if (devices.Count > 0) leftController = devices[0];

        devices.Clear();
        InputDevices.GetDevicesAtXRNode(rightControllerNode, devices);
        if (devices.Count > 0) rightController = devices[0];
#endif
    }

    void Update()
    {
        UpdateControllerTransforms();
        HandleVRInput();
    }

    void UpdateControllerTransforms()
    {
#if UNITY_XR_MANAGEMENT_ENABLED
        if (leftController.isValid && leftControllerTransform != null)
        {
            leftControllerTransform.localPosition = leftController.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 position) ? position : Vector3.zero;
            leftControllerTransform.localRotation = leftController.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rotation) ? rotation : Quaternion.identity;
        }

        if (rightController.isValid && rightControllerTransform != null)
        {
            rightControllerTransform.localPosition = rightController.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 position) ? position : Vector3.zero;
            rightControllerTransform.localRotation = rightController.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion rotation) ? rotation : Quaternion.identity;
        }
#endif
    }

    void HandleVRInput()
    {
        // Handle teleportation with right controller
        if (rightController.isValid)
        {
            if (rightController.TryGetFeatureValue(CommonUsages.triggerButton, out bool triggerPressed))
            {
                if (triggerPressed)
                {
                    ShowTeleportArc();
                    teleportReady = true;
                }
                else if (teleportReady)
                {
                    // Execute teleportation
                    ExecuteTeleport();
                    teleportReady = false;
                    HideTeleportArc();
                }
            }
        }

        // Handle robot control with left controller
        if (leftController.isValid)
        {
            if (leftController.TryGetFeatureValue(CommonUsages.gripButton, out bool gripPressed))
            {
                if (gripPressed)
                {
                    // Send control command to robot
                    SendVRControlCommand();
                }
            }
        }
    }

    void ShowTeleportArc()
    {
        if (teleportArc == null || teleportTargetIndicator == null) return;

        // Calculate teleport arc based on controller direction
        Transform controllerTransform = rightControllerTransform;
        if (controllerTransform != null)
        {
            Vector3 startPos = controllerTransform.position;
            Vector3 direction = useHeadDirection ? Camera.main.transform.forward : controllerTransform.forward;

            // Limit angle
            float angle = Vector3.Angle(direction, Vector3.down);
            if (angle < (90 - maxTeleportAngle) || angle > (90 + maxTeleportAngle))
            {
                direction = Vector3.ProjectOnPlane(direction, Vector3.up).normalized;
            }

            // Calculate arc points
            Vector3 velocity = direction * teleportDistance * 0.5f;
            velocity.y += 5f; // Give some upward arc

            List<Vector3> arcPoints = CalculateArc(startPos, velocity, 20);

            teleportArc.positionCount = arcPoints.Count;
            teleportArc.SetPositions(arcPoints.ToArray());

            // Show target indicator at landing point
            Vector3 landingPoint = arcPoints[arcPoints.Count - 1];
            teleportTargetIndicator.SetActive(true);
            teleportTargetIndicator.transform.position = landingPoint;
        }
    }

    List<Vector3> CalculateArc(Vector3 startPos, Vector3 velocity, int segments)
    {
        List<Vector3> points = new List<Vector3>();
        Vector3 gravity = Physics.gravity;
        float timeStep = 0.1f;

        Vector3 currentPosition = startPos;
        Vector3 currentVelocity = velocity;

        for (int i = 0; i < segments; i++)
        {
            points.Add(currentPosition);

            // Apply gravity
            currentVelocity += gravity * timeStep;
            currentPosition += currentVelocity * timeStep;

            // Stop if we hit the ground
            if (currentPosition.y < 0)
            {
                points.Add(currentPosition);
                break;
            }
        }

        return points;
    }

    void ExecuteTeleport()
    {
        if (teleportTargetIndicator != null && teleportTargetIndicator.activeSelf)
        {
            // In VR teleoperation, this might teleport the user's view
            // Or send a navigation command to the robot
            Vector3 targetPosition = teleportTargetIndicator.transform.position;
            SendNavigationCommand(targetPosition);
        }
    }

    void HideTeleportArc()
    {
        if (teleportArc != null)
        {
            teleportArc.positionCount = 0;
        }
        if (teleportTargetIndicator != null)
        {
            teleportTargetIndicator.SetActive(false);
        }
    }

    void SendVRControlCommand()
    {
        // Send control command based on controller position/orientation
        if (leftControllerTransform != null)
        {
            // Example: Use controller as a "virtual joystick"
            Vector3 controllerForward = leftControllerTransform.forward;
            Vector3 robotForward = Vector3.ProjectOnPlane(controllerForward, Vector3.up).normalized;

            float linearVel = Vector3.Dot(robotForward, leftControllerTransform.forward);
            float angularVel = Vector3.Dot(leftControllerTransform.right, controllerForward);

            // Send to robot
            if (GetComponent<UnityRobotController>() != null)
            {
                GetComponent<UnityRobotController>().SendVelocityCommand(linearVel, angularVel);
            }
        }
    }

    void SendNavigationCommand(Vector3 target)
    {
        // Send navigation goal to robot
        if (GetComponent<UnityRobotController>() != null)
        {
            GetComponent<UnityRobotController>().SendNavigationGoal(target, Vector3.zero);
        }
    }
}
```

## Safety and Emergency Features

### 1. Emergency Stop System

Implement safety features for robot control:

```csharp
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Events;

public class SafetySystem : MonoBehaviour
{
    [Header("Emergency Controls")]
    public KeyCode emergencyStopKey = KeyCode.Escape;
    public Button emergencyStopButton;
    public GameObject emergencyStopOverlay;

    [Header("Safety Limits")]
    public float maxLinearVelocity = 2.0f;
    public float maxAngularVelocity = 1.0f;
    public float maxJointVelocity = 3.0f;
    public float maxOperatingTime = 3600f; // 1 hour

    [Header("Monitoring")]
    public float heartbeatInterval = 1.0f;
    public float controlTimeout = 5.0f;
    public bool enableCollisionDetection = true;

    [Header("Events")]
    public UnityEvent onEmergencyStop;
    public UnityEvent onSafetyViolation;
    public UnityEvent onSystemReset;

    private float lastHeartbeat = 0f;
    private float lastControlInput = 0f;
    private bool emergencyActive = false;
    private float operatingStartTime;

    void Start()
    {
        InitializeSafetySystem();
    }

    void InitializeSafetySystem()
    {
        operatingStartTime = Time.time;

        if (emergencyStopButton != null)
        {
            emergencyStopButton.onClick.AddListener(EmergencyStop);
        }

        lastHeartbeat = Time.time;
        lastControlInput = Time.time;
    }

    void Update()
    {
        CheckSafetyConditions();
        CheckTimeouts();
        CheckOperatingLimits();
    }

    void CheckSafetyConditions()
    {
        // Check for emergency stop key press
        if (Input.GetKeyDown(emergencyStopKey))
        {
            EmergencyStop();
        }

        // Check for heartbeat
        if (Time.time - lastHeartbeat > heartbeatInterval * 2)
        {
            Debug.LogWarning("Heartbeat timeout detected!");
            onSafetyViolation?.Invoke();
        }

        // Check for collision if enabled
        if (enableCollisionDetection)
        {
            CheckForCollisions();
        }
    }

    void CheckTimeouts()
    {
        // Check if control input has timed out
        if (Time.time - lastControlInput > controlTimeout)
        {
            // Send stop command to robot
            if (GetComponent<UnityRobotController>() != null)
            {
                GetComponent<UnityRobotController>().SendVelocityCommand(0, 0);
            }
        }
    }

    void CheckOperatingLimits()
    {
        // Check operating time limit
        if (Time.time - operatingStartTime > maxOperatingTime)
        {
            Debug.LogWarning("Maximum operating time exceeded!");
            EmergencyStop();
        }
    }

    void CheckForCollisions()
    {
        // Perform collision detection using Unity's physics
        Collider[] colliders = Physics.OverlapSphere(transform.position, 2f);

        foreach (Collider col in colliders)
        {
            if (col.gameObject.CompareTag("Obstacle") || col.gameObject.CompareTag("Wall"))
            {
                Debug.LogWarning("Potential collision detected!");

                // Send stop command to robot
                if (GetComponent<UnityRobotController>() != null)
                {
                    GetComponent<UnityRobotController>().SendVelocityCommand(0, 0);
                }

                onSafetyViolation?.Invoke();
                break;
            }
        }
    }

    public void EmergencyStop()
    {
        if (emergencyActive) return;

        emergencyActive = true;

        // Send emergency stop to robot
        if (GetComponent<UnityRobotController>() != null)
        {
            GetComponent<UnityRobotController>().SendEmergencyStop();
        }

        // Activate emergency overlay
        if (emergencyStopOverlay != null)
        {
            emergencyStopOverlay.SetActive(true);
        }

        // Trigger emergency stop event
        onEmergencyStop?.Invoke();

        Debug.LogWarning("EMERGENCY STOP ACTIVATED!");
    }

    public void ResetSystem()
    {
        if (!emergencyActive) return;

        emergencyActive = false;

        // Deactivate emergency overlay
        if (emergencyStopOverlay != null)
        {
            emergencyStopOverlay.SetActive(false);
        }

        // Reset operating time
        operatingStartTime = Time.time;

        // Trigger reset event
        onSystemReset?.Invoke();

        Debug.Log("Safety system reset.");
    }

    public void RegisterControlInput()
    {
        lastControlInput = Time.time;
        lastHeartbeat = Time.time;
    }

    public bool IsEmergencyActive()
    {
        return emergencyActive;
    }

    // Validation methods for control commands
    public float ValidateLinearVelocity(float velocity)
    {
        return Mathf.Clamp(velocity, -maxLinearVelocity, maxLinearVelocity);
    }

    public float ValidateAngularVelocity(float velocity)
    {
        return Mathf.Clamp(velocity, -maxAngularVelocity, maxAngularVelocity);
    }

    public float ValidateJointVelocity(float velocity)
    {
        return Mathf.Clamp(velocity, -maxJointVelocity, maxJointVelocity);
    }
}
```

## Best Practices for Unity Robot Control

### 1. Performance Optimization

```csharp
using UnityEngine;

public class OptimizedRobotControl : MonoBehaviour
{
    [Header("Performance Settings")]
    public int controlUpdateRate = 30; // Hz
    public int visualizationUpdateRate = 60; // Hz
    public bool enableLOD = true;

    private int controlUpdateCounter = 0;
    private int vizUpdateCounter = 0;

    void Update()
    {
        // Control updates at lower frequency for performance
        controlUpdateCounter++;
        if (controlUpdateCounter >= (60 / controlUpdateRate)) // Assuming 60 FPS
        {
            UpdateRobotControls();
            controlUpdateCounter = 0;
        }

        // Visualization updates at higher frequency for smoothness
        vizUpdateCounter++;
        if (vizUpdateCounter >= (60 / visualizationUpdateRate))
        {
            UpdateVisualizations();
            vizUpdateCounter = 0;
        }
    }

    void UpdateRobotControls()
    {
        // Process control inputs and send commands
        ProcessControlInputs();
    }

    void UpdateVisualizations()
    {
        // Update robot model visualization
        UpdateRobotVisualization();
    }

    void ProcessControlInputs()
    {
        // Implementation for processing control inputs
    }

    void UpdateRobotVisualization()
    {
        // Smooth robot movement visualization
        // This could include interpolation between received poses
    }
}
```

### 2. User Experience Considerations

- Provide haptic feedback for control actions
- Implement intuitive gesture-based controls
- Use consistent color coding for different robot states
- Provide clear visual feedback for all actions
- Include tutorial mode for new users
- Implement adaptive interfaces for different skill levels

## Learning Objectives

After completing this example, you should understand:
- How to create Unity-based interfaces for robot teleoperation
- Techniques for integrating ROS communication with Unity
- Best practices for designing intuitive control interfaces
- Safety considerations for remote robot operation
- VR/AR integration for immersive robot control experiences
- Performance optimization for real-time robot control

## Next Steps

Continue to learn about [Module 2 Learning Outcomes](../outcomes) to review what you've learned and how to apply it in real-world scenarios.