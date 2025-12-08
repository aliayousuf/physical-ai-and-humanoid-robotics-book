---
title: "Unity Scene Creation Tutorial"
description: "Creating Unity scenes for robot visualization and human-robot interaction"
---

# Unity Scene Creation Tutorial

## Overview

This tutorial guides you through creating Unity scenes specifically designed for robot visualization and human-robot interaction. We'll cover setting up the Unity environment, importing robot models, configuring cameras and lighting, and creating interactive interfaces for teleoperation and monitoring.

## Prerequisites

Before starting this tutorial, you should have:
- Unity Hub and Unity Editor installed (Unity 2021.3 LTS or later recommended)
- Basic understanding of Unity interface and concepts
- Robot models in FBX, OBJ, or other Unity-compatible formats
- Understanding of coordinate system differences between ROS and Unity
- ROS# package or similar for ROS communication (optional for basic scenes)

## Setting Up Unity Project for Robotics

### 1. Creating a New Project

1. Open Unity Hub and click "New Project"
2. Select the "3D (Built-in Render Pipeline)" template
3. Name your project (e.g., "RobotVisualization")
4. Choose a location to save the project
5. Click "Create Project"

### 2. Project Structure for Robotics

Organize your project with these recommended folders:
```
Assets/
├── Models/
│   ├── Robots/
│   ├── Environments/
│   └── Objects/
├── Scripts/
│   ├── ROS/
│   ├── Robotics/
│   └── UI/
├── Materials/
├── Scenes/
├── Prefabs/
└── Plugins/
```

### 3. Configure Project Settings

1. Go to Edit → Project Settings → Player
2. In Other Settings, set the following:
   - Product Name: "Robot Visualization System"
   - Company Name: Your organization
   - Set the target platform (PC, Mac & Linux Standalone for most robotics applications)

## Importing Robot Models

### 1. Preparing Robot Models

Before importing, ensure your robot models are properly formatted:
- Correct scale (typically 1 unit = 1 meter in Unity)
- Proper coordinate system (convert from ROS if needed)
- Separate files for each link if needed for articulation
- Proper joint information preserved

### 2. Importing Models into Unity

1. Create a folder in Assets/Models/Robots/ for your robot
2. Drag and drop your model files (FBX, OBJ, etc.) into this folder
3. Unity will automatically import and create assets

### 3. Configuring Imported Models

After importing, configure your robot model:

#### Model Import Settings
- In the Inspector, select your imported model
- Under Model tab:
  - Scale Factor: Usually 1.0 (adjust if model is not in meters)
  - Mesh Compression: Disabled for precise robotics models
  - Read/Write enabled: Checked for dynamic manipulation
  - Optimize Mesh: Checked for better performance

#### Rig Configuration
- Animation Type: "Generic" or "Humanoid" (Generic for robots)
- If using articulation, ensure "Import Animation" is checked
- Configure avatar if needed for animation

## Creating Robot Prefabs

### 1. Setting Up Articulation Bodies

For realistic joint simulation in Unity, use Articulation Bodies:

```csharp
using UnityEngine;

public class RobotLink : MonoBehaviour
{
    public ArticulationBody body;
    public ArticulationJointType jointType = ArticulationJointType.RevoluteJoint;
    public float minAngle = -90f;
    public float maxAngle = 90f;
    public float stiffness = 1000f;
    public float damping = 100f;

    void Start()
    {
        SetupArticulationBody();
    }

    void SetupArticulationBody()
    {
        body = GetComponent<ArticulationBody>();
        body.jointType = jointType;

        if (jointType == ArticulationJointType.RevoluteJoint)
        {
            var drive = body.xDrive;
            drive.lowerLimit = minAngle;
            drive.upperLimit = maxAngle;
            drive.stiffness = stiffness;
            drive.damping = damping;
            drive.forceLimit = 1000f;
            body.xDrive = drive;
        }
    }
}
```

### 2. Creating Robot Prefab

1. Drag your robot model hierarchy from the Hierarchy to the Project window to create a prefab
2. Add necessary components:
   - Articulation Bodies for joints
   - Colliders for collision detection
   - Rigidbodies if needed
   - Custom scripts for robot control

## Environment Setup

### 1. Creating Basic Environment

Create a basic environment for your robot:

```csharp
using UnityEngine;

public class EnvironmentSetup : MonoBehaviour
{
    public GameObject floor;
    public GameObject[] walls;
    public GameObject ceiling;

    [Header("Environment Dimensions")]
    public Vector3 center = Vector3.zero;
    public Vector3 size = new Vector3(10f, 10f, 5f);

    void Start()
    {
        CreateEnvironment();
    }

    void CreateEnvironment()
    {
        // Create floor
        if (floor != null)
        {
            floor.transform.position = center - Vector3.up * size.z / 2f;
            floor.transform.localScale = new Vector3(size.x, 1f, size.y);
        }

        // Create walls
        if (walls != null && walls.Length >= 4)
        {
            // Left wall
            walls[0].transform.position = center + Vector3.left * size.x / 2f;
            walls[0].transform.localScale = new Vector3(1f, size.y, size.z);

            // Right wall
            walls[1].transform.position = center + Vector3.right * size.x / 2f;
            walls[1].transform.localScale = new Vector3(1f, size.y, size.z);

            // Front wall
            walls[2].transform.position = center + Vector3.forward * size.y / 2f;
            walls[2].transform.localScale = new Vector3(size.x, size.y, 1f);

            // Back wall
            walls[3].transform.position = center + Vector3.back * size.y / 2f;
            walls[3].transform.localScale = new Vector3(size.x, size.y, 1f);
        }
    }
}
```

### 2. Lighting Setup

Configure lighting for optimal visualization:

1. Remove the default Directional Light
2. Add a new Directional Light for main lighting:
   - Position: (0, 10, -10)
   - Rotation: (30, 0, 0)
   - Color: White (1, 1, 1)
   - Intensity: 1.0
3. Add additional lights if needed for specific areas

## Camera Configuration

### 1. Multiple Camera Setup

Set up different camera views for comprehensive visualization:

```csharp
using UnityEngine;

public class CameraManager : MonoBehaviour
{
    [Header("Camera Types")]
    public Camera mainCamera;
    public Camera robotCamera;  // First-person view from robot
    public Camera topDownCamera;  // Overhead view
    public Camera followCamera;  // Follows robot

    [Header("Camera Settings")]
    public float followDistance = 10f;
    public float followHeight = 5f;
    public float rotationSpeed = 2f;

    void Start()
    {
        SetupCameras();
    }

    void SetupCameras()
    {
        // Configure main camera
        if (mainCamera != null)
        {
            mainCamera.fieldOfView = 60f;
            mainCamera.nearClipPlane = 0.1f;
            mainCamera.farClipPlane = 1000f;
        }

        // Configure robot camera (if robot has a head)
        if (robotCamera != null)
        {
            robotCamera.fieldOfView = 90f;
        }

        // Configure top-down camera
        if (topDownCamera != null)
        {
            topDownCamera.transform.position = new Vector3(0, 20, 0);
            topDownCamera.transform.rotation = Quaternion.Euler(90, 0, 0);
        }
    }

    void Update()
    {
        UpdateFollowCamera();
    }

    void UpdateFollowCamera()
    {
        if (followCamera != null)
        {
            // Simple follow implementation
            // In practice, you'd track a specific robot object
        }
    }
}
```

### 2. Camera Switching Interface

Create an interface to switch between cameras:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class CameraSwitcher : MonoBehaviour
{
    public Camera[] cameras;
    public int currentCameraIndex = 0;
    public Text cameraNameDisplay;

    void Start()
    {
        if (cameras.Length > 0)
        {
            ActivateCamera(0);
        }
    }

    void Update()
    {
        // Switch cameras with number keys
        for (int i = 0; i < cameras.Length; i++)
        {
            if (Input.GetKeyDown(KeyCode.Alpha1 + i))
            {
                ActivateCamera(i);
            }
        }
    }

    public void ActivateCamera(int index)
    {
        if (index >= 0 && index < cameras.Length)
        {
            foreach (var cam in cameras)
            {
                cam.gameObject.SetActive(false);
            }

            cameras[index].gameObject.SetActive(true);
            currentCameraIndex = index;

            if (cameraNameDisplay != null)
            {
                cameraNameDisplay.text = cameras[index].name;
            }
        }
    }
}
```

## UI and Control Interface

### 1. Creating Basic UI Canvas

Set up a canvas for robot controls and information display:

```csharp
using UnityEngine;
using UnityEngine.UI;

public class RobotUIController : MonoBehaviour
{
    [Header("UI Elements")]
    public Text robotStatusText;
    public Text jointInfoText;
    public Text positionText;
    public Slider[] jointSliders;
    public Button[] controlButtons;

    [Header("Robot Reference")]
    public GameObject robot;

    void Start()
    {
        InitializeUI();
    }

    void InitializeUI()
    {
        // Initialize joint sliders if they exist
        if (jointSliders != null)
        {
            for (int i = 0; i < jointSliders.Length; i++)
            {
                int index = i; // Capture for closure
                jointSliders[i].onValueChanged.AddListener((value) =>
                    OnJointSliderChanged(index, value));
            }
        }

        // Initialize control buttons
        if (controlButtons != null)
        {
            controlButtons[0].onClick.AddListener(() => MoveRobotForward());
            controlButtons[1].onClick.AddListener(() => MoveRobotBackward());
            controlButtons[2].onClick.AddListener(() => StopRobot());
        }
    }

    void Update()
    {
        UpdateRobotInfo();
    }

    void UpdateRobotInfo()
    {
        if (robot != null && positionText != null)
        {
            positionText.text = $"Position: {robot.transform.position}";
        }
    }

    void OnJointSliderChanged(int jointIndex, float value)
    {
        // Handle joint movement based on slider value
        // This would typically communicate with the robot's joint controllers
    }

    void MoveRobotForward()
    {
        if (robot != null)
        {
            robot.transform.Translate(Vector3.forward * Time.deltaTime * 2f);
        }
    }

    void MoveRobotBackward()
    {
        if (robot != null)
        {
            robot.transform.Translate(Vector3.back * Time.deltaTime * 2f);
        }
    }

    void StopRobot()
    {
        // Implementation depends on your robot control system
    }
}
```

### 2. Sensor Visualization UI

Create UI elements to display sensor data:

```csharp
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

public class SensorVisualizer : MonoBehaviour
{
    [Header("Sensor UI Elements")]
    public RawImage cameraFeedDisplay;
    public Text laserScanText;
    public Text imuDataText;
    public Text[] lidarBeams;

    [Header("Sensor Data")]
    public float[] laserScanData;
    public Vector3 imuOrientation;
    public Texture2D cameraImage;

    void Update()
    {
        UpdateSensorDisplays();
    }

    void UpdateSensorDisplays()
    {
        // Update camera feed if available
        if (cameraImage != null && cameraFeedDisplay != null)
        {
            cameraFeedDisplay.texture = cameraImage;
        }

        // Update laser scan data
        if (laserScanText != null && laserScanData != null)
        {
            laserScanText.text = $"Laser Points: {laserScanData.Length}";
        }

        // Update IMU data
        if (imuDataText != null)
        {
            imuDataText.text = $"Orientation: {imuOrientation}";
        }
    }

    // Methods to update sensor data from external sources
    public void UpdateCameraFeed(Texture2D newImage)
    {
        cameraImage = newImage;
    }

    public void UpdateLaserScan(float[] scanData)
    {
        laserScanData = scanData;
    }

    public void UpdateIMUData(Vector3 orientation)
    {
        imuOrientation = orientation;
    }
}
```

## ROS Integration Setup

### 1. Installing ROS# Package

To connect Unity with ROS, install the ROS# package:

1. Download ROS# from the Unity Asset Store or GitHub
2. Import into your project
3. Set up WebSocket connection to ROS bridge

### 2. Basic ROS Connection Script

```csharp
using UnityEngine;
using RosSharp.RosBridgeClient;

public class RosConnector : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosBridgeServerUrl = "ws://localhost:9090";

    private RosSocket rosSocket;

    void Start()
    {
        ConnectToRos();
    }

    void ConnectToRos()
    {
        try
        {
            rosSocket = new RosSocket(new WebSocketSharpClient(rosBridgeServerUrl));
            Debug.Log("Connected to ROS Bridge");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Failed to connect to ROS: {e.Message}");
        }
    }

    void OnDestroy()
    {
        rosSocket?.Close();
    }

    // Example method to publish joint states
    public void PublishJointStates(float[] positions)
    {
        if (rosSocket != null)
        {
            // Implementation would go here
        }
    }

    // Example method to subscribe to odometry
    public void SubscribeToOdometry()
    {
        if (rosSocket != null)
        {
            rosSocket.Subscribe<Odometry>("/odom", ReceiveOdometry);
        }
    }

    void ReceiveOdometry(Odometry odom)
    {
        // Update robot position in Unity based on ROS odometry
        Vector3 rosPosition = new Vector3(
            (float)odom.pose.pose.position.x,
            (float)odom.pose.pose.position.z,  // Unity Y = ROS Z
            -(float)odom.pose.pose.position.y  // Unity Z = -ROS Y
        );

        transform.position = rosPosition;
    }
}
```

## Scene Organization Best Practices

### 1. Layer Management

Organize your scene with appropriate layers:

- Default: Standard objects
- Robot: Robot components
- Environment: Static environment objects
- UI: User interface elements
- Sensors: Sensor visualization objects

### 2. Tag Management

Use tags for easy object identification:

- PlayerRobot: Main robot object
- StaticObstacle: Non-moving obstacles
- DynamicObstacle: Moving obstacles
- Waypoint: Navigation waypoints
- SpawnPoint: Robot spawn locations

### 3. Scene Hierarchy Structure

Organize your hierarchy logically:

```
Main Camera
Directional Light
Environment/
├── Floor
├── Walls
└── Obstacles
Robot/
├── Base
├── Joint1
├── Joint2
└── EndEffector
UI/
├── Canvas
│   ├── RobotStatus
│   ├── JointControls
│   └── CameraSwitcher
└── SensorDisplays
```

## Performance Optimization

### 1. Level of Detail (LOD)

Implement LOD for complex robot models:

```csharp
using UnityEngine;

[RequireComponent(typeof(LODGroup))]
public class RobotLODController : MonoBehaviour
{
    public LODGroup lodGroup;
    public float[] lodDistances = { 10f, 30f, 60f };

    void Start()
    {
        SetupLOD();
    }

    void SetupLOD()
    {
        lodGroup = GetComponent<LODGroup>();

        LOD[] lods = new LOD[3];

        // LOD 0: High detail (up to 10m)
        lods[0] = new LOD(0.75f, GetRenderersForLOD(0));

        // LOD 1: Medium detail (up to 30m)
        lods[1] = new LOD(0.25f, GetRenderersForLOD(1));

        // LOD 2: Low detail (beyond 30m)
        lods[2] = new LOD(0.05f, GetRenderersForLOD(2));

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    Renderer[] GetRenderersForLOD(int lodLevel)
    {
        // Return appropriate renderers for each LOD level
        // Implementation depends on your specific model structure
        return new Renderer[0];
    }
}
```

### 2. Occlusion Culling

Enable occlusion culling in your scene:
1. Go to Window → Rendering → Occlusion Culling
2. Mark static objects as "Static" in their Inspector
3. Bake occlusion data

## Building and Deployment

### 1. Build Settings

1. Go to File → Build Settings
2. Select your target platform (Windows, macOS, Linux)
3. Add your scene to the build
4. Configure build options:
   - Development Build: For debugging
   - Autoconnect Profiler: For performance analysis
   - Script Debugging: For debugging scripts

### 2. Performance Considerations

- Optimize textures for your target platform
- Use appropriate mesh quality
- Limit the number of real-time shadows
- Consider using occlusion culling for complex scenes

## Testing Your Unity Scene

### 1. Basic Functionality Testing

- Verify robot model loads correctly
- Test camera switching functionality
- Check UI elements respond properly
- Validate sensor visualization (if applicable)

### 2. Integration Testing

- If connected to ROS, verify data flows correctly
- Test robot movement and joint control
- Validate collision detection
- Check performance metrics

## Troubleshooting Common Issues

### 1. Model Import Issues
- **Wrong scale**: Check import scale factor in model settings
- **Incorrect orientation**: Verify coordinate system conversion
- **Missing textures**: Ensure texture files are in the same folder

### 2. Performance Issues
- **Low frame rate**: Reduce model complexity or enable LOD
- **Memory issues**: Optimize textures and reduce polygon count
- **Physics instability**: Adjust physics timestep and solver settings

### 3. ROS Connection Issues
- **Connection fails**: Verify ROS bridge is running on correct port
- **No data received**: Check topic names and message types
- **Delayed updates**: Optimize network settings and update rates

## Advanced Features

### 1. VR Integration

For immersive robot teleoperation:
- Install XR Interaction Toolkit
- Set up VR camera rig
- Configure controller inputs
- Implement VR-specific UI

### 2. Multi-Robot Visualization

To visualize multiple robots:
- Create robot prefab with unique identifiers
- Use object pooling for performance
- Implement separate control systems for each robot
- Add color coding for identification

## Learning Objectives

After completing this tutorial, you should be able to:
- Set up a Unity project for robotics visualization
- Import and configure robot models with proper articulation
- Create multiple camera views for comprehensive visualization
- Implement basic UI for robot control and monitoring
- Optimize scenes for performance
- Troubleshoot common Unity-robotics integration issues

## Next Steps

Continue to learn about [Sensor Integration](../workflows/sensor-integration) to understand how to properly integrate and visualize sensor data from your robot simulation.