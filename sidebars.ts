import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  // By default, Docusaurus generates a sidebar from the docs folder structure
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics',
      collapsible: true,
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Module 1: The Robotic Nervous System (ROS 2)',
          collapsible: true,
          collapsed: false,
          items: [
            'module-1-ros2/index',
            {
              type: 'category',
              label: 'Concepts',
              items: [
                'module-1-ros2/concepts/ros2-middleware',
                'module-1-ros2/concepts/nodes-topics-services',
                'module-1-ros2/concepts/actions',
                'module-1-ros2/concepts/urdf'
              ]
            },
            {
              type: 'category',
              label: 'Workflows',
              items: [
                'module-1-ros2/workflows/creating-first-package',
                'module-1-ros2/workflows/launch-files',
                'module-1-ros2/workflows/rclpy-integration'
              ]
            },
            {
              type: 'category',
              label: 'Examples',
              items: [
                'module-1-ros2/examples/basic-ros2-node',
                'module-1-ros2/examples/urdf-robot-definition',
                'module-1-ros2/examples/python-control-loop'
              ]
            },
            'module-1-ros2/outcomes'
          ]
        },
        {
          type: 'category',
          label: 'Module 2: The Digital Twin (Gazebo & Unity)',
          collapsible: true,
          collapsed: false,
          items: [
            'module-2-digital-twin/index',
            {
              type: 'category',
              label: 'Concepts',
              items: [
                'module-2-digital-twin/concepts/gazebo-physics',
                'module-2-digital-twin/concepts/urdf-sdf-simulation',
                'module-2-digital-twin/concepts/sensor-simulation',
                'module-2-digital-twin/concepts/unity-visualization'
              ]
            },
            {
              type: 'category',
              label: 'Workflows',
              items: [
                'module-2-digital-twin/workflows/gazebo-environment',
                'module-2-digital-twin/workflows/unity-scenes',
                'module-2-digital-twin/workflows/sensor-integration'
              ]
            },
            {
              type: 'category',
              label: 'Examples',
              items: [
                'module-2-digital-twin/examples/humanoid-simulation',
                'module-2-digital-twin/examples/sensor-data-visualization',
                'module-2-digital-twin/examples/unity-robot-control'
              ]
            },
            'module-2-digital-twin/outcomes'
          ]
        },
        {
          type: 'category',
          label: 'Module 3: NVIDIA Isaac (Simulation & AI)',
          collapsible: true,
          collapsed: false,
          items: [
            'module-3-nvidia-isaac/index',
            {
              type: 'category',
              label: 'Concepts',
              items: [
                'module-3-nvidia-isaac/concepts/isaac-sim',
                'module-3-nvidia-isaac/concepts/isaac-ros',
                'module-3-nvidia-isaac/concepts/nav2-path-planning',
                'module-3-nvidia-isaac/concepts/rl-sim-to-real'
              ]
            },
            {
              type: 'category',
              label: 'Workflows',
              items: [
                'module-3-nvidia-isaac/workflows/perception-pipeline',
                'module-3-nvidia-isaac/workflows/navigation-setup',
                'module-3-nvidia-isaac/workflows/rl-training'
              ]
            },
            {
              type: 'category',
              label: 'Examples',
              items: [
                'module-3-nvidia-isaac/examples/vslam-implementation',
                'module-3-nvidia-isaac/examples/path-planning-demo',
                'module-3-nvidia-isaac/examples/rl-walk-cycle'
              ]
            },
            'module-3-nvidia-isaac/outcomes'
          ]
        },
        {
          type: 'category',
          label: 'Module 4: Vision-Language-Action (VLA) Models',
          collapsible: true,
          collapsed: false,
          items: [
            'module-4-vla/index',
            {
              type: 'category',
              label: 'Concepts',
              items: [
                'module-4-vla/concepts/llm-cognitive-planning',
                'module-4-vla/concepts/multi-modal-fusion',
                'module-4-vla/concepts/voice-command-processing',
                'module-4-vla/concepts/action-sequence-generation'
              ]
            },
            {
              type: 'category',
              label: 'Workflows',
              items: [
                'module-4-vla/workflows/vla-integration',
                'module-4-vla/workflows/ai-planning-workflow',
                'module-4-vla/workflows/capstone-implementation'
              ]
            },
            {
              type: 'category',
              label: 'Examples',
              items: [
                'module-4-vla/examples/voice-to-action',
                'module-4-vla/examples/object-detection-integration',
                'module-4-vla/examples/manipulation-control'
              ]
            },
            'module-4-vla/outcomes',
            'module-4-vla/capstone'
          ]
        }
      ]
    },
    {
      type: 'category',
      label: 'Resources',
      collapsible: true,
      collapsed: false,
      items: [
        'glossary',
        'references',
        'prerequisites',
        'accessibility-statement',
        'performance-optimization',
        'cross-browser-compatibility',
        'quality-assurance'
      ]
    }
  ]

  // But you can create a sidebar manually
  /*
  tutorialSidebar: [
    'intro',
    'hello',
    {
      type: 'category',
      label: 'Tutorial',
      items: ['tutorial-basics/create-a-document'],
    },
  ],
   */
};

export default sidebars;
