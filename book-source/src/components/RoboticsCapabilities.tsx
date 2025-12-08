import React, { useState } from 'react';

const RoboticsCapabilities = () => {
  const [activeCapability, setActiveCapability] = useState(0);

  const capabilities = [
    {
      title: "Perception",
      description: "Robots use sensors like cameras, LIDAR, and IMUs to understand their environment through computer vision, SLAM, and sensor fusion.",
      icon: "👁️"
    },
    {
      title: "Navigation",
      description: "Path planning and obstacle avoidance algorithms allow robots to move efficiently through complex environments using systems like ROS 2 Navigation2.",
      icon: "🧭"
    },
    {
      title: "Manipulation",
      description: "Robotic arms and end-effectors perform precise manipulation tasks using inverse kinematics, grasp planning, and control algorithms.",
      icon: "🦾"
    },
    {
      title: "Learning",
      description: "AI and machine learning enable robots to adapt, improve performance, and acquire new skills through reinforcement learning and imitation learning.",
      icon: "🧠"
    }
  ];

  return (
    <div className="robotics-capabilities" role="region" aria-labelledby="capabilities-heading">
      <h2 id="capabilities-heading">Core Robotics Capabilities</h2>
      <div className="capabilities-container">
        <div className="capability-selector" role="tablist" aria-label="Robotics capabilities tabs">
          {capabilities.map((capability, index) => (
            <button
              key={index}
              className={`capability-btn ${index === activeCapability ? 'active' : ''}`}
              onClick={() => setActiveCapability(index)}
              role="tab"
              aria-selected={index === activeCapability}
              aria-controls={`capability-panel-${index}`}
              id={`capability-tab-${index}`}
            >
              <span className="capability-icon">{capability.icon}</span>
              <span className="capability-title">{capability.title}</span>
            </button>
          ))}
        </div>

        <div
          className="capability-content"
          role="tabpanel"
          id={`capability-panel-${activeCapability}`}
          aria-labelledby={`capability-tab-${activeCapability}`}
        >
          <h3>{capabilities[activeCapability].title}</h3>
          <p>{capabilities[activeCapability].description}</p>
          <div className="capability-demo">
            <div className="demo-placeholder" aria-label="Capability demonstration area">
              {capabilities[activeCapability].icon} Demo visualization would appear here
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoboticsCapabilities;