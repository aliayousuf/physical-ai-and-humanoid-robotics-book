import React, { useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

// Simple humanoid robot model
const HumanoidRobot = () => {
  const groupRef = useRef();

  // Add subtle animation
  useFrame((state) => {
    if (groupRef.current) {
      // Gentle floating motion
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime) * 0.1;
      // Gentle rotation
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {/* Head */}
      <mesh position={[0, 1.5, 0]}>
        <boxGeometry args={[0.5, 0.5, 0.5]} />
        <meshStandardMaterial color="#666" />
      </mesh>

      {/* Eyes */}
      <mesh position={[-0.15, 1.6, 0.25]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial color="#ff6b6b" emissive="#ff6b6b" emissiveIntensity={0.5} />
      </mesh>
      <mesh position={[0.15, 1.6, 0.25]}>
        <sphereGeometry args={[0.05, 16, 16]} />
        <meshStandardMaterial color="#ff6b6b" emissive="#ff6b6b" emissiveIntensity={0.5} />
      </mesh>

      {/* Body */}
      <mesh position={[0, 0.8, 0]}>
        <boxGeometry args={[0.8, 1, 0.4]} />
        <meshStandardMaterial color="#777" />
      </mesh>

      {/* Left Arm */}
      <mesh position={[-0.6, 0.8, 0]} rotation={[0, 0, -0.3]}>
        <cylinderGeometry args={[0.1, 0.1, 0.8, 8]} />
        <meshStandardMaterial color="#888" />
      </mesh>

      {/* Right Arm */}
      <mesh position={[0.6, 0.8, 0]} rotation={[0, 0, 0.3]}>
        <cylinderGeometry args={[0.1, 0.1, 0.8, 8]} />
        <meshStandardMaterial color="#888" />
      </mesh>

      {/* Left Leg */}
      <mesh position={[-0.2, -0.2, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 0.8, 8]} />
        <meshStandardMaterial color="#999" />
      </mesh>

      {/* Right Leg */}
      <mesh position={[0.2, -0.2, 0]}>
        <cylinderGeometry args={[0.12, 0.12, 0.8, 8]} />
        <meshStandardMaterial color="#999" />
      </mesh>
    </group>
  );
};

const RobotScene = () => {
  return (
    <Canvas shadows aria-label="3D Humanoid Robot Visualization" role="img">
      <PerspectiveCamera makeDefault position={[3, 2, 5]} />
      <ambientLight intensity={0.5} />
      <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
      <pointLight position={[-10, -10, -10]} intensity={0.5} />
      <HumanoidRobot />
      <OrbitControls
        enableZoom={true}
        enablePan={false}
        minDistance={3}
        maxDistance={10}
        aria-label="Orbit controls for 3D robot visualization"
      />
      <axesHelper args={[5]} visible={false} /> {/* For accessibility - helps with spatial understanding */}
    </Canvas>
  );
};

export default RobotScene;