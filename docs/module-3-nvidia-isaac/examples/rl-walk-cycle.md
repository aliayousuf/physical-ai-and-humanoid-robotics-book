---
title: "Reinforcement Learning Walk Cycle"
description: "Training humanoid walking patterns using reinforcement learning"
---

# Reinforcement Learning Walk Cycle

## Overview

This example demonstrates how to train humanoid walking patterns using reinforcement learning with Isaac Sim and Isaac Gym. Reinforcement learning offers a powerful approach to develop robust walking controllers that can adapt to different terrains and maintain balance in challenging conditions.

## Prerequisites

Before implementing RL-based walk cycles, ensure you have:
- Completed Module 1 (ROS 2 fundamentals)
- Completed Module 2 (Digital Twin simulation)
- Understanding of basic machine learning concepts
- Isaac Sim with GPU acceleration configured
- Isaac Gym for parallel RL training
- Experience with neural networks and optimization

## RL for Humanoid Locomotion

### Why RL for Walking?

Traditional approaches to humanoid walking rely on complex analytical models and predefined patterns. RL offers several advantages:

- **Adaptive behavior**: Learns to adapt to different terrains and conditions
- **Robustness**: Can recover from perturbations and disturbances
- **Optimization**: Automatically optimizes for desired criteria (speed, stability, energy efficiency)
- **Generalization**: Can generalize to unseen situations after training

### Challenges in Humanoid Locomotion RL

- **High-dimensional action space**: Many joints to control simultaneously
- **Balance requirements**: Must maintain stability during movement
- **Contact dynamics**: Complex physics during foot-ground contact
- **Sparse rewards**: Difficulty in defining appropriate reward functions
- **Sim-to-real transfer**: Gap between simulation and real robot

## Isaac Gym for Parallel Training

Isaac Gym enables thousands of parallel environments for efficient RL training:

```python
# humanoid_locomotion_env.py
import numpy as np
import torch
import omni
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.core.tasks import RLTask
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf, UsdGeom
import carb

class HumanoidLocomotionTask(RLTask):
    def __init__(
        self,
        name: str,
        env: VecEnvBase,
        offset=None
    ):
        RLTask.__init__(self, name=name, offset=offset)

        # Environment parameters
        self._num_envs = 2048  # Number of parallel environments
        self._env_spacing = 3.0
        self._action_space = None

        # Robot parameters
        self._num_dof = 24  # Number of degrees of freedom
        self._num_actions = self._num_dof  # Joint position targets
        self._num_observations = 48  # Joint positions, velocities, etc.

        # Reward parameters
        self._rewards = {
            'linear_velocity_tracking': 1.0,
            'action_rate_penalty': -0.0001,
            'joint_deviation_penalty': -0.001,
            'upright_bonus': 0.5,
            'energy_efficiency': -0.0001
        }

        # Episode parameters
        self._max_episode_length = 1000  # 1000 steps per episode
        self._episode_current_step = torch.zeros(self._num_envs)

        # Robot-specific parameters
        self._base_init_pos = torch.tensor([0.0, 0.0, 1.0])
        self._target_velocity = 1.0  # Target forward velocity (m/s)

    def set_agents(self, num_envs):
        """Set up parallel agents for training"""
        self._num_envs = num_envs

    def get_observations(self):
        """Get observations from all environments"""
        # Get joint positions and velocities
        joint_pos = self._articulation_views['humanoid'].get_joint_positions()
        joint_vel = self._articulation_views['humanoid'].get_joint_velocities()

        # Get base position and orientation
        base_pos, base_quat = self._articulation_views['humanoid'].get_world_poses()

        # Get base linear and angular velocities
        base_lin_vel, base_ang_vel = self._articulation_views['humanoid'].get_velocities()

        # Calculate observations
        obs = torch.cat([
            joint_pos,          # Joint positions
            joint_vel,          # Joint velocities
            base_pos,           # Base position
            base_quat,          # Base orientation
            base_lin_vel,       # Base linear velocity
            base_ang_vel,       # Base angular velocity
            self._get_targets() # Target information
        ], dim=-1)

        return {self.name: {"obs": obs}}

    def calculate_metrics(self):
        """Calculate episode metrics for logging"""
        # Calculate various performance metrics
        metrics = {
            'average_speed': self._calculate_average_speed(),
            'balance_stability': self._calculate_balance_stability(),
            'energy_efficiency': self._calculate_energy_efficiency(),
            'task_completion': self._calculate_task_completion(),
            'episode_lengths': self._episode_current_step.clone()
        }

        return metrics

    def is_done(self):
        """Check if episodes are done"""
        # Check termination conditions for each environment
        dones = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        resets = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # Terminate if robot falls (base position too low)
        base_pos, _ = self._articulation_views['humanoid'].get_world_poses()
        fall_threshold = 0.5  # If base drops below 0.5m, consider fallen
        falls = base_pos[:, 2] < fall_threshold
        dones |= falls
        resets |= falls

        # Reset if episode reaches maximum length
        time_outs = self._episode_current_step >= self._max_episode_length
        resets |= time_outs

        # Update episode step counters
        self._episode_current_step += 1
        self._episode_current_step = torch.where(resets, torch.zeros_like(self._episode_current_step), self._episode_current_step)

        return dones, resets

    def pre_physics_step(self, actions):
        """Apply actions to robots before physics step"""
        # Convert actions to joint position targets
        joint_targets = actions.clone()

        # Add noise for exploration (could be reduced over time)
        noise_scale = 0.01
        noise = torch.randn_like(joint_targets) * noise_scale
        joint_targets += noise

        # Apply joint targets to all robots
        self._articulation_views['humanoid'].set_joint_position_targets(joint_targets)

    def _calculate_average_speed(self):
        """Calculate average forward speed across environments"""
        base_lin_vel, _ = self._articulation_views['humanoid'].get_velocities()
        forward_vel = base_lin_vel[:, 0]  # X-component is forward
        return torch.mean(forward_vel).item()

    def _calculate_balance_stability(self):
        """Calculate balance stability metrics"""
        _, base_quat = self._articulation_views['humanoid'].get_world_poses()

        # Calculate upright orientation (z-axis should be up)
        z_up = torch.tensor([0, 0, 1], dtype=torch.float, device=self._device)

        # Rotate z_up by robot's orientation
        base_rot = torch_utils.quat_rotate(base_quat, z_up)

        # Measure how much z-axis is pointing up
        stability = base_rot[:, 2]  # Z component of rotated up vector
        return torch.mean(stability).item()

    def _calculate_energy_efficiency(self):
        """Calculate energy efficiency based on joint torques"""
        joint_velocities = self._articulation_views['humanoid'].get_joint_velocities()
        joint_positions = self._articulation_views['humanoid'].get_joint_positions()

        # Simplified energy calculation based on joint velocities
        energy = torch.mean(torch.abs(joint_velocities))
        return energy.item()

    def _calculate_task_completion(self):
        """Calculate task completion metrics"""
        base_pos, _ = self._articulation_views['humanoid'].get_world_poses()

        # For locomotion, task completion could be distance traveled
        forward_distance = torch.mean(base_pos[:, 0]).item()
        return forward_distance
```

## PPO Implementation for Humanoid Control

### 1. Actor-Critic Network Architecture

```python
# humanoid_ppo_networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class HumanoidActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Actor-specific head
        self.action_mean = nn.Linear(prev_dim, action_dim)
        self.action_std = nn.Parameter(torch.zeros(action_dim))  # Learnable std

    def forward(self, obs):
        features = self.shared_layers(obs)
        mean = torch.tanh(self.action_mean(features))  # Use tanh for bounded actions
        std = torch.exp(self.action_std)

        return mean, std

class HumanoidCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        # Shared layers
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared_layers = nn.Sequential(*layers)

        # Critic-specific head
        self.value = nn.Linear(prev_dim, 1)

    def forward(self, obs):
        features = self.shared_layers(obs)
        value = self.value(features)

        return value

class HumanoidActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()

        self.actor = HumanoidActor(obs_dim, action_dim, hidden_dims)
        self.critic = HumanoidCritic(obs_dim, hidden_dims)

    def act(self, obs):
        action_mean, action_std = self.actor(obs)
        dist = torch.distributions.Normal(action_mean, action_std)

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob

    def evaluate(self, obs, action):
        action_mean, action_std = self.actor(obs)
        dist = torch.distributions.Normal(action_mean, action_std)

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(obs)

        return log_prob, entropy, value
```

### 2. PPO Training Algorithm

```python
# humanoid_ppo_trainer.py
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import os

class HumanoidPPOTrainer:
    def __init__(self,
                 actor_critic: HumanoidActorCritic,
                 clip_param: float = 0.2,
                 ppo_epoch: int = 10,
                 num_mini_batches: int = 32,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 lr: float = 3e-4,
                 eps: float = 1e-5,
                 max_grad_norm: float = 1.0):

        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        # For logging
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)

    def update(self, rollouts):
        """Update the policy using PPO"""
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.feed_forward_generator(
                advantages, self.num_mini_batches)

            for sample in data_generator:
                obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                adv_targ = sample

                # Reshape to (N, -1)
                obs_batch = obs_batch.view(-1, *obs_batch.shape[2:])
                actions_batch = actions_batch.view(-1, actions_batch.shape[-1])
                old_action_log_probs_batch = old_action_log_probs_batch.view(-1, 1)
                adv_targ = adv_targ.view(-1, 1)

                # Calculate ratio (pi_theta / pi_theta_old)
                log_prob, dist_entropy, value_pred = self.actor_critic.evaluate(
                    obs_batch, actions_batch)

                ratio = torch.exp(log_prob - old_action_log_probs_batch)

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                           1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = value_preds_batch + \
                    (value_pred - value_preds_batch).clamp(-self.clip_param, self.clip_param)

                value_losses = (value_pred - return_batch).pow(2)
                value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                value_loss = 0.5 * torch.max(value_losses,
                                            value_losses_clipped).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()

                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                              self.max_grad_norm)

                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batches

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

    def collect_rollouts(self, env, num_steps=2048):
        """Collect rollouts for training"""
        rollouts = RolloutStorage(num_steps, env.num_envs, env.observation_space.shape,
                                  env.action_space.shape)

        obs = env.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        for step in range(num_steps):
            with torch.no_grad():
                value, action, action_log_prob = self.actor_critic.act(
                    rollouts.obs[step])

            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])

            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])

            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        return rollouts

class RolloutStorage:
    def __init__(self, num_steps, num_envs, obs_shape, action_space):
        self.obs = torch.zeros(num_steps + 1, num_envs, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1)
        self.actions = torch.zeros(num_steps, num_envs, action_space[0])
        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)

    def insert(self, obs, action, action_log_prob, value_pred, reward, mask):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_prob)
        self.value_preds[self.step].copy_(value_pred)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = self.rewards[step] + gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]
```

## Isaac Sim Integration for Training

### 1. Training Loop Integration

```python
# training_loop.py
import os
import torch
import numpy as np
from datetime import datetime
import json

def train_humanoid_walk_policy():
    """Main training loop for humanoid walk policy"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize Isaac Sim environment
    env = initialize_isaac_environment()

    # Initialize policy network
    obs_dim = env.num_obs
    action_dim = env.num_actions

    actor_critic = HumanoidActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=[512, 256, 128]
    ).to(device)

    # Initialize trainer
    trainer = HumanoidPPOTrainer(
        actor_critic=actor_critic,
        clip_param=0.2,
        ppo_epoch=10,
        num_mini_batches=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=3e-4
    )

    # Training parameters
    num_iterations = 10000
    save_interval = 100
    log_interval = 10

    # Create directory for saving models
    save_dir = "models/humanoid_walk"
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for iteration in range(num_iterations):
        # Collect rollouts
        rollouts = trainer.collect_rollouts(env, num_steps=2048)

        # Update policy
        value_loss, action_loss, dist_entropy = trainer.update(rollouts)

        # Log metrics
        if iteration % log_interval == 0 and len(trainer.episode_rewards) > 0:
            avg_reward = np.mean(trainer.episode_rewards)
            avg_length = np.mean(trainer.episode_lengths)

            print(f"Iteration {iteration}:")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.2f}")
            print(f"  Value Loss: {value_loss:.4f}")
            print(f"  Action Loss: {action_loss:.4f}")
            print(f"  Entropy: {dist_entropy:.4f}")

            # Save metrics
            metrics = {
                'iteration': iteration,
                'avg_reward': avg_reward,
                'avg_length': avg_length,
                'value_loss': value_loss,
                'action_loss': action_loss,
                'entropy': dist_entropy,
                'timestamp': datetime.now().isoformat()
            }

            with open(f"{save_dir}/metrics.json", "a") as f:
                f.write(json.dumps(metrics) + "\n")

        # Save model checkpoint
        if iteration % save_interval == 0:
            checkpoint_path = f"{save_dir}/checkpoint_{iteration}.pt"
            torch.save({
                'iteration': iteration,
                'actor_critic_state_dict': actor_critic.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': {
                    'avg_reward': np.mean(trainer.episode_rewards) if trainer.episode_rewards else 0,
                    'avg_length': np.mean(trainer.episode_lengths) if trainer.episode_lengths else 0
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = f"{save_dir}/final_policy.pt"
    torch.save(actor_critic.state_dict(), final_path)
    print(f"Saved final policy: {final_path}")

    return actor_critic

def initialize_isaac_environment():
    """Initialize Isaac Sim environment for training"""
    from omni.isaac.gym.vec_env import VecEnvImplementation
    from omni.isaac.core import World

    # Initialize Isaac Sim world
    world = World(stage_units_in_meters=1.0)

    # Create vectorized environment
    env = VecEnvImplementation(
        task=HumanoidLocomotionTask(
            name="HumanoidLocomotion",
            env=None,
            offset=None
        ),
        num_envs=2048,
        sim_device="gpu",
        graphics_device="gpu",
        headless=True
    )

    return env

def test_policy(model_path, env):
    """Test trained policy in Isaac Sim"""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load trained model
    actor_critic = torch.load(model_path, map_location=device)
    actor_critic.eval()

    # Test the policy
    obs = env.reset()
    total_reward = 0

    for step in range(1000):  # Test for 1000 steps
        with torch.no_grad():
            action, _ = actor_critic.act(torch.from_numpy(obs).float().to(device))

        obs, reward, done, info = env.step(action.cpu().numpy())
        total_reward += np.mean(reward)

        if np.any(done):
            print(f"Episode finished at step {step}, total reward: {total_reward}")
            obs = env.reset()
            total_reward = 0

    print(f"Testing completed")
```

## Reward Engineering for Stable Walking

### 1. Shaped Reward Function

```python
# reward_shaping.py
import torch

class HumanoidWalkReward:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.weights = {
            'forward_velocity': 1.0,
            'upright_posture': 0.5,
            'energy_efficiency': -0.001,
            'smooth_movement': 0.1,
            'joint_limits': -1.0,
            'fall_penalty': -10.0
        }

    def calculate_reward(self,
                        base_pos, base_quat, base_lin_vel, base_ang_vel,
                        joint_pos, joint_vel, joint_targets,
                        prev_joint_pos, prev_joint_vel):
        """
        Calculate reward for humanoid walking behavior
        """
        reward = torch.zeros(len(base_pos), device=self.device)

        # Forward velocity reward - encourage forward movement
        forward_vel = base_lin_vel[:, 0]  # X component is forward
        target_vel = 1.0  # Desired forward velocity
        vel_error = torch.abs(forward_vel - target_vel)
        forward_reward = self.weights['forward_velocity'] * torch.exp(-vel_error)
        reward += forward_reward

        # Upright posture reward - keep robot upright
        z_up = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device)
        base_rot = self.rotate_vector_by_quat(base_quat, z_up)
        upright_reward = self.weights['upright_posture'] * base_rot[:, 2]  # Z component of up vector
        reward += upright_reward

        # Energy efficiency - penalize excessive joint velocities
        energy_penalty = self.weights['energy_efficiency'] * torch.mean(torch.abs(joint_vel), dim=1)
        reward += energy_penalty

        # Smooth movement - penalize jerky motions
        if prev_joint_vel is not None:
            jerk_penalty = -torch.mean(torch.abs(joint_vel - prev_joint_vel), dim=1)
            smooth_reward = self.weights['smooth_movement'] * jerk_penalty
            reward += smooth_reward

        # Joint limits - penalize approaching joint limits
        joint_limit_penalty = self._joint_limits_penalty(joint_pos)
        reward += joint_limit_penalty

        # Fall penalty - heavily penalize falling
        fall_penalty = self._fall_penalty(base_pos)
        reward += fall_penalty

        return reward

    def rotate_vector_by_quat(self, quats, vectors):
        """Rotate vectors by quaternions"""
        # Convert quats and vectors to appropriate format
        # Implementation of quaternion rotation
        w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        vx, vy, vz = vectors[0], vectors[1], vectors[2]

        # Quaternion rotation formula
        rotated_x = vx * (1 - 2*y*y - 2*z*z) + vy * (2*x*y - 2*w*z) + vz * (2*x*z + 2*w*y)
        rotated_y = vx * (2*x*y + 2*w*z) + vy * (1 - 2*x*x - 2*z*z) + vz * (2*y*z - 2*w*x)
        rotated_z = vx * (2*x*z - 2*w*y) + vy * (2*y*z + 2*w*x) + vz * (1 - 2*x*x - 2*y*y)

        return torch.stack([rotated_x, rotated_y, rotated_z], dim=1)

    def _joint_limits_penalty(self, joint_pos):
        """Calculate penalty for approaching joint limits"""
        # This would compare joint positions to defined limits
        # For demonstration, we'll use a simple penalty
        # In practice, you'd have specific joint limits for each joint
        limits_violated = torch.abs(joint_pos) > 2.5  # Example limit
        penalty = self.weights['joint_limits'] * torch.sum(limits_violated.float(), dim=1)
        return penalty

    def _fall_penalty(self, base_pos):
        """Calculate penalty for falling"""
        fall_threshold = 0.5  # If base Z position drops below this, robot has fallen
        has_fallen = base_pos[:, 2] < fall_threshold
        penalty = torch.where(has_fallen,
                             torch.tensor(self.weights['fall_penalty'], device=self.device),
                             torch.tensor(0.0, device=self.device))
        return penalty
```

## Domain Randomization for Robust Training

### 1. Randomization Techniques

```python
# domain_randomization.py
import torch
import numpy as np

class DomainRandomization:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.randomization_params = {
            'mass_variation': (-0.2, 0.2),  # ±20% mass variation
            'friction_variation': (0.5, 1.5),  # Range of friction coefficients
            'motor_strength_variation': (0.8, 1.2),  # ±20% motor strength
            'sensor_noise_std': (0.0, 0.05),  # Sensor noise range
            'terrain_roughness': (0.0, 0.02),  # Terrain variation
            'gravity_variation': (-0.1, 0.1)  # ±0.1 m/s² gravity variation
        }

    def randomize_robot_properties(self, env_ids=None):
        """Randomize robot physical properties"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Randomize mass properties
        mass_multipliers = torch.rand(len(env_ids), device=self.device) * \
                          (self.randomization_params['mass_variation'][1] -
                           self.randomization_params['mass_variation'][0]) + \
                          self.randomization_params['mass_variation'][0]

        # Apply mass multipliers to robot links
        self._apply_mass_randomization(env_ids, mass_multipliers)

        # Randomize friction coefficients
        friction_coeffs = torch.rand(len(env_ids), device=self.device) * \
                         (self.randomization_params['friction_variation'][1] -
                          self.randomization_params['friction_variation'][0]) + \
                         self.randomization_params['friction_variation'][0]

        # Apply friction randomization
        self._apply_friction_randomization(env_ids, friction_coeffs)

        # Randomize motor strengths
        motor_multipliers = torch.rand(len(env_ids), device=self.device) * \
                           (self.randomization_params['motor_strength_variation'][1] -
                            self.randomization_params['motor_strength_variation'][0]) + \
                           self.randomization_params['motor_strength_variation'][0]

        # Store for later use in control
        self.motor_strength_multipliers[env_ids] = motor_multipliers

    def randomize_sensor_noise(self, env_ids=None):
        """Randomize sensor noise parameters"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Generate random noise parameters
        noise_stds = torch.rand(len(env_ids), device=self.device) * \
                    (self.randomization_params['sensor_noise_std'][1] -
                     self.randomization_params['sensor_noise_std'][0]) + \
                    self.randomization_params['sensor_noise_std'][0]

        # Apply to sensor models
        self._apply_sensor_noise_randomization(env_ids, noise_stds)

    def randomize_terrain(self, env_ids=None):
        """Randomize terrain properties"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Create randomized terrain variations
        roughness_values = torch.rand(len(env_ids), device=self.device) * \
                          (self.randomization_params['terrain_roughness'][1] -
                           self.randomization_params['terrain_roughness'][0]) + \
                          self.randomization_params['terrain_roughness'][0]

        # Apply terrain variations
        self._apply_terrain_randomization(env_ids, roughness_values)

    def randomize_dynamics(self, env_ids=None):
        """Randomize dynamic properties including gravity"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Randomize gravity
        gravity_delta = torch.rand(len(env_ids), 3, device=self.device) * \
                       (self.randomization_params['gravity_variation'][1] -
                        self.randomization_params['gravity_variation'][0]) + \
                       self.randomization_params['gravity_variation'][0]

        # Add gravity variation (keeping Z mostly dominant)
        gravity_delta[:, :2] *= 0.1  # Limit X/Y variation
        gravity_delta[:, 2] += 9.8  # Add to Z component

        # Apply gravity randomization
        self._apply_gravity_randomization(env_ids, gravity_delta)

    def reset_randomization(self, env_ids):
        """Reset randomization for specific environments"""
        # Randomize all properties for the specified environments
        self.randomize_robot_properties(env_ids)
        self.randomize_sensor_noise(env_ids)
        self.randomize_terrain(env_ids)
        self.randomize_dynamics(env_ids)
```

## Sim-to-Real Transfer Techniques

### 1. Reducing the Reality Gap

```python
# sim_to_real_transfer.py
import torch
import numpy as np

class SimToRealTransfer:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.transfer_strategies = {
            'domain_randomization': True,
            'system_identification': True,
            'policy_regularization': True,
            'robust_control': True
        }

    def apply_domain_randomization(self, obs):
        """Apply domain randomization to observations"""
        # Add realistic noise to observations
        noise_level = 0.01
        noise = torch.randn_like(obs) * noise_level
        noisy_obs = obs + noise

        return noisy_obs

    def implement_robust_control(self, action):
        """Implement robust control techniques to handle model uncertainty"""
        # Add small amount of exploration noise to actions
        # This makes the policy more robust to modeling errors
        noise_scale = 0.05
        robust_action = action + torch.randn_like(action) * noise_scale

        # Clamp to valid range
        robust_action = torch.clamp(robust_action, -1.0, 1.0)

        return robust_action

    def adapt_to_real_robot(self, sim_policy, real_robot_params):
        """Adapt simulation policy to real robot parameters"""
        # This would typically involve:
        # 1. System identification to find real robot parameters
        # 2. Policy adaptation techniques
        # 3. Transfer learning approaches

        # For now, we'll return the original policy with some safety modifications
        adapted_policy = sim_policy

        # Modify control gains to be more conservative
        # This is a simplified approach - real implementations would be more sophisticated
        return adapted_policy

    def validate_transfer_safety(self, action, robot_state):
        """Validate that actions are safe for real robot execution"""
        # Check if action violates safety constraints
        # This would include joint limits, velocity limits, etc.

        # Implement safety checks
        safe_action = self._check_joint_limits(action, robot_state)
        safe_action = self._check_velocity_limits(safe_action, robot_state)
        safe_action = self._check_stability(safe_action, robot_state)

        return safe_action

    def _check_joint_limits(self, action, robot_state):
        """Check and enforce joint limits"""
        # This would check against actual robot joint limits
        # For now, we'll just clamp to reasonable ranges
        return torch.clamp(action, -2.0, 2.0)  # Reasonable joint limits

    def _check_velocity_limits(self, action, robot_state):
        """Check and enforce velocity limits"""
        # This would implement velocity limiting based on robot capabilities
        return action

    def _check_stability(self, action, robot_state):
        """Check if action maintains robot stability"""
        # This would check Center of Mass (CoM) position relative to support polygon
        # For now, we'll just return the action
        return action
```

## Training Monitoring and Evaluation

### 1. Performance Metrics

```python
# training_monitoring.py
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class TrainingMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.episode_count = 0

    def record_episode(self, episode_data):
        """Record data from completed episode"""
        self.metrics['episode_rewards'].append(episode_data['reward'])
        self.metrics['episode_lengths'].append(episode_data['length'])
        self.metrics['episode_velocities'].append(episode_data['avg_velocity'])
        self.metrics['episode_stability'].append(episode_data['balance_score'])
        self.metrics['episode_energy'].append(episode_data['energy_usage'])

        self.episode_count += 1

    def get_performance_summary(self):
        """Get summary of training performance"""
        if len(self.metrics['episode_rewards']) == 0:
            return "No episodes completed yet"

        avg_reward = np.mean(self.metrics['episode_rewards'][-100:])  # Last 100 episodes
        avg_length = np.mean(self.metrics['episode_lengths'][-100:])
        avg_velocity = np.mean(self.metrics['episode_velocities'][-100:])
        avg_stability = np.mean(self.metrics['episode_stability'][-100:])
        avg_energy = np.mean(self.metrics['episode_energy'][-100:])

        summary = f"""
        Performance Summary (Last 100 Episodes):
        - Average Reward: {avg_reward:.2f}
        - Average Episode Length: {avg_length:.2f}
        - Average Velocity: {avg_velocity:.2f} m/s
        - Average Stability Score: {avg_stability:.2f}
        - Average Energy Usage: {avg_energy:.2f}
        """

        return summary

    def plot_training_progress(self):
        """Plot training progress metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot episode rewards
        axes[0, 0].plot(self.metrics['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Plot episode lengths
        axes[0, 1].plot(self.metrics['episode_lengths'])
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Steps')

        # Plot velocities
        axes[0, 2].plot(self.metrics['episode_velocities'])
        axes[0, 2].set_title('Average Velocities')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Velocity (m/s)')

        # Plot stability scores
        axes[1, 0].plot(self.metrics['episode_stability'])
        axes[1, 0].set_title('Balance/Stability Scores')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Score')

        # Plot energy usage
        axes[1, 1].plot(self.metrics['episode_energy'])
        axes[1, 1].set_title('Energy Usage')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Energy')

        # Plot moving averages
        if len(self.metrics['episode_rewards']) > 50:
            window = 50
            moving_avg = np.convolve(self.metrics['episode_rewards'],
                                   np.ones(window)/window, mode='valid')
            axes[1, 2].plot(moving_avg)
            axes[1, 2].set_title(f'Moving Average (window={window})')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Average Reward')
        else:
            axes[1, 2].text(0.5, 0.5, 'Not enough data',
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Moving Average')

        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.show()
```

## Learning Objectives

After completing this example, you should understand:
- How to implement reinforcement learning for humanoid locomotion
- The architecture of actor-critic networks for continuous control
- Techniques for reward engineering in robotic tasks
- Domain randomization for robust policy training
- Approaches for sim-to-real transfer in robotics
- Methods for monitoring and evaluating RL training progress

## Best Practices

### 1. Training Stability
- Use appropriate reward scaling to prevent gradient explosions
- Implement proper normalization of observations and actions
- Use domain randomization to improve policy robustness
- Monitor training metrics continuously for early intervention

### 2. Safety Considerations
- Implement safety checks during policy execution
- Use conservative control parameters initially
- Gradually increase performance requirements
- Include safety constraints in reward function

### 3. Efficiency Optimization
- Use parallel environments for faster training
- Implement efficient neural network architectures
- Optimize simulation performance with appropriate settings
- Use curriculum learning for complex behaviors

## Troubleshooting Common Issues

### 1. Training Instability
- **Symptoms**: Oscillating rewards, diverging policies
- **Solutions**: Reduce learning rate, increase batch size, add regularization

### 2. Poor Convergence
- **Symptoms**: Slow improvement, getting stuck in local minima
- **Solutions**: Adjust network architecture, modify reward function, change hyperparameters

### 3. Sim-to-Real Gap
- **Symptoms**: Good simulation performance, poor real-world performance
- **Solutions**: Increase domain randomization, implement system identification, add robustness training

## Next Steps

Continue to learn about [Module 4: Vision-Language-Action Integration](../module-4-vla/index) to understand how to combine perception, language understanding, and physical action for complete humanoid robot systems.