"""
GPU Rollout Buffer for PPO
All storage and computations on GPU with no CPU synchronization.
"""

import torch
from typing import Generator, Tuple


class GPURolloutBuffer:
    """
    Rollout buffer stored entirely on GPU.
    
    Supports:
    - Efficient storage of rollout data
    - GAE computation on GPU
    - GPU-side minibatch generation with shuffling
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Total samples per rollout
        self.total_size = buffer_size * num_envs
        
        # Allocate all tensors on GPU
        self.observations = torch.zeros(
            buffer_size, num_envs, obs_dim, device=device
        )
        self.actions = torch.zeros(
            buffer_size, num_envs, action_dim, device=device
        )
        self.log_probs = torch.zeros(
            buffer_size, num_envs, device=device
        )
        self.rewards = torch.zeros(
            buffer_size, num_envs, device=device
        )
        self.values = torch.zeros(
            buffer_size, num_envs, device=device
        )
        self.dones = torch.zeros(
            buffer_size, num_envs, device=device, dtype=torch.bool
        )
        
        # Computed after rollout
        self.advantages = torch.zeros(
            buffer_size, num_envs, device=device
        )
        self.returns = torch.zeros(
            buffer_size, num_envs, device=device
        )
        
        # Position tracker
        self.pos = 0
        self.full = False
        
        # Pre-allocate index buffer for shuffling
        self.indices = torch.arange(self.total_size, device=device)
        
        # GPU random generator for shuffling
        self.generator = torch.Generator(device=device)
    
    def reset(self):
        """Reset buffer position"""
        self.pos = 0
        self.full = False
    
    def add(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """
        Add a step to the buffer.
        
        Args:
            obs: [num_envs, obs_dim]
            action: [num_envs, action_dim]
            reward: [num_envs]
            done: [num_envs]
            value: [num_envs]
            log_prob: [num_envs]
        """
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
    
    def compute_gae(self, last_values: torch.Tensor, last_dones: torch.Tensor):
        """
        Compute Generalized Advantage Estimation on GPU.
        
        Args:
            last_values: [num_envs] - value estimates for the last state
            last_dones: [num_envs] - done flags for the last state
        """
        # Initialize GAE
        gae = torch.zeros(self.num_envs, device=self.device)
        next_values = last_values
        next_dones = last_dones.float()
        
        # Backward pass through buffer
        for step in reversed(range(self.buffer_size)):
            # Mask for non-terminal transitions
            not_done = 1.0 - next_dones
            
            # TD error
            delta = (
                self.rewards[step] + 
                self.gamma * next_values * not_done - 
                self.values[step]
            )
            
            # GAE
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
            
            next_values = self.values[step]
            next_dones = self.dones[step].float()
        
        # Normalize advantages on GPU
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std()
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
    
    def get_samples(
        self, 
        batch_size: int
    ) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate minibatches with GPU-side shuffling.
        
        Args:
            batch_size: size of each minibatch
            
        Yields:
            Tuple of (obs, actions, old_values, old_log_probs, advantages, returns)
        """
        # Flatten all data
        obs_flat = self.observations.reshape(-1, self.obs_dim)
        actions_flat = self.actions.reshape(-1, self.action_dim)
        values_flat = self.values.reshape(-1)
        log_probs_flat = self.log_probs.reshape(-1)
        advantages_flat = self.advantages.reshape(-1)
        returns_flat = self.returns.reshape(-1)
        
        # Shuffle indices on GPU
        perm = torch.randperm(self.total_size, device=self.device, generator=self.generator)
        
        # Generate batches
        num_batches = self.total_size // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_indices = perm[start:end]
            
            yield (
                obs_flat[batch_indices],
                actions_flat[batch_indices],
                values_flat[batch_indices],
                log_probs_flat[batch_indices],
                advantages_flat[batch_indices],
                returns_flat[batch_indices]
            )
    
    def get_all_data(self) -> Tuple[torch.Tensor, ...]:
        """Get all data as flattened tensors"""
        return (
            self.observations.reshape(-1, self.obs_dim),
            self.actions.reshape(-1, self.action_dim),
            self.values.reshape(-1),
            self.log_probs.reshape(-1),
            self.advantages.reshape(-1),
            self.returns.reshape(-1)
        )
