"""
Full GPU PPO Algorithm
All operations (forward, loss, backward, optimizer) run on CUDA.
No CPU gradient computation. Uses PyTorch autograd on GPU tensors.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .policy import ActorCritic
from .buffer import GPURolloutBuffer
from .env import GPUQuadrotorEnv, QuadrotorConfig


@dataclass
class PPOConfig:
    """PPO hyperparameters"""
    # Environment
    num_envs: int = 512
    rollout_steps: int = 512
    
    # PPO
    ppo_epochs: int = 10
    minibatch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    clip_value: bool = True
    value_clip_epsilon: float = 0.2
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Optimizer
    learning_rate: float = 3e-4
    adam_eps: float = 1e-5
    
    # Training
    total_timesteps: int = 50_000_000
    log_interval: int = 10
    save_interval: int = 50
    
    # Network
    hidden_dim: int = 64
    
    # Seed
    seed: int = 42


class GPUPPO:
    """
    Proximal Policy Optimization running entirely on GPU.
    
    PyTorch autograd runs on GPU when tensors are CUDA tensors.
    All optimizer operations are on GPU tensors.
    No CPU gradient path exists.
    """
    
    def __init__(
        self,
        config: PPOConfig,
        device: torch.device = None,
        checkpoint_dir: str = "checkpoints"
    ):
        self.config = config
        self.device = device or torch.device('cuda')
        self.checkpoint_dir = checkpoint_dir
        
        # Set seeds for reproducibility
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = True
        
        # Create environment
        self.env = GPUQuadrotorEnv(
            num_envs=config.num_envs,
            device=self.device,
            seed=config.seed
        )
        
        # Create policy
        self.policy = ActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            hidden_dim=config.hidden_dim,
            device=self.device
        )
        
        # Optimizer (operates on GPU tensors)
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=config.adam_eps
        )
        
        # Rollout buffer (GPU)
        self.buffer = GPURolloutBuffer(
            buffer_size=config.rollout_steps,
            num_envs=config.num_envs,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            device=self.device,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda
        )
        
        # Training state
        self.global_step = 0
        self.update_count = 0
        
        # Enable TF32 for faster matrix ops on Ampere/Ada GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Optionally compile model for faster execution (PyTorch 2.0+)
        # Note: torch.compile adds significant overhead on first run
        # For short training, disable. For long training (>50M steps), enable.
        # if hasattr(torch, 'compile'):
        #     self.policy = torch.compile(self.policy, mode='reduce-overhead')
    
    def collect_rollout(self) -> Dict[str, float]:
        """
        Collect rollout data using current policy.
        All operations on GPU.
        
        Returns:
            Dictionary of rollout statistics
        """
        self.buffer.reset()
        self.policy.eval()
        
        # Get initial observations
        obs = self.env.reset()
        
        episode_returns = []
        episode_lengths = []
        
        with torch.no_grad():
            for step in range(self.config.rollout_steps):
                # Get action from policy (on GPU)
                action, log_prob, value = self.policy.get_action(obs)
                
                # Step environment (on GPU)
                next_obs, reward, done, info = self.env.step(action)
                
                # Store transition
                self.buffer.add(
                    obs=obs,
                    action=action,
                    reward=reward,
                    done=done,
                    value=value,
                    log_prob=log_prob
                )
                
                # Track episode statistics
                if len(info['episode_return']) > 0:
                    episode_returns.extend(info['episode_return'].tolist())
                
                obs = next_obs
                self.global_step += self.config.num_envs
            
            # Get value for last state (for GAE)
            _, _, last_value = self.policy.get_action(obs)
            last_done = self.env._check_done()
        
        # Update observation statistics
        all_obs = self.buffer.observations.reshape(-1, self.env.obs_dim)
        self.env.update_obs_stats(all_obs)
        
        # Compute GAE on GPU
        self.buffer.compute_gae(last_value, last_done)
        
        # Compute statistics
        stats = {
            'mean_reward': self.buffer.rewards.mean().item(),
            'mean_value': self.buffer.values.mean().item(),
            'mean_return': sum(episode_returns) / max(len(episode_returns), 1),
            'num_episodes': len(episode_returns)
        }
        
        return stats
    
    def update_policy(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        All operations (forward, loss, backward, optimizer) on GPU.
        
        Returns:
            Dictionary of training statistics
        """
        self.policy.train()
        
        # Statistics accumulators (on GPU for efficiency)
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy = torch.tensor(0.0, device=self.device)
        total_approx_kl = torch.tensor(0.0, device=self.device)
        total_clip_frac = torch.tensor(0.0, device=self.device)
        num_updates = 0
        
        for epoch in range(self.config.ppo_epochs):
            for batch in self.buffer.get_samples(self.config.minibatch_size):
                (
                    obs_batch,
                    actions_batch,
                    old_values_batch,
                    old_log_probs_batch,
                    advantages_batch,
                    returns_batch
                ) = batch
                
                # Forward pass (on GPU)
                new_log_probs, entropy, new_values = self.policy.evaluate_actions(
                    obs_batch, actions_batch
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(
                    ratio, 
                    1.0 - self.config.clip_epsilon, 
                    1.0 + self.config.clip_epsilon
                ) * advantages_batch
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (optionally clipped)
                if self.config.clip_value:
                    values_clipped = old_values_batch + torch.clamp(
                        new_values - old_values_batch,
                        -self.config.value_clip_epsilon,
                        self.config.value_clip_epsilon
                    )
                    value_loss1 = (new_values - returns_batch) ** 2
                    value_loss2 = (values_clipped - returns_batch) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((new_values - returns_batch) ** 2).mean()
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss + 
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                # Backward pass (on GPU via autograd)
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), 
                    self.config.max_grad_norm
                )
                
                # Optimizer step (on GPU)
                self.optimizer.step()
                
                # Track statistics (keep on GPU)
                with torch.no_grad():
                    total_policy_loss += policy_loss
                    total_value_loss += value_loss
                    total_entropy += entropy.mean()
                    
                    # Approximate KL divergence
                    log_ratio = new_log_probs - old_log_probs_batch
                    approx_kl = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    total_approx_kl += approx_kl
                    
                    # Clipping fraction
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
                    total_clip_frac += clip_frac
                
                num_updates += 1
        
        self.update_count += 1
        
        # Move statistics to CPU only at the end
        stats = {
            'policy_loss': (total_policy_loss / num_updates).item(),
            'value_loss': (total_value_loss / num_updates).item(),
            'entropy': (total_entropy / num_updates).item(),
            'approx_kl': (total_approx_kl / num_updates).item(),
            'clip_fraction': (total_clip_frac / num_updates).item(),
        }
        
        return stats
    
    def train(self, total_timesteps: int = None) -> Dict[str, list]:
        """
        Main training loop.
        
        Args:
            total_timesteps: Total environment steps (overrides config)
            
        Returns:
            Dictionary of training history
        """
        import time
        import os
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        num_updates = total_timesteps // (self.config.num_envs * self.config.rollout_steps)
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        history = {
            'mean_return': [],
            'mean_reward': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'steps_per_second': []
        }
        
        print(f"Training PPO on GPU: {torch.cuda.get_device_name()}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Num updates: {num_updates}")
        print(f"Batch size: {self.config.num_envs * self.config.rollout_steps:,}")
        print()
        
        start_time = time.time()
        
        for update in range(1, num_updates + 1):
            update_start = time.time()
            
            # Collect rollout (GPU)
            rollout_stats = self.collect_rollout()
            
            # Update policy (GPU)
            train_stats = self.update_policy()
            
            # Compute timing
            update_time = time.time() - update_start
            steps_per_sec = (self.config.num_envs * self.config.rollout_steps) / update_time
            
            # Log
            if update % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"Update {update}/{num_updates} | "
                    f"Steps: {self.global_step:,} | "
                    f"Return: {rollout_stats['mean_return']:.2f} | "
                    f"Reward: {rollout_stats['mean_reward']:.4f} | "
                    f"PL: {train_stats['policy_loss']:.4f} | "
                    f"VL: {train_stats['value_loss']:.4f} | "
                    f"Ent: {train_stats['entropy']:.4f} | "
                    f"KL: {train_stats['approx_kl']:.4f} | "
                    f"Steps/s: {steps_per_sec:.0f}"
                )
                
                history['mean_return'].append(rollout_stats['mean_return'])
                history['mean_reward'].append(rollout_stats['mean_reward'])
                history['policy_loss'].append(train_stats['policy_loss'])
                history['value_loss'].append(train_stats['value_loss'])
                history['entropy'].append(train_stats['entropy'])
                history['steps_per_second'].append(steps_per_sec)
            
            # Save checkpoint
            if update % self.config.save_interval == 0:
                self.save_checkpoint(f"checkpoint_{update}.pt")
        
        # Final save
        self.save_checkpoint("final.pt")
        
        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average steps/s: {self.global_step / total_time:.0f}")
        
        return history
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        import os
        
        path = os.path.join(self.checkpoint_dir, filename)
        
        # Get underlying model if compiled
        model = self.policy
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        
        torch.save({
            'policy_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'update_count': self.update_count,
            'config': self.config,
            'obs_mean': self.env.obs_mean.cpu(),
            'obs_std': torch.sqrt(self.env.obs_var + 1e-8).cpu()
        }, path)
        
        print(f"  Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Get underlying model if compiled
        model = self.policy
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        
        model.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.update_count = checkpoint['update_count']
        
        if 'obs_mean' in checkpoint:
            self.env.obs_mean = checkpoint['obs_mean'].to(self.device)
            self.env.obs_var = (checkpoint['obs_std'] ** 2).to(self.device)
        
        print(f"Loaded checkpoint: {path}")
    
    def evaluate(self, num_episodes: int = 100) -> Dict[str, float]:
        """
        Evaluate policy with deterministic actions.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary of evaluation statistics
        """
        self.policy.eval()
        
        obs = self.env.reset()
        
        episode_returns = []
        episode_lengths = []
        
        with torch.no_grad():
            while len(episode_returns) < num_episodes:
                # Get deterministic action
                action = self.policy.get_deterministic_action(obs)
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                # Track completed episodes
                if len(info['episode_return']) > 0:
                    episode_returns.extend(info['episode_return'].tolist())
        
        return {
            'mean_return': sum(episode_returns) / len(episode_returns),
            'std_return': torch.tensor(episode_returns).std().item(),
            'min_return': min(episode_returns),
            'max_return': max(episode_returns)
        }


def create_ppo(
    config: PPOConfig = None,
    device: torch.device = None,
    checkpoint_dir: str = "checkpoints"
) -> GPUPPO:
    """Factory function to create PPO trainer"""
    config = config or PPOConfig()
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return GPUPPO(config=config, device=device, checkpoint_dir=checkpoint_dir)
