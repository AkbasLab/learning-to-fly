"""
Neural Network Policy for PPO
All operations on GPU with proper initialization.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor: Outputs action mean with tanh squashing.
    Critic: Outputs state value estimate.
    
    Architecture: MLP with tanh activations (matching rl_tools).
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        init_log_std: float = -1.0,
        device: torch.device = None
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device or torch.device('cuda')
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.full((action_dim,), init_log_std, device=self.device)
        )
        
        # Critic network (separate from actor)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Move to device
        self.to(self.device)
    
    def _init_weights(self):
        """Xavier initialization matching rl_tools"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in = module.weight.shape[1]
                fan_out = module.weight.shape[0]
                std = math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.normal_(module.weight, mean=0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both actor and critic.
        
        Args:
            obs: [batch, obs_dim] observations
            
        Returns:
            action_mean: [batch, action_dim] in [-1, 1]
            value: [batch, 1] state value
        """
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: [batch, obs_dim]
            deterministic: if True, return mean action
            
        Returns:
            action: [batch, action_dim]
            log_prob: [batch]
            value: [batch]
        """
        action_mean, value = self.forward(obs)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros(obs.shape[0], device=self.device)
        else:
            # Sample from Gaussian in pre-tanh space
            std = torch.exp(self.log_std)
            
            # Convert mean from tanh space to pre-tanh space
            # mean_pre = atanh(action_mean)
            action_mean_clamped = torch.clamp(action_mean, -0.999, 0.999)
            mean_pre = torch.atanh(action_mean_clamped)
            
            # Sample
            noise = torch.randn_like(action_mean)
            action_pre = mean_pre + std * noise
            
            # Apply tanh squashing
            action = torch.tanh(action_pre)
            
            # Log probability with tanh correction
            log_prob = self._compute_log_prob(action_mean, action, std, noise)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Args:
            obs: [batch, obs_dim]
            actions: [batch, action_dim]
            
        Returns:
            log_prob: [batch]
            entropy: [batch]
            value: [batch]
        """
        action_mean, value = self.forward(obs)
        std = torch.exp(self.log_std)
        
        # Compute log probability
        action_clamped = torch.clamp(actions, -0.999, 0.999)
        action_mean_clamped = torch.clamp(action_mean, -0.999, 0.999)
        
        # Inverse tanh
        u = torch.atanh(action_clamped)
        mu = torch.atanh(action_mean_clamped)
        
        # Gaussian log prob in pre-tanh space
        var = std ** 2
        log_prob_gaussian = -0.5 * ((u - mu) ** 2 / var + 
                                     torch.log(var) + 
                                     math.log(2 * math.pi))
        
        # Jacobian correction for tanh
        log_prob_correction = torch.log(1 - action_clamped ** 2 + 1e-6)
        
        log_prob = (log_prob_gaussian - log_prob_correction).sum(dim=-1)
        
        # Entropy (Gaussian entropy)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum()
        entropy = entropy.expand(obs.shape[0])
        
        return log_prob, entropy, value.squeeze(-1)
    
    def _compute_log_prob(
        self, 
        action_mean: torch.Tensor, 
        action: torch.Tensor,
        std: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Compute log probability of action"""
        # Standard normal log prob
        log_prob_noise = -0.5 * (noise ** 2 + math.log(2 * math.pi))
        log_prob_noise = log_prob_noise - self.log_std
        
        # Jacobian correction for tanh
        log_prob_correction = torch.log(1 - action ** 2 + 1e-6)
        
        log_prob = (log_prob_noise - log_prob_correction).sum(dim=-1)
        return log_prob
    
    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (for deployment)"""
        with torch.no_grad():
            action_mean = self.actor(obs)
        return action_mean
    
    def export_weights(self) -> dict:
        """Export weights for C header generation"""
        weights = {}
        
        # Actor layer 1
        weights['actor_l1_w'] = self.actor[0].weight.detach().cpu().numpy()
        weights['actor_l1_b'] = self.actor[0].bias.detach().cpu().numpy()
        
        # Actor layer 2
        weights['actor_l2_w'] = self.actor[2].weight.detach().cpu().numpy()
        weights['actor_l2_b'] = self.actor[2].bias.detach().cpu().numpy()
        
        # Actor layer 3
        weights['actor_l3_w'] = self.actor[4].weight.detach().cpu().numpy()
        weights['actor_l3_b'] = self.actor[4].bias.detach().cpu().numpy()
        
        # Log std
        weights['log_std'] = self.log_std.detach().cpu().numpy()
        
        return weights


def create_actor_critic(
    obs_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    device: torch.device = None
) -> ActorCritic:
    """Factory function to create actor-critic network"""
    return ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device
    )
