"""
GPU Vectorized Quadrotor Environment
All physics simulation runs on CUDA with no CPU synchronization.
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple


@dataclass
class QuadrotorConfig:
    """Quadrotor physical parameters (Crazyflie 2.1)"""
    mass: float = 0.027  # kg
    arm_length: float = 0.046  # m
    max_thrust_per_motor: float = 0.06  # N (0.015 kg * 4 motors)
    
    # Inertia tensor
    Jxx: float = 1.4e-5
    Jyy: float = 1.4e-5
    Jzz: float = 2.17e-5
    
    # Motor dynamics
    motor_time_constant: float = 0.02  # seconds
    
    # Physics
    gravity: float = 9.81
    dt: float = 0.01
    substeps: int = 4
    
    # Episode
    max_steps: int = 500
    position_bound: float = 5.0
    
    # Reward weights
    position_weight: float = 1.0
    velocity_weight: float = 0.01
    angular_velocity_weight: float = 0.001
    action_weight: float = 0.0001
    termination_penalty: float = 10.0


class GPUQuadrotorEnv:
    """
    Vectorized quadrotor environment running entirely on GPU.
    
    State: [pos(3), quat(4), vel(3), omega(3)] = 13 dims
    Action: [motor1, motor2, motor3, motor4] in [-1, 1] -> mapped to thrust
    """
    
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        config: QuadrotorConfig = None,
        seed: int = 42
    ):
        self.num_envs = num_envs
        self.device = device
        self.config = config or QuadrotorConfig()
        
        # Set deterministic seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
        # Observation and action dimensions
        self.obs_dim = 13
        self.action_dim = 4
        
        # Pre-compute constants on GPU
        self.dt_sub = torch.tensor(
            self.config.dt / self.config.substeps, 
            device=device, dtype=torch.float32
        )
        self.gravity = torch.tensor([0, 0, -self.config.gravity], device=device)
        self.mass = self.config.mass
        self.J_inv = torch.tensor([
            1.0 / self.config.Jxx,
            1.0 / self.config.Jyy,
            1.0 / self.config.Jzz
        ], device=device)
        
        # Motor mixing matrix for X-configuration
        # Motors: front-right, front-left, back-left, back-right
        L = self.config.arm_length
        self.thrust_to_torque = torch.tensor([
            [L, -L, -L, L],      # Roll torque
            [L, L, -L, -L],      # Pitch torque  
            [0.01, -0.01, 0.01, -0.01]  # Yaw torque (drag)
        ], device=device)
        
        # Motor time constant for low-pass filter
        self.motor_alpha = self.dt_sub / (self.config.motor_time_constant + self.dt_sub)
        
        # Allocate state tensors on GPU
        self._allocate_state()
        
        # Observation normalization (running statistics on GPU)
        self.obs_mean = torch.zeros(self.obs_dim, device=device)
        self.obs_var = torch.ones(self.obs_dim, device=device)
        self.obs_count = torch.tensor(1e-4, device=device)
        
    def _allocate_state(self):
        """Allocate all state tensors on GPU"""
        n = self.num_envs
        dev = self.device
        
        # Position [n, 3]
        self.pos = torch.zeros(n, 3, device=dev)
        # Quaternion [n, 4] (w, x, y, z)
        self.quat = torch.zeros(n, 4, device=dev)
        self.quat[:, 0] = 1.0  # Identity quaternion
        # Velocity [n, 3]
        self.vel = torch.zeros(n, 3, device=dev)
        # Angular velocity [n, 3]
        self.omega = torch.zeros(n, 3, device=dev)
        # Motor states (normalized 0-1)
        self.motors = torch.zeros(n, 4, device=dev)
        # Target position
        self.target = torch.zeros(n, 3, device=dev)
        # Step count
        self.step_count = torch.zeros(n, dtype=torch.int32, device=dev)
        # Episode return tracking
        self.episode_return = torch.zeros(n, device=dev)
        
    def reset(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Reset specified environments. If env_ids is None, reset all.
        Returns: observations [num_reset, obs_dim]
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        n = len(env_ids)
        
        # Random initial position within [-1, 1]^3
        self.pos[env_ids] = (torch.rand(n, 3, device=self.device) * 2 - 1)
        
        # Identity quaternion (upright)
        self.quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(n, 4)
        
        # Zero velocity
        self.vel[env_ids] = 0.0
        self.omega[env_ids] = 0.0
        
        # Hover motor state
        hover_thrust = self.mass * self.config.gravity / (4 * self.config.max_thrust_per_motor)
        self.motors[env_ids] = hover_thrust
        
        # Zero step count
        self.step_count[env_ids] = 0
        self.episode_return[env_ids] = 0
        
        # Target at origin
        self.target[env_ids] = 0
        
        return self._get_obs(env_ids)
    
    def _get_obs(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Get observations for specified environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Observation: [pos_error, quat, vel, omega]
        pos_error = self.pos[env_ids] - self.target[env_ids]
        
        obs = torch.cat([
            pos_error,           # 3
            self.quat[env_ids],  # 4
            self.vel[env_ids],   # 3
            self.omega[env_ids]  # 3
        ], dim=-1)  # Total: 13
        
        return obs
    
    def _normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize observations using running statistics"""
        return (obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
    
    def update_obs_stats(self, obs: torch.Tensor):
        """Update running observation statistics (Welford's algorithm on GPU)"""
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.obs_mean
        total_count = self.obs_count + batch_count
        
        self.obs_mean = self.obs_mean + delta * batch_count / total_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.obs_count * batch_count / total_count
        self.obs_var = M2 / total_count
        self.obs_count = total_count
    
    @torch.jit.export
    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication: q1 * q2"""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)
    
    @torch.jit.export
    def _rotate_vector(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector v by quaternion q"""
        # q * [0, v] * q_conj
        qv = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
        q_conj = q * torch.tensor([1, -1, -1, -1], device=q.device)
        
        temp = self._quat_multiply(q, qv)
        result = self._quat_multiply(temp, q_conj)
        
        return result[..., 1:]
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Step all environments with given actions.
        
        Args:
            actions: [num_envs, 4] in range [-1, 1]
            
        Returns:
            obs: [num_envs, obs_dim]
            rewards: [num_envs]
            dones: [num_envs] (bool)
            info: dict with additional info
        """
        # Clamp actions to valid range
        actions = torch.clamp(actions, -1, 1)
        
        # Map actions from [-1, 1] to [0, 1] motor command
        motor_cmd = (actions + 1) * 0.5
        
        # Substep physics
        for _ in range(self.config.substeps):
            self._physics_step(motor_cmd)
        
        self.step_count += 1
        
        # Compute rewards
        rewards = self._compute_reward(actions)
        self.episode_return += rewards
        
        # Check termination
        dones = self._check_done()
        
        # Get observations
        obs = self._get_obs()
        
        # Auto-reset done environments
        done_ids = torch.where(dones)[0]
        if len(done_ids) > 0:
            final_returns = self.episode_return[done_ids].clone()
            self.reset(done_ids)
        else:
            final_returns = torch.tensor([], device=self.device)
        
        info = {
            'episode_return': final_returns,
            'episode_length': self.step_count.float().mean()
        }
        
        return obs, rewards, dones, info
    
    def _physics_step(self, motor_cmd: torch.Tensor):
        """Single physics substep"""
        # Low-pass filter motor response
        self.motors = self.motors + self.motor_alpha * (motor_cmd - self.motors)
        
        # Compute thrust per motor [N]
        thrusts = self.motors * self.config.max_thrust_per_motor
        total_thrust = thrusts.sum(dim=-1, keepdim=True)
        
        # Thrust in body frame (always +Z)
        thrust_body = torch.zeros_like(self.pos)
        thrust_body[:, 2] = total_thrust.squeeze() / self.mass
        
        # Rotate to world frame
        thrust_world = self._rotate_vector(self.quat, thrust_body)
        
        # Linear acceleration (thrust + gravity)
        acc = thrust_world + self.gravity
        
        # Angular acceleration from torques
        torques = torch.matmul(thrusts, self.thrust_to_torque.T)  # [n, 3]
        alpha = torques * self.J_inv  # Simplified (diagonal inertia)
        
        # Integrate velocity and position
        self.vel = self.vel + acc * self.dt_sub
        self.pos = self.pos + self.vel * self.dt_sub
        
        # Integrate angular velocity
        self.omega = self.omega + alpha * self.dt_sub
        
        # Integrate quaternion
        omega_quat = torch.cat([
            torch.zeros(self.num_envs, 1, device=self.device),
            self.omega * self.dt_sub * 0.5
        ], dim=-1)
        
        self.quat = self.quat + self._quat_multiply(self.quat, omega_quat)
        
        # Normalize quaternion
        self.quat = F.normalize(self.quat, dim=-1)
    
    def _compute_reward(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute reward for all environments"""
        cfg = self.config
        
        # Position error
        pos_error = self.pos - self.target
        pos_cost = (pos_error ** 2).sum(dim=-1)
        
        # Velocity cost
        vel_cost = (self.vel ** 2).sum(dim=-1)
        
        # Angular velocity cost
        omega_cost = (self.omega ** 2).sum(dim=-1)
        
        # Action cost (encourage smooth control)
        action_cost = (actions ** 2).sum(dim=-1)
        
        reward = -(
            cfg.position_weight * pos_cost +
            cfg.velocity_weight * vel_cost +
            cfg.angular_velocity_weight * omega_cost +
            cfg.action_weight * action_cost
        )
        
        return reward
    
    def _check_done(self) -> torch.Tensor:
        """Check termination conditions"""
        # Out of bounds
        out_of_bounds = (self.pos.abs() > self.config.position_bound).any(dim=-1)
        
        # Timeout
        timeout = self.step_count >= self.config.max_steps
        
        return out_of_bounds | timeout
    
    def get_obs_normalization_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get observation normalization parameters for export"""
        return self.obs_mean.clone(), torch.sqrt(self.obs_var + 1e-8).clone()
