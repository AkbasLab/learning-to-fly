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
    """Quadrotor physical parameters (Crazyflie 2.1) - matching rl_tools"""
    mass: float = 0.027  # kg
    arm_length: float = 0.028  # m (from rl_tools)
    
    # Thrust model: thrust = thrust_coeff * rpm^2
    # Action in [-1, 1] maps to RPM in [0, max_rpm]
    max_rpm: float = 21702.0
    thrust_coeff: float = 3.16e-10  # N per rpm^2
    
    # Inertia tensor (from rl_tools crazy_flie.h)
    Jxx: float = 3.85e-6
    Jyy: float = 3.85e-6
    Jzz: float = 5.9675e-6
    
    # Torque constant
    torque_constant: float = 0.005964552
    
    # Motor dynamics
    motor_time_constant: float = 0.15  # seconds (from rl_tools)
    
    # Physics
    gravity: float = 9.81
    dt: float = 0.01
    substeps: int = 4
    
    # Episode
    max_steps: int = 500
    position_bound: float = 5.0
    
    # Takeoff and hover target
    target_height: float = 1.0  # meters above ground
    init_height_range: float = 0.02  # small random variation at start
    init_height_min: float = 0.05  # minimum starting height (avoid ground contact)
    xy_spawn_range: float = 0.1  # spawn near origin in x/y
    
    # Reward weights - SCALED FOR TAKEOFF LEARNING
    # Key insight: reward must be positive when doing the right thing
    # Position cost is SMALL so altitude bonus dominates early learning
    position_weight: float = 0.1  # Reduced - don't overwhelm altitude bonus
    velocity_weight: float = 0.01
    angular_velocity_weight: float = 0.001
    action_weight: float = 0.0
    termination_penalty: float = 10.0
    
    # Takeoff reward shaping - LARGE bonuses to encourage altitude gain
    altitude_bonus_scale: float = 2.0  # Primary reward: gain altitude!
    hover_bonus: float = 5.0  # Large bonus for reaching target
    hover_radius: float = 0.3  # Within 30cm counts as hovering
    upright_bonus: float = 0.5  # Bonus for staying upright
    velocity_penalty_at_hover: float = 0.5  # Penalize velocity when near target


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
        cfg = self.config
        
        # Start on the ground with small random variation
        # x, y near origin, z near ground level but above contact
        xy_pos = (torch.rand(n, 2, device=self.device) * 2 - 1) * cfg.xy_spawn_range
        z_pos = cfg.init_height_min + torch.rand(n, 1, device=self.device) * cfg.init_height_range
        self.pos[env_ids] = torch.cat([xy_pos, z_pos], dim=-1)
        
        # Identity quaternion (upright)
        self.quat[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device).expand(n, 4)
        
        # Zero velocity (starting from rest on ground)
        self.vel[env_ids] = 0.0
        self.omega[env_ids] = 0.0
        
        # Motors off initially (on the ground)
        self.motors[env_ids] = 0.0
        
        # Zero step count
        self.step_count[env_ids] = 0
        self.episode_return[env_ids] = 0
        
        # Target: hover at specified height above ground, centered at origin
        self.target[env_ids, 0] = 0.0  # x = 0
        self.target[env_ids, 1] = 0.0  # y = 0
        self.target[env_ids, 2] = cfg.target_height  # z = target height
        
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
        """Single physics substep - matching rl_tools physics"""
        # Low-pass filter motor response (motor_cmd is normalized 0-1)
        self.motors = self.motors + self.motor_alpha * (motor_cmd - self.motors)
        
        # Convert to RPM and compute thrust
        # motors is in [0, 1], map to [0, max_rpm]
        rpm = self.motors * self.config.max_rpm
        
        # Thrust per motor: thrust = thrust_coeff * rpm^2
        thrusts = self.config.thrust_coeff * (rpm ** 2)  # [n, 4] in Newtons
        total_thrust = thrusts.sum(dim=-1)  # [n] total thrust
        
        # Thrust acceleration in body frame (always +Z body axis)
        thrust_acc_body = torch.zeros_like(self.pos)
        thrust_acc_body[:, 2] = total_thrust / self.mass
        
        # Rotate to world frame
        thrust_acc_world = self._rotate_vector(self.quat, thrust_acc_body)
        
        # Linear acceleration (thrust + gravity)
        acc = thrust_acc_world + self.gravity
        
        # Torque computation
        # Roll/pitch from differential thrust at arm positions
        # Yaw from motor torque (alternating direction)
        L = self.config.arm_length
        tc = self.config.torque_constant
        
        # Motor layout (X config from rl_tools):
        # 0: (+L, -L) front-right, CCW (-1 torque)
        # 1: (-L, -L) back-right, CW (+1 torque) 
        # 2: (-L, +L) back-left, CCW (-1 torque)
        # 3: (+L, +L) front-left, CW (+1 torque)
        
        # Roll torque (about x-axis, from y-offset thrust)
        roll_torque = L * (thrusts[:, 0] - thrusts[:, 1] - thrusts[:, 2] + thrusts[:, 3])
        # Pitch torque (about y-axis, from x-offset thrust)
        pitch_torque = L * (thrusts[:, 0] + thrusts[:, 1] - thrusts[:, 2] - thrusts[:, 3])
        # Yaw torque (from motor drag, alternating)
        yaw_torque = tc * (-thrusts[:, 0] + thrusts[:, 1] - thrusts[:, 2] + thrusts[:, 3])
        
        torques = torch.stack([roll_torque, pitch_torque, yaw_torque], dim=-1)
        alpha = torques * self.J_inv  # Angular acceleration
        
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
        """
        Compute reward optimized for takeoff learning.
        
        Key insight: Early in training, the drone needs to learn:
        1. Thrust all motors = go up (most important!)
        2. Stay upright
        3. Reach target height
        4. Stabilize at target
        
        Reward is designed so that even random exploration that produces
        altitude gain gets positive reward, creating a clear gradient.
        """
        cfg = self.config
        
        # Current altitude (height above ground)
        altitude = self.pos[:, 2].clamp(min=0)
        
        # Distance to target (3D)
        pos_error = self.pos - self.target
        pos_dist = torch.norm(pos_error, dim=-1)
        
        # === PRIMARY REWARD: ALTITUDE PROGRESS ===
        # This is the most important signal for takeoff
        # Reward scales from 0 (ground) to altitude_bonus_scale (at target height)
        altitude_normalized = (altitude / cfg.target_height).clamp(max=1.5)
        altitude_reward = cfg.altitude_bonus_scale * altitude_normalized
        
        # === HOVER REWARD: Reached target zone ===
        # Large bonus for being within hover_radius of target
        in_hover_zone = pos_dist < cfg.hover_radius
        hover_reward = torch.where(
            in_hover_zone,
            cfg.hover_bonus * (1.0 - pos_dist / cfg.hover_radius),
            torch.zeros_like(pos_dist)
        )
        
        # === STABILITY REWARDS ===
        # Upright bonus (quat w component = 1 means upright)
        upright = self.quat[:, 0].abs()
        upright_reward = cfg.upright_bonus * upright
        
        # Velocity penalty when hovering (encourage stillness at target)
        vel_magnitude = torch.norm(self.vel, dim=-1)
        hover_vel_penalty = torch.where(
            in_hover_zone,
            cfg.velocity_penalty_at_hover * vel_magnitude,
            torch.zeros_like(vel_magnitude)
        )
        
        # === SMALL COSTS (don't overwhelm altitude learning) ===
        # XY position error (small penalty for drifting laterally)
        xy_error = (pos_error[:, :2] ** 2).sum(dim=-1)
        xy_cost = cfg.position_weight * xy_error * 0.1  # Very small
        
        # Angular velocity cost (small)
        omega_cost = cfg.angular_velocity_weight * (self.omega ** 2).sum(dim=-1)
        
        # === COMBINE ===
        reward = (
            altitude_reward +      # Go up! (main signal)
            hover_reward +         # Stay at target
            upright_reward -       # Stay level
            hover_vel_penalty -    # Be still when hovering
            xy_cost -              # Don't drift
            omega_cost             # Don't spin
        )
        
        return reward
    
    def _check_done(self) -> torch.Tensor:
        """Check termination conditions"""
        # Out of bounds (x, y)
        xy_out_of_bounds = (self.pos[:, :2].abs() > self.config.position_bound).any(dim=-1)
        
        # Too high
        too_high = self.pos[:, 2] > self.config.position_bound
        
        # Ground collision (actual crash, not just settling)
        # Drone starts at ~0.05m, so z < 0 means it crashed
        ground_collision = self.pos[:, 2] < 0.0
        
        # Timeout
        timeout = self.step_count >= self.config.max_steps
        
        return xy_out_of_bounds | too_high | ground_collision | timeout
    
    def get_obs_normalization_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get observation normalization parameters for export"""
        return self.obs_mean.clone(), torch.sqrt(self.obs_var + 1e-8).clone()
