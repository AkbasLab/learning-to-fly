/**
 * @file gpu_rollout.cuh
 * @brief GPU-accelerated rollout collection for PPO
 * 
 * This header provides CUDA kernels for parallel environment stepping
 * and integrates with the PPO training loop.
 */

#ifndef GPU_ROLLOUT_CUH
#define GPU_ROLLOUT_CUH

#include <cuda_runtime.h>

namespace learning_to_fly {
namespace cuda {

using T = float;
using TI = unsigned int;

template <TI T_N_ENVS, TI T_THREADS_PER_BLOCK>
struct GPURolloutSpec {
    static constexpr TI N_ENVS = T_N_ENVS;
    static constexpr TI THREADS_PER_BLOCK = T_THREADS_PER_BLOCK;
    static constexpr TI N_BLOCKS = (N_ENVS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
};

// Simplified multirotor state for GPU
struct GPUQuadState {
    T position[3];
    T orientation[4];  // quaternion (w, x, y, z)
    T linear_velocity[3];
    T angular_velocity[3];
};

// Quadrotor physical parameters
struct GPUQuadParams {
    T mass;
    T arm_length;
    T thrust_coeff;
    T torque_coeff;
    T inertia[3];  // Diagonal inertia matrix
    T gravity;
    T dt;
    T max_rpm;
};

/**
 * @brief Quaternion multiplication helper
 */
__device__ inline void quat_mult(const T* q1, const T* q2, T* out) {
    out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

/**
 * @brief Normalize quaternion
 */
__device__ inline void quat_normalize(T* q) {
    T norm = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (norm > 1e-8f) {
        q[0] /= norm;
        q[1] /= norm;
        q[2] /= norm;
        q[3] /= norm;
    }
}

/**
 * @brief Rotate vector by quaternion
 */
__device__ inline void quat_rotate(const T* q, const T* v, T* out) {
    // v' = q * v * q^-1
    T qv[4] = {0, v[0], v[1], v[2]};
    T q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
    T temp[4], result[4];
    quat_mult(q, qv, temp);
    quat_mult(temp, q_conj, result);
    out[0] = result[1];
    out[1] = result[2];
    out[2] = result[3];
}

/**
 * @brief CUDA kernel for parallel quadrotor simulation
 * 
 * Implements simplified quadrotor dynamics with Euler integration.
 * Good enough for PPO training, much faster than full RK4.
 */
__global__ void step_quadrotors_kernel(
    const GPUQuadParams params,
    GPUQuadState* states,
    const T* actions,       // [N_ENVS, 4] normalized motor commands in [-1,1]
    T* observations,        // [N_ENVS, OBS_DIM] output
    T* rewards,             // [N_ENVS] output
    bool* dones,            // [N_ENVS] output
    TI n_envs,
    TI obs_dim
) {
    TI idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_envs) return;
    
    GPUQuadState& state = states[idx];
    
    // Get actions for this environment (normalized [-1,1])
    T motor_cmds[4];
    for (int i = 0; i < 4; i++) {
        motor_cmds[i] = actions[idx * 4 + i];
    }
    
    // Convert to RPM (0 to max_rpm)
    T rpm[4];
    for (int i = 0; i < 4; i++) {
        T normalized = (motor_cmds[i] + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
        rpm[i] = normalized * params.max_rpm;
    }
    
    // Compute motor forces and torques
    T forces[4];
    T torques[4];
    for (int i = 0; i < 4; i++) {
        T rpm_sq = rpm[i] * rpm[i];
        forces[i] = params.thrust_coeff * rpm_sq;
        torques[i] = params.torque_coeff * rpm_sq * ((i % 2 == 0) ? 1.0f : -1.0f);  // Alternating directions
    }
    
    // Total thrust (in body frame, along z)
    T total_thrust = forces[0] + forces[1] + forces[2] + forces[3];
    
    // Body torques (simplified X-config)
    T L = params.arm_length;
    T tau_x = L * (forces[0] - forces[1] - forces[2] + forces[3]) * 0.707f;  // Roll
    T tau_y = L * (forces[0] + forces[1] - forces[2] - forces[3]) * 0.707f;  // Pitch
    T tau_z = torques[0] + torques[1] + torques[2] + torques[3];             // Yaw
    
    // Body-frame thrust vector
    T thrust_body[3] = {0, 0, total_thrust / params.mass};
    
    // Rotate thrust to world frame
    T thrust_world[3];
    quat_rotate(state.orientation, thrust_body, thrust_world);
    
    // Gravity in world frame
    T gravity_world[3] = {0, 0, -params.gravity};
    
    // Linear acceleration in world frame
    T lin_acc[3];
    for (int i = 0; i < 3; i++) {
        lin_acc[i] = thrust_world[i] + gravity_world[i];
    }
    
    // Angular acceleration in body frame (Euler equations, simplified diagonal inertia)
    T ang_acc[3];
    ang_acc[0] = tau_x / params.inertia[0];
    ang_acc[1] = tau_y / params.inertia[1];
    ang_acc[2] = tau_z / params.inertia[2];
    
    // Euler integration
    T dt = params.dt;
    
    // Update linear velocity and position
    for (int i = 0; i < 3; i++) {
        state.linear_velocity[i] += lin_acc[i] * dt;
        state.position[i] += state.linear_velocity[i] * dt;
    }
    
    // Update angular velocity
    for (int i = 0; i < 3; i++) {
        state.angular_velocity[i] += ang_acc[i] * dt;
    }
    
    // Update orientation (quaternion integration)
    T omega[4] = {0, state.angular_velocity[0], state.angular_velocity[1], state.angular_velocity[2]};
    T q_dot[4];
    quat_mult(state.orientation, omega, q_dot);
    for (int i = 0; i < 4; i++) {
        state.orientation[i] += 0.5f * q_dot[i] * dt;
    }
    quat_normalize(state.orientation);
    
    // Compute observation
    TI obs_idx = idx * obs_dim;
    observations[obs_idx + 0] = state.position[0];
    observations[obs_idx + 1] = state.position[1];
    observations[obs_idx + 2] = state.position[2];
    observations[obs_idx + 3] = state.orientation[0];
    observations[obs_idx + 4] = state.orientation[1];
    observations[obs_idx + 5] = state.orientation[2];
    observations[obs_idx + 6] = state.orientation[3];
    observations[obs_idx + 7] = state.linear_velocity[0];
    observations[obs_idx + 8] = state.linear_velocity[1];
    observations[obs_idx + 9] = state.linear_velocity[2];
    observations[obs_idx + 10] = state.angular_velocity[0];
    observations[obs_idx + 11] = state.angular_velocity[1];
    observations[obs_idx + 12] = state.angular_velocity[2];
    
    // Compute reward (hover at origin)
    T dist_sq = state.position[0] * state.position[0] +
                state.position[1] * state.position[1] +
                state.position[2] * state.position[2];
    T dist = sqrtf(dist_sq);
    
    T vel_sq = state.linear_velocity[0] * state.linear_velocity[0] +
               state.linear_velocity[1] * state.linear_velocity[1] +
               state.linear_velocity[2] * state.linear_velocity[2];
    
    // Upright bonus (w component of quaternion close to 1 means upright)
    T upright = state.orientation[0] * state.orientation[0];
    
    rewards[idx] = -dist - 0.1f * sqrtf(vel_sq) - 0.5f * (1.0f - upright);
    
    // Termination conditions
    bool terminated = false;
    if (dist > 3.0f) terminated = true;  // Too far from origin
    if (upright < 0.3f) terminated = true;  // Flipped
    if (fabsf(state.position[2]) > 2.0f) terminated = true;  // Too high/low
    
    dones[idx] = terminated;
}

/**
 * @brief GPU Rollout Manager
 * 
 * Manages device memory and kernel launches for parallel rollout collection.
 */
template <typename CONFIG>
class GPURolloutManager {
public:
    static constexpr TI N_ENVS = CONFIG::N_ENVIRONMENTS;
    static constexpr TI ACTION_DIM = CONFIG::ACTION_DIM;
    static constexpr TI OBS_DIM = CONFIG::OBSERVATION_DIM;
    static constexpr TI THREADS_PER_BLOCK = 64;
    static constexpr TI N_BLOCKS = (N_ENVS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
private:
    // Device memory
    GPUQuadState* d_states = nullptr;
    T* d_actions = nullptr;
    T* d_observations = nullptr;
    T* d_rewards = nullptr;
    bool* d_dones = nullptr;
    
    // Host memory for results
    T* h_observations = nullptr;
    T* h_rewards = nullptr;
    bool* h_dones = nullptr;
    
    // Parameters (copied to device constant memory not needed, passed by value)
    GPUQuadParams params;
    
    bool initialized = false;
    cudaStream_t stream;
    
public:
    template<typename ENVIRONMENT>
    void init(const ENVIRONMENT* host_envs) {
        // Create CUDA stream
        cudaStreamCreate(&stream);
        
        // Set up Crazyflie parameters
        params.mass = 0.027f;           // 27 grams
        params.arm_length = 0.046f;     // 46mm arm length
        params.thrust_coeff = 1.0e-7f;  // Thrust coefficient
        params.torque_coeff = 1.0e-9f;  // Torque coefficient
        params.inertia[0] = 1.4e-5f;    // Ixx
        params.inertia[1] = 1.4e-5f;    // Iyy
        params.inertia[2] = 2.17e-5f;   // Izz
        params.gravity = 9.81f;
        params.dt = 0.01f;              // 100Hz
        params.max_rpm = 21702.0f;
        
        // Allocate device memory
        cudaMalloc(&d_states, N_ENVS * sizeof(GPUQuadState));
        cudaMalloc(&d_actions, N_ENVS * ACTION_DIM * sizeof(T));
        cudaMalloc(&d_observations, N_ENVS * OBS_DIM * sizeof(T));
        cudaMalloc(&d_rewards, N_ENVS * sizeof(T));
        cudaMalloc(&d_dones, N_ENVS * sizeof(bool));
        
        // Allocate host memory for results
        h_observations = new T[N_ENVS * OBS_DIM];
        h_rewards = new T[N_ENVS];
        h_dones = new bool[N_ENVS];
        
        // Initialize states (all at origin, upright)
        GPUQuadState* h_states = new GPUQuadState[N_ENVS];
        for (TI i = 0; i < N_ENVS; i++) {
            for (int j = 0; j < 3; j++) {
                h_states[i].position[j] = 0;
                h_states[i].linear_velocity[j] = 0;
                h_states[i].angular_velocity[j] = 0;
            }
            h_states[i].orientation[0] = 1.0f;  // w
            h_states[i].orientation[1] = 0;     // x
            h_states[i].orientation[2] = 0;     // y
            h_states[i].orientation[3] = 0;     // z
        }
        cudaMemcpy(d_states, h_states, N_ENVS * sizeof(GPUQuadState), cudaMemcpyHostToDevice);
        delete[] h_states;
        
        initialized = true;
    }
    
    void step(const T* host_actions) {
        if (!initialized) return;
        
        // Copy actions to device
        cudaMemcpyAsync(d_actions, host_actions, N_ENVS * ACTION_DIM * sizeof(T), 
                        cudaMemcpyHostToDevice, stream);
        
        // Launch kernel
        dim3 grid(N_BLOCKS);
        dim3 block(THREADS_PER_BLOCK);
        
        step_quadrotors_kernel<<<grid, block, 0, stream>>>(
            params,
            d_states,
            d_actions, d_observations, d_rewards, d_dones,
            N_ENVS, OBS_DIM
        );
        
        // Copy results back
        cudaMemcpyAsync(h_observations, d_observations, N_ENVS * OBS_DIM * sizeof(T),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_rewards, d_rewards, N_ENVS * sizeof(T),
                        cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_dones, d_dones, N_ENVS * sizeof(bool),
                        cudaMemcpyDeviceToHost, stream);
        
        // Sync
        cudaStreamSynchronize(stream);
    }
    
    const T* get_observations() const { return h_observations; }
    const T* get_rewards() const { return h_rewards; }
    const bool* get_dones() const { return h_dones; }
    
    void reset_env(TI env_idx) {
        GPUQuadState initial_state;
        for (int j = 0; j < 3; j++) {
            initial_state.position[j] = 0;
            initial_state.linear_velocity[j] = 0;
            initial_state.angular_velocity[j] = 0;
        }
        initial_state.orientation[0] = 1.0f;
        initial_state.orientation[1] = 0;
        initial_state.orientation[2] = 0;
        initial_state.orientation[3] = 0;
        
        cudaMemcpy(d_states + env_idx, &initial_state, sizeof(GPUQuadState), cudaMemcpyHostToDevice);
    }
    
    void cleanup() {
        if (!initialized) return;
        
        cudaFree(d_states);
        cudaFree(d_actions);
        cudaFree(d_observations);
        cudaFree(d_rewards);
        cudaFree(d_dones);
        
        delete[] h_observations;
        delete[] h_rewards;
        delete[] h_dones;
        
        cudaStreamDestroy(stream);
        initialized = false;
    }
    
    ~GPURolloutManager() {
        cleanup();
    }
};

} // namespace cuda
} // namespace learning_to_fly

#endif // GPU_ROLLOUT_CUH
