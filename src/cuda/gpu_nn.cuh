/**
 * @file gpu_nn.cuh
 * @brief Full GPU neural network implementation for PPO
 * 
 * All neural network operations run on GPU:
 * - Forward pass for all environments in parallel
 * - Action sampling with cuRAND
 * - Backward pass for PPO updates
 */

#ifndef GPU_NN_CUH
#define GPU_NN_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

namespace learning_to_fly {
namespace cuda {
namespace gpu_nn {

// Network dimensions
constexpr int OBS_DIM = 13;
constexpr int HIDDEN_DIM = 64;
constexpr int ACTION_DIM = 4;

/**
 * @brief GPU Neural Network weights structure
 */
struct GPUNetworkWeights {
    // Actor layer 1: OBS_DIM -> HIDDEN_DIM
    float* actor_w1;  // [HIDDEN_DIM, OBS_DIM]
    float* actor_b1;  // [HIDDEN_DIM]
    
    // Actor layer 2: HIDDEN_DIM -> HIDDEN_DIM
    float* actor_w2;  // [HIDDEN_DIM, HIDDEN_DIM]
    float* actor_b2;  // [HIDDEN_DIM]
    
    // Actor layer 3: HIDDEN_DIM -> ACTION_DIM
    float* actor_w3;  // [ACTION_DIM, HIDDEN_DIM]
    float* actor_b3;  // [ACTION_DIM]
    
    // Actor log_std
    float* actor_log_std;  // [ACTION_DIM]
    
    // Critic layer 1: OBS_DIM -> HIDDEN_DIM
    float* critic_w1;
    float* critic_b1;
    
    // Critic layer 2: HIDDEN_DIM -> HIDDEN_DIM
    float* critic_w2;
    float* critic_b2;
    
    // Critic layer 3: HIDDEN_DIM -> 1
    float* critic_w3;
    float* critic_b3;
    
    // Gradient buffers (same layout as weights)
    float* actor_dw1, *actor_db1;
    float* actor_dw2, *actor_db2;
    float* actor_dw3, *actor_db3;
    float* actor_dlog_std;
    float* critic_dw1, *critic_db1;
    float* critic_dw2, *critic_db2;
    float* critic_dw3, *critic_db3;
    
    // Adam optimizer state (m and v for each weight)
    float* actor_m_w1, *actor_v_w1, *actor_m_b1, *actor_v_b1;
    float* actor_m_w2, *actor_v_w2, *actor_m_b2, *actor_v_b2;
    float* actor_m_w3, *actor_v_w3, *actor_m_b3, *actor_v_b3;
    float* actor_m_log_std, *actor_v_log_std;
    float* critic_m_w1, *critic_v_w1, *critic_m_b1, *critic_v_b1;
    float* critic_m_w2, *critic_v_w2, *critic_m_b2, *critic_v_b2;
    float* critic_m_w3, *critic_v_w3, *critic_m_b3, *critic_v_b3;
};

/**
 * @brief GPU activation buffers for forward/backward pass
 */
struct GPUActivations {
    int n_envs;
    
    // Actor activations [n_envs, dim]
    float* actor_h1_pre;   // Pre-activation
    float* actor_h1;       // Post-activation
    float* actor_h2_pre;
    float* actor_h2;
    float* actor_out_pre;
    float* actor_out;      // tanh output (action mean)
    
    // Sampled actions and log probs
    float* actions;        // [n_envs, ACTION_DIM]
    float* log_probs;      // [n_envs]
    
    // Critic activations
    float* critic_h1_pre;
    float* critic_h1;
    float* critic_h2_pre;
    float* critic_h2;
    float* values;         // [n_envs]
    
    // Gradient buffers
    float* actor_dh1, *actor_dh2, *actor_dout;
    float* critic_dh1, *critic_dh2, *critic_dvalue;
};

// Device helper functions
__device__ inline float fast_tanh_d(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

__device__ inline float tanh_deriv_d(float tanh_x) {
    return 1.0f - tanh_x * tanh_x;
}

/**
 * @brief Dense layer forward pass kernel
 * Each block handles one environment, threads handle output neurons
 */
template<int IN_DIM, int OUT_DIM>
__global__ void dense_forward_tanh_kernel(
    const float* __restrict__ input,     // [n_envs, IN_DIM]
    const float* __restrict__ weights,   // [OUT_DIM, IN_DIM]
    const float* __restrict__ biases,    // [OUT_DIM]
    float* __restrict__ pre_act,         // [n_envs, OUT_DIM]
    float* __restrict__ output,          // [n_envs, OUT_DIM]
    int n_envs
) {
    int env = blockIdx.x;
    int out_idx = threadIdx.x;
    
    if (env >= n_envs || out_idx >= OUT_DIM) return;
    
    const float* in = input + env * IN_DIM;
    float sum = biases[out_idx];
    
    #pragma unroll 4
    for (int i = 0; i < IN_DIM; i++) {
        sum += weights[out_idx * IN_DIM + i] * in[i];
    }
    
    int idx = env * OUT_DIM + out_idx;
    pre_act[idx] = sum;
    output[idx] = fast_tanh_d(sum);
}

/**
 * @brief Dense layer forward (linear, no activation)
 */
template<int IN_DIM, int OUT_DIM>
__global__ void dense_forward_linear_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ biases,
    float* __restrict__ output,
    int n_envs
) {
    int env = blockIdx.x;
    int out_idx = threadIdx.x;
    
    if (env >= n_envs || out_idx >= OUT_DIM) return;
    
    const float* in = input + env * IN_DIM;
    float sum = biases[out_idx];
    
    #pragma unroll 4
    for (int i = 0; i < IN_DIM; i++) {
        sum += weights[out_idx * IN_DIM + i] * in[i];
    }
    
    output[env * OUT_DIM + out_idx] = sum;
}

/**
 * @brief Sample actions from Gaussian policy with tanh squashing
 */
__global__ void sample_actions_kernel(
    const float* __restrict__ action_mean,  // [n_envs, ACTION_DIM]
    const float* __restrict__ log_std,      // [ACTION_DIM]
    float* __restrict__ actions,            // [n_envs, ACTION_DIM]
    float* __restrict__ log_probs,          // [n_envs]
    curandState* __restrict__ rng_states,   // [n_envs]
    int n_envs
) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n_envs) return;
    
    curandState local_state = rng_states[env];
    float total_log_prob = 0;
    
    for (int a = 0; a < ACTION_DIM; a++) {
        float mean = action_mean[env * ACTION_DIM + a];
        float std = expf(log_std[a]);
        
        // Sample from standard normal
        float noise = curand_normal(&local_state);
        
        // Pre-squash action (sample from Gaussian around atanh(mean))
        float u = atanhf(fmaxf(-0.999f, fminf(0.999f, mean))) + std * noise;
        
        // Squash with tanh
        float action = tanhf(u);
        actions[env * ACTION_DIM + a] = action;
        
        // Log probability with tanh correction
        float log_p = -0.5f * noise * noise - log_std[a] - 0.5f * logf(2.0f * M_PI);
        log_p -= logf(1.0f - action * action + 1e-6f);
        total_log_prob += log_p;
    }
    
    log_probs[env] = total_log_prob;
    rng_states[env] = local_state;
}

/**
 * @brief Compute log probability of actions under current policy
 */
__global__ void compute_log_prob_kernel(
    const float* __restrict__ action_mean,  // [n_samples, ACTION_DIM]
    const float* __restrict__ log_std,      // [ACTION_DIM]
    const float* __restrict__ actions,      // [n_samples, ACTION_DIM]
    float* __restrict__ log_probs,          // [n_samples]
    int n_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;
    
    float total_log_prob = 0;
    
    for (int a = 0; a < ACTION_DIM; a++) {
        float mean = action_mean[idx * ACTION_DIM + a];
        float std = expf(log_std[a]);
        float action = actions[idx * ACTION_DIM + a];
        
        // Inverse tanh
        float action_clamped = fmaxf(-0.999f, fminf(0.999f, action));
        float mean_clamped = fmaxf(-0.999f, fminf(0.999f, mean));
        float u = atanhf(action_clamped);
        float mu = atanhf(mean_clamped);
        
        float diff = u - mu;
        float log_p = -0.5f * (diff * diff) / (std * std) - log_std[a] - 0.5f * logf(2.0f * M_PI);
        log_p -= logf(1.0f - action_clamped * action_clamped + 1e-6f);
        total_log_prob += log_p;
    }
    
    log_probs[idx] = total_log_prob;
}

/**
 * @brief Initialize cuRAND states
 */
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * @brief Compute GAE advantages on GPU
 */
__global__ void compute_gae_kernel(
    const float* __restrict__ rewards,      // [rollout_steps, n_envs]
    const float* __restrict__ values,       // [rollout_steps, n_envs]
    const float* __restrict__ dones,        // [rollout_steps, n_envs]
    const float* __restrict__ last_values,  // [n_envs]
    float* __restrict__ advantages,         // [rollout_steps, n_envs]
    float* __restrict__ returns,            // [rollout_steps, n_envs]
    float gamma,
    float gae_lambda,
    int rollout_steps,
    int n_envs
) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= n_envs) return;
    
    float gae = 0;
    float next_value = last_values[env];
    float next_done = 0;
    
    // Backward pass through time
    for (int step = rollout_steps - 1; step >= 0; step--) {
        int idx = step * n_envs + env;
        float reward = rewards[idx];
        float value = values[idx];
        float done = dones[idx];
        
        float delta = reward + gamma * next_value * (1.0f - next_done) - value;
        gae = delta + gamma * gae_lambda * (1.0f - next_done) * gae;
        
        advantages[idx] = gae;
        returns[idx] = gae + value;
        
        next_value = value;
        next_done = done;
    }
}

/**
 * @brief Normalize advantages using parallel reduction
 */
__global__ void normalize_advantages_kernel(
    float* __restrict__ advantages,
    int n,
    float mean,
    float std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    advantages[idx] = (advantages[idx] - mean) / (std + 1e-8f);
}

/**
 * @brief Compute mean using parallel reduction
 */
__global__ void reduce_sum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * @brief Compute variance using parallel reduction
 */
__global__ void reduce_variance_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float mean,
    int n
) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? (input[idx] - mean) : 0;
    sdata[tid] = val * val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * @brief PPO loss computation kernel
 * Computes clipped surrogate loss and value loss for each sample
 */
__global__ void ppo_loss_kernel(
    const float* __restrict__ new_log_probs,   // [batch_size]
    const float* __restrict__ old_log_probs,   // [batch_size]
    const float* __restrict__ advantages,      // [batch_size]
    const float* __restrict__ new_values,      // [batch_size]
    const float* __restrict__ old_values,      // [batch_size]
    const float* __restrict__ returns,         // [batch_size]
    float* __restrict__ policy_loss,           // [batch_size] (for gradient)
    float* __restrict__ value_loss,            // [batch_size]
    float* __restrict__ ratio_out,             // [batch_size]
    float clip_epsilon,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float ratio = expf(new_log_probs[idx] - old_log_probs[idx]);
    float clipped_ratio = fmaxf(1.0f - clip_epsilon, fminf(1.0f + clip_epsilon, ratio));
    
    float adv = advantages[idx];
    float surr1 = ratio * adv;
    float surr2 = clipped_ratio * adv;
    
    // Policy loss (negated for gradient ascent)
    policy_loss[idx] = -fminf(surr1, surr2);
    ratio_out[idx] = ratio;
    
    // Value loss (clipped)
    float new_val = new_values[idx];
    float old_val = old_values[idx];
    float ret = returns[idx];
    
    float val_clipped = old_val + fmaxf(-clip_epsilon, fminf(clip_epsilon, new_val - old_val));
    float vl1 = (new_val - ret) * (new_val - ret);
    float vl2 = (val_clipped - ret) * (val_clipped - ret);
    value_loss[idx] = 0.5f * fmaxf(vl1, vl2);
}

/**
 * @brief Adam optimizer update kernel
 */
__global__ void adam_update_kernel(
    float* __restrict__ param,
    float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float bc1,  // bias correction 1
    float bc2,  // bias correction 2
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float g = grad[idx];
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;
    
    m[idx] = m_new;
    v[idx] = v_new;
    
    float m_hat = m_new / bc1;
    float v_hat = v_new / bc2;
    
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
    grad[idx] = 0;  // Zero gradient
}

/**
 * @brief GPU Network Manager
 * Handles allocation, initialization, and operations
 */
class GPUNetworkManager {
public:
    GPUNetworkWeights weights;
    GPUActivations acts;
    curandState* rng_states;
    
    int n_envs;
    int adam_step;
    bool initialized;
    
    cudaStream_t stream;
    
    // Host copies for checkpointing
    float h_actor_w1[HIDDEN_DIM * OBS_DIM];
    float h_actor_b1[HIDDEN_DIM];
    float h_actor_w2[HIDDEN_DIM * HIDDEN_DIM];
    float h_actor_b2[HIDDEN_DIM];
    float h_actor_w3[ACTION_DIM * HIDDEN_DIM];
    float h_actor_b3[ACTION_DIM];
    float h_actor_log_std[ACTION_DIM];
    
    GPUNetworkManager() : initialized(false), adam_step(0) {}
    
    void allocate_weights() {
        // Actor weights
        cudaMalloc(&weights.actor_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.actor_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_w3, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_b3, ACTION_DIM * sizeof(float));
        cudaMalloc(&weights.actor_log_std, ACTION_DIM * sizeof(float));
        
        // Critic weights
        cudaMalloc(&weights.critic_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.critic_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_w3, 1 * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_b3, 1 * sizeof(float));
        
        // Gradient buffers
        cudaMalloc(&weights.actor_dw1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.actor_db1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_dw2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_db2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_dw3, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_db3, ACTION_DIM * sizeof(float));
        cudaMalloc(&weights.actor_dlog_std, ACTION_DIM * sizeof(float));
        
        cudaMalloc(&weights.critic_dw1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.critic_db1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_dw2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_db2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_dw3, 1 * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_db3, 1 * sizeof(float));
        
        // Adam state (m and v for each weight/gradient pair)
        cudaMalloc(&weights.actor_m_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.actor_m_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_m_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_m_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_m_w3, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_w3, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.actor_m_b3, ACTION_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_b3, ACTION_DIM * sizeof(float));
        cudaMalloc(&weights.actor_m_log_std, ACTION_DIM * sizeof(float));
        cudaMalloc(&weights.actor_v_log_std, ACTION_DIM * sizeof(float));
        
        cudaMalloc(&weights.critic_m_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.critic_v_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&weights.critic_m_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_v_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_m_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_v_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_m_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_v_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_m_w3, 1 * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_v_w3, 1 * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&weights.critic_m_b3, 1 * sizeof(float));
        cudaMalloc(&weights.critic_v_b3, 1 * sizeof(float));
    }
    
    void allocate_activations(int n_envs_) {
        n_envs = n_envs_;
        
        cudaMalloc(&acts.actor_h1_pre, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.actor_h1, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.actor_h2_pre, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.actor_h2, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.actor_out_pre, n_envs * ACTION_DIM * sizeof(float));
        cudaMalloc(&acts.actor_out, n_envs * ACTION_DIM * sizeof(float));
        cudaMalloc(&acts.actions, n_envs * ACTION_DIM * sizeof(float));
        cudaMalloc(&acts.log_probs, n_envs * sizeof(float));
        
        cudaMalloc(&acts.critic_h1_pre, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.critic_h1, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.critic_h2_pre, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.critic_h2, n_envs * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&acts.values, n_envs * sizeof(float));
        
        // RNG states
        cudaMalloc(&rng_states, n_envs * sizeof(curandState));
    }
    
    void init(int n_envs_, unsigned int seed) {
        cudaStreamCreate(&stream);
        
        allocate_weights();
        allocate_activations(n_envs_);
        
        // Initialize weights with Xavier
        std::mt19937 rng(seed);
        
        // Actor layer 1
        float std1 = std::sqrt(2.0f / (OBS_DIM + HIDDEN_DIM));
        std::normal_distribution<float> dist1(0, std1);
        for (int i = 0; i < HIDDEN_DIM * OBS_DIM; i++) h_actor_w1[i] = dist1(rng);
        for (int i = 0; i < HIDDEN_DIM; i++) h_actor_b1[i] = 0;
        
        // Actor layer 2
        float std2 = std::sqrt(2.0f / (HIDDEN_DIM + HIDDEN_DIM));
        std::normal_distribution<float> dist2(0, std2);
        for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++) h_actor_w2[i] = dist2(rng);
        for (int i = 0; i < HIDDEN_DIM; i++) h_actor_b2[i] = 0;
        
        // Actor layer 3
        float std3 = std::sqrt(2.0f / (HIDDEN_DIM + ACTION_DIM));
        std::normal_distribution<float> dist3(0, std3);
        for (int i = 0; i < ACTION_DIM * HIDDEN_DIM; i++) h_actor_w3[i] = dist3(rng);
        for (int i = 0; i < ACTION_DIM; i++) h_actor_b3[i] = 0;
        
        // Log std
        for (int i = 0; i < ACTION_DIM; i++) h_actor_log_std[i] = -1.0f;
        
        // Copy to device
        cudaMemcpy(weights.actor_w1, h_actor_w1, HIDDEN_DIM * OBS_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights.actor_b1, h_actor_b1, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights.actor_w2, h_actor_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights.actor_b2, h_actor_b2, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights.actor_w3, h_actor_w3, ACTION_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights.actor_b3, h_actor_b3, ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(weights.actor_log_std, h_actor_log_std, ACTION_DIM * sizeof(float), cudaMemcpyHostToDevice);
        
        // Initialize critic similarly (reuse host buffers)
        for (int i = 0; i < HIDDEN_DIM * OBS_DIM; i++) h_actor_w1[i] = dist1(rng);
        cudaMemcpy(weights.critic_w1, h_actor_w1, HIDDEN_DIM * OBS_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(weights.critic_b1, 0, HIDDEN_DIM * sizeof(float));
        
        for (int i = 0; i < HIDDEN_DIM * HIDDEN_DIM; i++) h_actor_w2[i] = dist2(rng);
        cudaMemcpy(weights.critic_w2, h_actor_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(weights.critic_b2, 0, HIDDEN_DIM * sizeof(float));
        
        float h_critic_w3[HIDDEN_DIM];
        float std_c3 = std::sqrt(2.0f / (HIDDEN_DIM + 1));
        std::normal_distribution<float> dist_c3(0, std_c3);
        for (int i = 0; i < HIDDEN_DIM; i++) h_critic_w3[i] = dist_c3(rng);
        cudaMemcpy(weights.critic_w3, h_critic_w3, HIDDEN_DIM * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(weights.critic_b3, 0, sizeof(float));
        
        // Zero Adam state
        cudaMemset(weights.actor_m_w1, 0, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMemset(weights.actor_v_w1, 0, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMemset(weights.actor_m_b1, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_v_b1, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_m_w2, 0, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_v_w2, 0, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_m_b2, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_v_b2, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_m_w3, 0, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_v_w3, 0, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_m_b3, 0, ACTION_DIM * sizeof(float));
        cudaMemset(weights.actor_v_b3, 0, ACTION_DIM * sizeof(float));
        cudaMemset(weights.actor_m_log_std, 0, ACTION_DIM * sizeof(float));
        cudaMemset(weights.actor_v_log_std, 0, ACTION_DIM * sizeof(float));
        
        cudaMemset(weights.critic_m_w1, 0, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMemset(weights.critic_v_w1, 0, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMemset(weights.critic_m_b1, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_v_b1, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_m_w2, 0, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_v_w2, 0, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_m_b2, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_v_b2, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_m_w3, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_v_w3, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_m_b3, 0, sizeof(float));
        cudaMemset(weights.critic_v_b3, 0, sizeof(float));
        
        // Zero gradients
        cudaMemset(weights.actor_dw1, 0, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMemset(weights.actor_db1, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_dw2, 0, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_db2, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_dw3, 0, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.actor_db3, 0, ACTION_DIM * sizeof(float));
        cudaMemset(weights.actor_dlog_std, 0, ACTION_DIM * sizeof(float));
        cudaMemset(weights.critic_dw1, 0, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMemset(weights.critic_db1, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_dw2, 0, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_db2, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_dw3, 0, HIDDEN_DIM * sizeof(float));
        cudaMemset(weights.critic_db3, 0, sizeof(float));
        
        // Initialize cuRAND
        int blocks = (n_envs + 255) / 256;
        init_curand_kernel<<<blocks, 256, 0, stream>>>(rng_states, seed, n_envs);
        
        cudaStreamSynchronize(stream);
        initialized = true;
    }
    
    void actor_forward(const float* d_obs, int batch_size) {
        // Layer 1
        dense_forward_tanh_kernel<OBS_DIM, HIDDEN_DIM><<<batch_size, HIDDEN_DIM, 0, stream>>>(
            d_obs, weights.actor_w1, weights.actor_b1,
            acts.actor_h1_pre, acts.actor_h1, batch_size);
        
        // Layer 2
        dense_forward_tanh_kernel<HIDDEN_DIM, HIDDEN_DIM><<<batch_size, HIDDEN_DIM, 0, stream>>>(
            acts.actor_h1, weights.actor_w2, weights.actor_b2,
            acts.actor_h2_pre, acts.actor_h2, batch_size);
        
        // Layer 3 (tanh output)
        dense_forward_tanh_kernel<HIDDEN_DIM, ACTION_DIM><<<batch_size, ACTION_DIM, 0, stream>>>(
            acts.actor_h2, weights.actor_w3, weights.actor_b3,
            acts.actor_out_pre, acts.actor_out, batch_size);
    }
    
    void critic_forward(const float* d_obs, int batch_size) {
        // Layer 1
        dense_forward_tanh_kernel<OBS_DIM, HIDDEN_DIM><<<batch_size, HIDDEN_DIM, 0, stream>>>(
            d_obs, weights.critic_w1, weights.critic_b1,
            acts.critic_h1_pre, acts.critic_h1, batch_size);
        
        // Layer 2
        dense_forward_tanh_kernel<HIDDEN_DIM, HIDDEN_DIM><<<batch_size, HIDDEN_DIM, 0, stream>>>(
            acts.critic_h1, weights.critic_w2, weights.critic_b2,
            acts.critic_h2_pre, acts.critic_h2, batch_size);
        
        // Layer 3 (linear output)
        dense_forward_linear_kernel<HIDDEN_DIM, 1><<<batch_size, 1, 0, stream>>>(
            acts.critic_h2, weights.critic_w3, weights.critic_b3,
            acts.values, batch_size);
    }
    
    void sample_actions(int batch_size) {
        int blocks = (batch_size + 255) / 256;
        sample_actions_kernel<<<blocks, 256, 0, stream>>>(
            acts.actor_out, weights.actor_log_std,
            acts.actions, acts.log_probs,
            rng_states, batch_size);
    }
    
    void copy_weights_to_host() {
        cudaMemcpy(h_actor_w1, weights.actor_w1, HIDDEN_DIM * OBS_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_actor_b1, weights.actor_b1, HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_actor_w2, weights.actor_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_actor_b2, weights.actor_b2, HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_actor_w3, weights.actor_w3, ACTION_DIM * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_actor_b3, weights.actor_b3, ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_actor_log_std, weights.actor_log_std, ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    void cleanup() {
        if (!initialized) return;
        
        // Free all allocated memory
        cudaFree(weights.actor_w1); cudaFree(weights.actor_b1);
        cudaFree(weights.actor_w2); cudaFree(weights.actor_b2);
        cudaFree(weights.actor_w3); cudaFree(weights.actor_b3);
        cudaFree(weights.actor_log_std);
        cudaFree(weights.critic_w1); cudaFree(weights.critic_b1);
        cudaFree(weights.critic_w2); cudaFree(weights.critic_b2);
        cudaFree(weights.critic_w3); cudaFree(weights.critic_b3);
        
        cudaFree(weights.actor_dw1); cudaFree(weights.actor_db1);
        cudaFree(weights.actor_dw2); cudaFree(weights.actor_db2);
        cudaFree(weights.actor_dw3); cudaFree(weights.actor_db3);
        cudaFree(weights.actor_dlog_std);
        cudaFree(weights.critic_dw1); cudaFree(weights.critic_db1);
        cudaFree(weights.critic_dw2); cudaFree(weights.critic_db2);
        cudaFree(weights.critic_dw3); cudaFree(weights.critic_db3);
        
        cudaFree(weights.actor_m_w1); cudaFree(weights.actor_v_w1);
        cudaFree(weights.actor_m_b1); cudaFree(weights.actor_v_b1);
        cudaFree(weights.actor_m_w2); cudaFree(weights.actor_v_w2);
        cudaFree(weights.actor_m_b2); cudaFree(weights.actor_v_b2);
        cudaFree(weights.actor_m_w3); cudaFree(weights.actor_v_w3);
        cudaFree(weights.actor_m_b3); cudaFree(weights.actor_v_b3);
        cudaFree(weights.actor_m_log_std); cudaFree(weights.actor_v_log_std);
        cudaFree(weights.critic_m_w1); cudaFree(weights.critic_v_w1);
        cudaFree(weights.critic_m_b1); cudaFree(weights.critic_v_b1);
        cudaFree(weights.critic_m_w2); cudaFree(weights.critic_v_w2);
        cudaFree(weights.critic_m_b2); cudaFree(weights.critic_v_b2);
        cudaFree(weights.critic_m_w3); cudaFree(weights.critic_v_w3);
        cudaFree(weights.critic_m_b3); cudaFree(weights.critic_v_b3);
        
        cudaFree(acts.actor_h1_pre); cudaFree(acts.actor_h1);
        cudaFree(acts.actor_h2_pre); cudaFree(acts.actor_h2);
        cudaFree(acts.actor_out_pre); cudaFree(acts.actor_out);
        cudaFree(acts.actions); cudaFree(acts.log_probs);
        cudaFree(acts.critic_h1_pre); cudaFree(acts.critic_h1);
        cudaFree(acts.critic_h2_pre); cudaFree(acts.critic_h2);
        cudaFree(acts.values);
        
        cudaFree(rng_states);
        cudaStreamDestroy(stream);
        
        initialized = false;
    }
};

} // namespace gpu_nn
} // namespace cuda
} // namespace learning_to_fly

#endif // GPU_NN_CUH
