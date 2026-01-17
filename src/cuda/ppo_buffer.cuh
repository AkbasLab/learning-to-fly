/**
 * @file ppo_buffer.cuh
 * @brief Rollout buffer and GAE computation for PPO
 */

#ifndef PPO_BUFFER_CUH
#define PPO_BUFFER_CUH

#include <cstring>
#include <cmath>
#include <algorithm>

namespace learning_to_fly {
namespace cuda {

/**
 * @brief Rollout buffer for PPO
 * Stores trajectories collected during rollout phase
 */
template<int N_ENVS, int ROLLOUT_STEPS, int OBS_DIM, int ACTION_DIM>
struct RolloutBuffer {
    static constexpr int CAPACITY = N_ENVS * ROLLOUT_STEPS;
    
    // Trajectory data
    float observations[CAPACITY][OBS_DIM];
    float actions[CAPACITY][ACTION_DIM];
    float log_probs[CAPACITY];
    float values[CAPACITY];
    float rewards[CAPACITY];
    float dones[CAPACITY];
    
    // Computed advantages and returns
    float advantages[CAPACITY];
    float returns[CAPACITY];
    
    // Buffer state
    int position;
    bool full;
    
    void reset() {
        position = 0;
        full = false;
    }
    
    int get_index(int env_idx, int step) const {
        return step * N_ENVS + env_idx;
    }
    
    void store(int env_idx, int step,
               const float* obs, const float* action,
               float log_prob, float value, float reward, float done) {
        int idx = get_index(env_idx, step);
        memcpy(observations[idx], obs, OBS_DIM * sizeof(float));
        memcpy(actions[idx], action, ACTION_DIM * sizeof(float));
        log_probs[idx] = log_prob;
        values[idx] = value;
        rewards[idx] = reward;
        dones[idx] = done;
    }
    
    /**
     * @brief Compute Generalized Advantage Estimation (GAE)
     */
    void compute_gae(const float* last_values, float gamma, float gae_lambda) {
        // Process each environment separately
        for (int env = 0; env < N_ENVS; env++) {
            float gae = 0;
            float next_value = last_values[env];
            float next_done = 0;
            
            // Backward pass through time
            for (int step = ROLLOUT_STEPS - 1; step >= 0; step--) {
                int idx = get_index(env, step);
                
                float delta = rewards[idx] + gamma * next_value * (1 - next_done) - values[idx];
                gae = delta + gamma * gae_lambda * (1 - next_done) * gae;
                
                advantages[idx] = gae;
                returns[idx] = gae + values[idx];
                
                next_value = values[idx];
                next_done = dones[idx];
            }
        }
    }
    
    /**
     * @brief Normalize advantages across the buffer
     */
    void normalize_advantages(float eps = 1e-8f) {
        // Compute mean
        float sum = 0;
        for (int i = 0; i < CAPACITY; i++) {
            sum += advantages[i];
        }
        float mean = sum / CAPACITY;
        
        // Compute std
        float var_sum = 0;
        for (int i = 0; i < CAPACITY; i++) {
            float diff = advantages[i] - mean;
            var_sum += diff * diff;
        }
        float std = std::sqrt(var_sum / CAPACITY + eps);
        
        // Normalize
        for (int i = 0; i < CAPACITY; i++) {
            advantages[i] = (advantages[i] - mean) / std;
        }
    }
};

/**
 * @brief Observation normalizer using Welford's online algorithm
 */
template<int OBS_DIM>
struct ObservationNormalizer {
    float mean[OBS_DIM];
    float var[OBS_DIM];
    float count;
    
    void init() {
        memset(mean, 0, sizeof(mean));
        for (int i = 0; i < OBS_DIM; i++) {
            var[i] = 1.0f;
        }
        count = 1e-4f;  // Small initial count for stability
    }
    
    void update(const float* obs) {
        count += 1;
        for (int i = 0; i < OBS_DIM; i++) {
            float delta = obs[i] - mean[i];
            mean[i] += delta / count;
            float delta2 = obs[i] - mean[i];
            var[i] += delta * delta2;
        }
    }
    
    void normalize(const float* obs, float* normalized, float eps = 1e-8f) const {
        for (int i = 0; i < OBS_DIM; i++) {
            float std = std::sqrt(var[i] / std::max(count, 1.0f) + eps);
            normalized[i] = (obs[i] - mean[i]) / std;
            // Clip to prevent extreme values
            normalized[i] = std::max(-10.0f, std::min(10.0f, normalized[i]));
        }
    }
    
    void update_batch(const float* obs_batch, int batch_size) {
        for (int b = 0; b < batch_size; b++) {
            update(obs_batch + b * OBS_DIM);
        }
    }
};

} // namespace cuda
} // namespace learning_to_fly

#endif // PPO_BUFFER_CUH
