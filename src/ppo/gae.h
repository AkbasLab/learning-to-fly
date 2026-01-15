/**
 * @file gae.h
 * @brief Generalized Advantage Estimation (GAE) for PPO
 * 
 * Implements GAE as described in "High-Dimensional Continuous Control 
 * Using Generalized Advantage Estimation" (Schulman et al., 2015).
 * 
 * GAE provides a family of policy gradient estimators controlled by λ:
 * - λ = 0: One-step TD advantage (high bias, low variance)
 * - λ = 1: Monte Carlo advantage (low bias, high variance)
 * - 0 < λ < 1: Interpolation between the two
 * 
 * The advantage at time t is:
 *   A_t = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
 * where δ_t = r_t + γ * V(s_{t+1}) - V(s_t) is the TD residual.
 * 
 * NUMERICAL STABILITY:
 * - Computation is done in reverse order for stability
 * - Returns are computed as advantage + value for critic training
 */

#ifndef LEARNING_TO_FLY_PPO_GAE_H
#define LEARNING_TO_FLY_PPO_GAE_H

#include "rollout_buffer.h"
#include <cmath>

namespace rl_tools::rl::algorithms::ppo {

    /**
     * @brief Compute GAE advantages and returns for a rollout buffer
     * 
     * @tparam DEVICE Device type (CPU, CUDA, etc.)
     * @tparam SPEC Rollout buffer specification
     * @tparam T Floating point type
     * 
     * @param device Compute device
     * @param buffer Rollout buffer with collected transitions
     * @param last_values Value estimates V(s_T) for bootstrap (one per environment)
     * @param last_dones Whether the last state was terminal (one per environment)
     * @param gamma Discount factor
     * @param gae_lambda GAE lambda parameter
     * 
     * This function computes advantages and returns in-place in the buffer.
     * Must be called after rollout collection, before PPO update.
     */
    template<typename DEVICE, typename SPEC, typename T>
    void compute_gae(
        DEVICE& device,
        RolloutBuffer<SPEC>& buffer,
        const T* last_values,
        const T* last_dones,
        T gamma,
        T gae_lambda
    ) {
        using TI = typename SPEC::INDEX;
        constexpr TI CAPACITY = SPEC::CAPACITY;
        constexpr TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
        
        // Compute GAE for each environment independently
        for (TI env_idx = 0; env_idx < N_ENVIRONMENTS; env_idx++) {
            // Initialize last GAE advantage
            T last_gae_advantage = 0;
            
            // Bootstrap value for computing the last TD residual
            T next_value = last_values[env_idx];
            T next_non_terminal = (T)1.0 - last_dones[env_idx];
            
            // Process transitions in reverse order (from last to first)
            for (TI step = CAPACITY; step > 0; step--) {
                TI t = step - 1;  // Current timestep
                TI idx = flat_index<SPEC>(env_idx, t, CAPACITY);
                
                T reward = buffer.rewards[idx];
                T value = buffer.values[idx];
                T done = buffer.dones[idx];
                T non_terminal = (T)1.0 - done;
                
                // TD residual: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
                T delta = reward + gamma * next_value * next_non_terminal - value;
                
                // GAE: A_t = δ_t + γλ * (1 - done) * A_{t+1}
                last_gae_advantage = delta + gamma * gae_lambda * next_non_terminal * last_gae_advantage;
                
                // Store advantage
                buffer.advantages[idx] = last_gae_advantage;
                
                // Return = advantage + value (target for value function)
                buffer.returns[idx] = last_gae_advantage + value;
                
                // Update for next iteration (going backwards)
                next_value = value;
                next_non_terminal = non_terminal;
                
                // If episode ended, reset GAE (handled implicitly by non_terminal multiplier)
                // The done flag from the current timestep affects the NEXT backward iteration
                if (done > 0.5) {
                    last_gae_advantage = 0;
                }
            }
        }
    }

    /**
     * @brief Normalize advantages across the entire buffer
     * 
     * Normalizing advantages can improve training stability:
     * A_normalized = (A - mean(A)) / (std(A) + eps)
     * 
     * @param buffer Rollout buffer with computed advantages
     * @param eps Small constant for numerical stability
     */
    template<typename SPEC, typename T>
    void normalize_advantages(RolloutBuffer<SPEC>& buffer, T eps = 1e-8) {
        using TI = typename SPEC::INDEX;
        constexpr TI TOTAL = SPEC::TOTAL_CAPACITY;
        
        // Compute mean
        T mean = 0;
        for (TI i = 0; i < TOTAL; i++) {
            mean += buffer.advantages[i];
        }
        mean /= (T)TOTAL;
        
        // Compute standard deviation
        T var = 0;
        for (TI i = 0; i < TOTAL; i++) {
            T diff = buffer.advantages[i] - mean;
            var += diff * diff;
        }
        var /= (T)TOTAL;
        T std = std::sqrt(var) + eps;
        
        // Normalize
        for (TI i = 0; i < TOTAL; i++) {
            buffer.normalized_advantages[i] = (buffer.advantages[i] - mean) / std;
        }
    }

    /**
     * @brief Copy raw advantages to normalized_advantages without normalization
     * 
     * Used when advantage normalization is disabled.
     */
    template<typename SPEC>
    void copy_advantages_to_normalized(RolloutBuffer<SPEC>& buffer) {
        using TI = typename SPEC::INDEX;
        for (TI i = 0; i < SPEC::TOTAL_CAPACITY; i++) {
            buffer.normalized_advantages[i] = buffer.advantages[i];
        }
    }

} // namespace rl_tools::rl::algorithms::ppo

#endif // LEARNING_TO_FLY_PPO_GAE_H
