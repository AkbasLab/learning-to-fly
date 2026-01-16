/**
 * @file operations_generic.h
 * @brief Core PPO training operations
 * 
 * This file implements the core PPO update step including:
 * - Policy loss (clipped surrogate objective)
 * - Value function loss (optionally clipped)
 * - Entropy bonus
 * - Gradient computation and parameter updates
 * 
 * PPO ALGORITHM SUMMARY:
 * 1. Collect rollout using current policy
 * 2. Compute advantages using GAE
 * 3. For K epochs:
 *    a. Sample mini-batches from rollout
 *    b. Compute ratio r = π_new(a|s) / π_old(a|s)
 *    c. Compute clipped surrogate: L = min(r*A, clip(r, 1-ε, 1+ε)*A)
 *    d. Compute value loss and entropy bonus
 *    e. Update actor and critic
 * 4. Clear rollout buffer
 * 
 * DEPLOYMENT NOTE:
 * - All code in this file is training-only
 * - The actor network architecture is unchanged from TD3
 * - Inference remains deterministic: action = actor(observation)
 */

#ifndef LEARNING_TO_FLY_PPO_OPERATIONS_GENERIC_H
#define LEARNING_TO_FLY_PPO_OPERATIONS_GENERIC_H

#include "ppo.h"
#include "rollout_buffer.h"
#include "gae.h"

#include <cmath>
#include <algorithm>

namespace rl_tools::rl::algorithms::ppo {

    // ================================================================
    // Memory allocation and initialization
    // ================================================================

    /**
     * @brief Initialize actor-critic with random weights
     */
    template<typename DEVICE, typename SPEC, typename RNG>
    void init(DEVICE& device, ActorCritic<SPEC>& actor_critic, RNG& rng) {
        using T = typename SPEC::ENVIRONMENT::T;
        using TI = typename SPEC::ENVIRONMENT::TI;
        using PARAMS = typename SPEC::PARAMETERS;
        constexpr TI ACTION_DIM = SPEC::ACTION_DIM;
        
        // Initialize actor network
        rlt::init_weights(device, actor_critic.actor, rng);
        
        // Initialize critic network
        rlt::init_weights(device, actor_critic.critic, rng);
        
        // Initialize log_std to initial value
        for (TI i = 0; i < ACTION_DIM; i++) {
            actor_critic.log_std[i] = PARAMS::INITIAL_LOG_STD;
            actor_critic.log_std_gradient[i] = 0;
            actor_critic.log_std_first_moment[i] = 0;
            actor_critic.log_std_second_moment[i] = 0;
        }
        
        // Initialize optimizers
        rlt::reset_optimizer_state(device, actor_critic.actor_optimizer, actor_critic.actor);
        rlt::reset_optimizer_state(device, actor_critic.critic_optimizer, actor_critic.critic);
    }

    /**
     * @brief Reset rollout buffer for new collection
     */
    template<typename SPEC>
    void reset_buffer(RolloutBuffer<SPEC>& buffer) {
        buffer.position = 0;
        buffer.full = false;
    }

    // ================================================================
    // Action sampling and log probability computation
    // ================================================================

    /**
     * @brief Sample action from Gaussian policy and compute log probability
     * 
     * Policy is: π(a|s) = N(μ(s), σ²) where μ(s) = actor(s)
     * 
     * @param mean Action mean from actor network
     * @param log_std Log standard deviation (learnable parameter)
     * @param action [out] Sampled action
     * @param log_prob [out] Log probability of sampled action
     * @param rng Random number generator
     */
    template<typename DEVICE, typename T, typename TI, typename RNG>
    void sample_action(
        DEVICE& device,
        const T* mean,
        const T* log_std,
        T* action,
        T& log_prob,
        TI action_dim,
        RNG& rng
    ) {
        constexpr T LOG_2PI = 1.8378770664093453;  // log(2π)
        
        log_prob = 0;
        
        for (TI i = 0; i < action_dim; i++) {
            T std = std::exp(log_std[i]);
            
            // Sample from standard normal and transform
            T eps = rlt::random::normal_distribution::sample(
                typename DEVICE::SPEC::RANDOM(), 
                (T)0, (T)1, rng
            );
            action[i] = mean[i] + std * eps;
            
            // Clamp action to valid range [-1, 1]
            action[i] = std::max((T)-1, std::min((T)1, action[i]));
            
            // Log probability: log N(a; μ, σ²) = -0.5 * [(a-μ)²/σ² + log(σ²) + log(2π)]
            T normalized = (action[i] - mean[i]) / std;
            log_prob += -0.5 * (normalized * normalized + 2 * log_std[i] + LOG_2PI);
        }
    }

    /**
     * @brief Compute log probability of action under current policy
     * 
     * Used during PPO update to compute π_new(a|s).
     */
    template<typename T, typename TI>
    T compute_log_prob(
        const T* mean,
        const T* log_std,
        const T* action,
        TI action_dim
    ) {
        constexpr T LOG_2PI = 1.8378770664093453;
        
        T log_prob = 0;
        for (TI i = 0; i < action_dim; i++) {
            T std = std::exp(log_std[i]);
            T normalized = (action[i] - mean[i]) / std;
            log_prob += -0.5 * (normalized * normalized + 2 * log_std[i] + LOG_2PI);
        }
        return log_prob;
    }

    /**
     * @brief Compute entropy of the current policy
     * 
     * For Gaussian: H(π) = 0.5 * Σ_i [1 + log(2π) + 2*log_std_i]
     */
    template<typename T, typename TI>
    T compute_entropy(const T* log_std, TI action_dim) {
        constexpr T LOG_2PI = 1.8378770664093453;
        
        T entropy = 0;
        for (TI i = 0; i < action_dim; i++) {
            entropy += 0.5 * (1 + LOG_2PI + 2 * log_std[i]);
        }
        return entropy;
    }

    // ================================================================
    // Gradient computation for log_std
    // ================================================================

    /**
     * @brief Compute gradient of log probability w.r.t. log_std
     * 
     * ∂log_prob/∂log_std_i = (a_i - μ_i)²/σ_i² - 1
     * 
     * Note: This is the gradient for MAXIMIZING log_prob.
     * For the PPO loss, we multiply by advantage.
     */
    template<typename T, typename TI>
    void compute_log_prob_gradient_log_std(
        const T* mean,
        const T* log_std,
        const T* action,
        T* gradient,
        TI action_dim
    ) {
        for (TI i = 0; i < action_dim; i++) {
            T std = std::exp(log_std[i]);
            T normalized = (action[i] - mean[i]) / std;
            // d(log_prob)/d(log_std) = (normalized² - 1)
            gradient[i] = normalized * normalized - 1;
        }
    }

    /**
     * @brief Compute gradient of entropy w.r.t. log_std
     * 
     * ∂H/∂log_std_i = 1 (constant)
     */
    template<typename T, typename TI>
    void compute_entropy_gradient_log_std(T* gradient, TI action_dim) {
        for (TI i = 0; i < action_dim; i++) {
            gradient[i] = 1;
        }
    }

    // ================================================================
    // PPO Loss Computation
    // ================================================================

    /**
     * @brief Compute PPO clipped surrogate loss for a single sample
     * 
     * L_clip = -min(r*A, clip(r, 1-ε, 1+ε)*A)
     * 
     * where r = exp(log_prob_new - log_prob_old)
     * 
     * @return The clipped surrogate loss (to be minimized)
     */
    template<typename T>
    T compute_policy_loss(
        T log_prob_new,
        T log_prob_old,
        T advantage,
        T clip_epsilon
    ) {
        T ratio = std::exp(log_prob_new - log_prob_old);
        T clipped_ratio = std::max(1 - clip_epsilon, std::min(1 + clip_epsilon, ratio));
        
        T unclipped_loss = ratio * advantage;
        T clipped_loss = clipped_ratio * advantage;
        
        // Return negative because we minimize
        return -std::min(unclipped_loss, clipped_loss);
    }

    /**
     * @brief Compute value function loss
     * 
     * If clipping disabled: L_v = (V_pred - V_target)²
     * If clipping enabled: L_v = max((V_pred - V_target)², (V_clipped - V_target)²)
     * 
     * where V_clipped = V_old + clip(V_pred - V_old, -ε, ε)
     */
    template<typename T>
    T compute_value_loss(
        T value_pred,
        T value_old,
        T value_target,
        T clip_range,
        bool clip_value
    ) {
        T value_diff = value_pred - value_target;
        T unclipped_loss = value_diff * value_diff;
        
        if (clip_value) {
            T value_clipped = value_old + std::max(-clip_range, std::min(clip_range, value_pred - value_old));
            T clipped_diff = value_clipped - value_target;
            T clipped_loss = clipped_diff * clipped_diff;
            return std::max(unclipped_loss, clipped_loss);
        }
        
        return unclipped_loss;
    }

    // ================================================================
    // Mini-batch sampling
    // ================================================================

    /**
     * @brief Generate random index using uniform real distribution
     * 
     * Since rl_tools may not have uniform_int_distribution, we use
     * uniform_real_distribution and floor.
     */
    template<typename DEVICE, typename TI, typename T, typename RNG>
    TI random_index(DEVICE& device, TI max_exclusive, RNG& rng) {
        T val = rlt::random::uniform_real_distribution(
            typename DEVICE::SPEC::RANDOM(),
            (T)0, (T)max_exclusive, rng
        );
        TI result = (TI)val;
        // Handle edge case where val == max_exclusive (very rare but possible)
        if (result >= max_exclusive) result = max_exclusive - 1;
        return result;
    }

    /**
     * @brief Generate random indices for mini-batch sampling
     */
    template<typename DEVICE, typename TI, typename T, typename RNG>
    void generate_batch_indices(DEVICE& device, TI* indices, TI batch_size, TI total_size, RNG& rng) {
        // Simple random sampling with replacement
        for (TI i = 0; i < batch_size; i++) {
            indices[i] = random_index<DEVICE, TI, T>(device, total_size, rng);
        }
    }

    /**
     * @brief Fisher-Yates shuffle for generating permutation
     */
    template<typename DEVICE, typename TI, typename T, typename RNG>
    void shuffle_indices(DEVICE& device, TI* indices, TI size, RNG& rng) {
        // Initialize with sequential indices
        for (TI i = 0; i < size; i++) {
            indices[i] = i;
        }
        
        // Fisher-Yates shuffle
        for (TI i = size - 1; i > 0; i--) {
            TI j = random_index<DEVICE, TI, T>(device, i + 1, rng);
            // Swap
            TI tmp = indices[i];
            indices[i] = indices[j];
            indices[j] = tmp;
        }
    }

    // ================================================================
    // Adam update for log_std parameter
    // ================================================================

    /**
     * @brief Adam optimizer step for log_std
     */
    template<typename T, typename TI>
    void adam_update_log_std(
        T* log_std,
        T* gradient,
        T* first_moment,
        T* second_moment,
        TI action_dim,
        T learning_rate,
        T beta1,
        T beta2,
        T epsilon,
        TI timestep,
        T min_log_std,
        T max_log_std
    ) {
        T beta1_t = std::pow(beta1, (T)timestep);
        T beta2_t = std::pow(beta2, (T)timestep);
        
        for (TI i = 0; i < action_dim; i++) {
            // Update biased first moment estimate
            first_moment[i] = beta1 * first_moment[i] + (1 - beta1) * gradient[i];
            
            // Update biased second moment estimate
            second_moment[i] = beta2 * second_moment[i] + (1 - beta2) * gradient[i] * gradient[i];
            
            // Compute bias-corrected estimates
            T m_hat = first_moment[i] / (1 - beta1_t);
            T v_hat = second_moment[i] / (1 - beta2_t);
            
            // Update parameter (note: gradient descent, so subtract)
            log_std[i] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            
            // Clamp to valid range
            log_std[i] = std::max(min_log_std, std::min(max_log_std, log_std[i]));
        }
    }

} // namespace rl_tools::rl::algorithms::ppo

#endif // LEARNING_TO_FLY_PPO_OPERATIONS_GENERIC_H
