/**
 * @file loop.h
 * @brief PPO Training Loop
 * 
 * This file implements the main PPO training loop, analogous to
 * rl_tools/rl/algorithms/td3/loop.h but for on-policy learning.
 * 
 * TRAINING LOOP STRUCTURE:
 * 1. Collect trajectories using current policy (rollout phase)
 * 2. Compute advantages using GAE
 * 3. Perform PPO updates for K epochs over collected data
 * 4. Clear buffer and repeat
 * 
 * KEY DIFFERENCES FROM TD3:
 * - On-policy: uses fresh data each update cycle
 * - No replay buffer; uses rollout buffer cleared after each update
 * - Explicit action sampling from Gaussian distribution during training
 * - Deterministic inference (mean action) for deployment
 * 
 * DEPLOYMENT COMPATIBILITY:
 * - Actor network has same architecture as TD3
 * - Checkpoint export uses identical format
 * - STM32 inference code unchanged
 */

#ifndef LEARNING_TO_FLY_PPO_LOOP_H
#define LEARNING_TO_FLY_PPO_LOOP_H

#include "ppo.h"
#include "rollout_buffer.h"
#include "gae.h"
#include "operations_generic.h"

#include <rl_tools/rl/environments/operations_generic.h>
#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace rl_tools::rl::algorithms::ppo::loop {

    /**
     * @brief PPO Training State
     * 
     * Contains all data needed for PPO training, analogous to TD3 TrainingState.
     */
    template<typename T_CONFIG>
    struct TrainingState {
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using SPEC = typename CONFIG::PPO_SPEC;
        using ROLLOUT_BUFFER_SPEC = typename CONFIG::ROLLOUT_BUFFER_SPEC;
        
        // Device and RNG
        typename CONFIG::DEVICE device;
        typename CONFIG::RNG rng;
        typename CONFIG::RNG rng_eval;
        
        // Actor-Critic (actor is deployed, critic is training-only)
        ActorCritic<SPEC> actor_critic;
        
        // Environment(s)
        typename CONFIG::ENVIRONMENT envs[CONFIG::N_ENVIRONMENTS];
        typename CONFIG::ENVIRONMENT env_eval;
        
        // Environment states
        typename CONFIG::ENVIRONMENT::State states[CONFIG::N_ENVIRONMENTS];
        typename CONFIG::ENVIRONMENT::State next_states[CONFIG::N_ENVIRONMENTS];
        
        // Rollout buffer (on-policy, cleared after each update)
        RolloutBuffer<ROLLOUT_BUFFER_SPEC> rollout_buffer;
        
        // Network buffers for forward/backward passes
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> single_actor_buffer;
        typename CONFIG::CRITIC_TYPE::template DoubleBuffer<1> single_critic_buffer;
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<CONFIG::BATCH_SIZE> actor_buffer;
        typename CONFIG::CRITIC_TYPE::template DoubleBuffer<CONFIG::BATCH_SIZE> critic_buffer;
        
        // Matrices for single-sample operations
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::OBSERVATION_DIM>> single_obs_matrix;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::ACTION_DIM>> single_action_matrix;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 1>> single_value_matrix;
        
        // Matrices for batch operations
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::OBSERVATION_DIM>> batch_obs_matrix;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::ACTION_DIM>> batch_action_matrix;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::ACTION_DIM>> batch_actor_output;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, 1>> batch_value_matrix;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::ACTION_DIM>> actor_output_grad;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, 1>> critic_output_grad;
        
        // Input gradient matrices for backward pass (required by sequential model API)
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::OBSERVATION_DIM>> actor_d_input;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, CONFIG::BATCH_SIZE, CONFIG::OBSERVATION_DIM>> critic_d_input;
        
        // Temporary storage for batch data (non-matrix form)
        T batch_old_log_probs[CONFIG::BATCH_SIZE];
        T batch_advantages[CONFIG::BATCH_SIZE];
        T batch_returns[CONFIG::BATCH_SIZE];
        T batch_old_values[CONFIG::BATCH_SIZE];
        
        // Indices for mini-batch sampling
        TI batch_indices[CONFIG::ROLLOUT_BUFFER_SPEC::TOTAL_CAPACITY];
        
        // Training statistics
        TI step;
        TI total_steps;
        TI ppo_update_count;
        T episode_return[CONFIG::N_ENVIRONMENTS];
        TI episode_length[CONFIG::N_ENVIRONMENTS];
        
        // Last values for GAE bootstrap
        T last_values[CONFIG::N_ENVIRONMENTS];
        T last_dones[CONFIG::N_ENVIRONMENTS];
        
        // Logging
        T avg_policy_loss;
        T avg_value_loss;
        T avg_entropy;
        T avg_clip_fraction;
        T avg_approx_kl;
    };

    /**
     * @brief Initialize PPO training state
     */
    template<typename CONFIG>
    void init(TrainingState<CONFIG>& ts, typename CONFIG::TI seed = 0) {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        
        // Initialize RNG
        ts.rng = rlt::random::default_engine(typename CONFIG::DEVICE::SPEC::RANDOM(), seed);
        ts.rng_eval = rlt::random::default_engine(typename CONFIG::DEVICE::SPEC::RANDOM(), seed + 1000);
        
        // Allocate networks
        rlt::malloc(ts.device, ts.actor_critic.actor);
        rlt::malloc(ts.device, ts.actor_critic.critic);
        
        // Allocate buffers
        rlt::malloc(ts.device, ts.single_actor_buffer);
        rlt::malloc(ts.device, ts.single_critic_buffer);
        rlt::malloc(ts.device, ts.actor_buffer);
        rlt::malloc(ts.device, ts.critic_buffer);
        
        // Allocate matrices
        rlt::malloc(ts.device, ts.single_obs_matrix);
        rlt::malloc(ts.device, ts.single_action_matrix);
        rlt::malloc(ts.device, ts.single_value_matrix);
        rlt::malloc(ts.device, ts.batch_obs_matrix);
        rlt::malloc(ts.device, ts.batch_action_matrix);
        rlt::malloc(ts.device, ts.batch_actor_output);
        rlt::malloc(ts.device, ts.batch_value_matrix);
        rlt::malloc(ts.device, ts.actor_output_grad);
        rlt::malloc(ts.device, ts.critic_output_grad);
        rlt::malloc(ts.device, ts.actor_d_input);
        rlt::malloc(ts.device, ts.critic_d_input);
        
        // Initialize actor-critic
        ppo::init(ts.device, ts.actor_critic, ts.rng);
        
        // Initialize environments
        for (TI i = 0; i < CONFIG::N_ENVIRONMENTS; i++) {
            rlt::malloc(ts.device, ts.envs[i]);
            rlt::sample_initial_state(ts.device, ts.envs[i], ts.states[i], ts.rng);
            ts.episode_return[i] = 0;
            ts.episode_length[i] = 0;
        }
        rlt::malloc(ts.device, ts.env_eval);
        
        // Initialize rollout buffer
        reset_buffer(ts.rollout_buffer);
        
        // Initialize counters
        ts.step = 0;
        ts.total_steps = 0;
        ts.ppo_update_count = 0;
        
        // Initialize logging
        ts.avg_policy_loss = 0;
        ts.avg_value_loss = 0;
        ts.avg_entropy = 0;
        ts.avg_clip_fraction = 0;
        ts.avg_approx_kl = 0;
    }

    /**
     * @brief Collect a single environment step and store in rollout buffer
     */
    template<typename CONFIG>
    void collect_step(TrainingState<CONFIG>& ts, typename CONFIG::TI env_idx) {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using SPEC = typename CONFIG::PPO_SPEC;
        using PARAMS = typename SPEC::PARAMETERS;
        constexpr TI OBSERVATION_DIM = CONFIG::OBSERVATION_DIM;
        constexpr TI ACTION_DIM = CONFIG::ACTION_DIM;
        
        auto& buffer = ts.rollout_buffer;
        TI step = buffer.position;
        TI idx = flat_index<typename CONFIG::ROLLOUT_BUFFER_SPEC>(env_idx, step, CONFIG::ROLLOUT_STEPS);
        
        // Get observation from current state into matrix
        rlt::observe(ts.device, ts.envs[env_idx], ts.states[env_idx], ts.single_obs_matrix, ts.rng);
        
        // Copy observation to buffer
        T* obs = get_observation(buffer, idx);
        for (TI i = 0; i < OBSERVATION_DIM; i++) {
            obs[i] = rlt::get(ts.single_obs_matrix, 0, i);
        }
        
        // Forward pass through actor to get action mean
        rlt::evaluate(ts.device, ts.actor_critic.actor, ts.single_obs_matrix, ts.single_action_matrix, ts.single_actor_buffer);
        
        // Get action mean
        T action_mean[ACTION_DIM];
        for (TI i = 0; i < ACTION_DIM; i++) {
            action_mean[i] = rlt::get(ts.single_action_matrix, 0, i);
        }
        
        // Sample action from Gaussian and compute log probability
        T* action = get_action(buffer, idx);
        T log_prob;
        sample_action(ts.device, action_mean, ts.actor_critic.log_std, action, log_prob, ACTION_DIM, ts.rng);
        buffer.log_probs[idx] = log_prob;
        
        // Get value estimate from critic
        rlt::evaluate(ts.device, ts.actor_critic.critic, ts.single_obs_matrix, ts.single_value_matrix, ts.single_critic_buffer);
        buffer.values[idx] = rlt::get(ts.single_value_matrix, 0, 0);
        
        // Copy action to matrix for environment step
        for (TI i = 0; i < ACTION_DIM; i++) {
            rlt::set(ts.single_action_matrix, 0, i, action[i]);
        }
        
        // Step environment
        rlt::step(ts.device, ts.envs[env_idx], ts.states[env_idx], ts.single_action_matrix, ts.next_states[env_idx], ts.rng);
        T reward = rlt::reward(ts.device, ts.envs[env_idx], ts.states[env_idx], ts.single_action_matrix, ts.next_states[env_idx], ts.rng);
        bool terminated = rlt::terminated(ts.device, ts.envs[env_idx], ts.next_states[env_idx], ts.rng);
        
        // Store transition data
        buffer.rewards[idx] = reward;
        buffer.dones[idx] = terminated ? (T)1 : (T)0;
        
        // Store next observation
        rlt::observe(ts.device, ts.envs[env_idx], ts.next_states[env_idx], ts.single_obs_matrix, ts.rng);
        T* next_obs = get_next_observation(buffer, idx);
        for (TI i = 0; i < OBSERVATION_DIM; i++) {
            next_obs[i] = rlt::get(ts.single_obs_matrix, 0, i);
        }
        
        // Update episode statistics
        ts.episode_return[env_idx] += reward;
        ts.episode_length[env_idx]++;
        
        // Handle episode termination or max length
        bool truncated = ts.episode_length[env_idx] >= CONFIG::ENVIRONMENT_STEP_LIMIT;
        
        if (terminated || truncated) {
            // Log episode
            rlt::add_scalar(ts.device, ts.device.logger, "episode/return", ts.episode_return[env_idx]);
            rlt::add_scalar(ts.device, ts.device.logger, "episode/length", (T)ts.episode_length[env_idx]);
            
            // Reset episode
            ts.episode_return[env_idx] = 0;
            ts.episode_length[env_idx] = 0;
            rlt::sample_initial_state(ts.device, ts.envs[env_idx], ts.next_states[env_idx], ts.rng);
        }
        
        // Update state
        ts.states[env_idx] = ts.next_states[env_idx];
        ts.total_steps++;
    }

    /**
     * @brief Collect full rollout for all environments
     */
    template<typename CONFIG>
    void collect_rollout(TrainingState<CONFIG>& ts) {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        
        reset_buffer(ts.rollout_buffer);
        
        for (TI step = 0; step < CONFIG::ROLLOUT_STEPS; step++) {
            for (TI env_idx = 0; env_idx < CONFIG::N_ENVIRONMENTS; env_idx++) {
                collect_step(ts, env_idx);
            }
            ts.rollout_buffer.position++;
        }
        
        ts.rollout_buffer.full = true;
        
        // Compute last values for GAE bootstrap
        for (TI env_idx = 0; env_idx < CONFIG::N_ENVIRONMENTS; env_idx++) {
            rlt::observe(ts.device, ts.envs[env_idx], ts.states[env_idx], ts.single_obs_matrix, ts.rng);
            rlt::evaluate(ts.device, ts.actor_critic.critic, ts.single_obs_matrix, ts.single_value_matrix, ts.single_critic_buffer);
            ts.last_values[env_idx] = rlt::get(ts.single_value_matrix, 0, 0);
            ts.last_dones[env_idx] = 0;  // Not done yet
        }
    }

    /**
     * @brief Perform PPO update over collected rollout
     */
    template<typename CONFIG>
    void ppo_update(TrainingState<CONFIG>& ts) {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        using SPEC = typename CONFIG::PPO_SPEC;
        using PARAMS = typename SPEC::PARAMETERS;
        constexpr TI OBSERVATION_DIM = CONFIG::OBSERVATION_DIM;
        constexpr TI ACTION_DIM = CONFIG::ACTION_DIM;
        constexpr TI BATCH_SIZE = CONFIG::BATCH_SIZE;
        constexpr TI TOTAL_SAMPLES = CONFIG::ROLLOUT_BUFFER_SPEC::TOTAL_CAPACITY;
        
        auto& buffer = ts.rollout_buffer;
        
        // Compute GAE advantages
        compute_gae(
            ts.device,
            buffer,
            ts.last_values,
            ts.last_dones,
            PARAMS::GAMMA,
            PARAMS::GAE_LAMBDA
        );
        
        // Normalize advantages if enabled
        if constexpr (PARAMS::NORMALIZE_ADVANTAGE) {
            normalize_advantages<typename CONFIG::ROLLOUT_BUFFER_SPEC, T>(buffer, (T)1e-8);
        } else {
            copy_advantages_to_normalized(buffer);
        }
        
        // Reset statistics
        ts.avg_policy_loss = 0;
        ts.avg_value_loss = 0;
        ts.avg_entropy = 0;
        ts.avg_clip_fraction = 0;
        ts.avg_approx_kl = 0;
        TI update_count = 0;
        
        // PPO epochs
        for (TI epoch = 0; epoch < PARAMS::N_EPOCHS; epoch++) {
            // Shuffle indices for this epoch
            shuffle_indices<typename CONFIG::DEVICE, TI, T>(ts.device, ts.batch_indices, TOTAL_SAMPLES, ts.rng);
            
            // Process mini-batches
            for (TI batch_start = 0; batch_start + BATCH_SIZE <= TOTAL_SAMPLES; batch_start += BATCH_SIZE) {
                // Fill batch data
                for (TI i = 0; i < BATCH_SIZE; i++) {
                    TI idx = ts.batch_indices[batch_start + i];
                    
                    // Copy observation to batch matrix
                    const T* obs = get_observation(buffer, idx);
                    for (TI j = 0; j < OBSERVATION_DIM; j++) {
                        rlt::set(ts.batch_obs_matrix, i, j, obs[j]);
                    }
                    
                    // Copy action to batch matrix
                    const T* act = get_action(buffer, idx);
                    for (TI j = 0; j < ACTION_DIM; j++) {
                        rlt::set(ts.batch_action_matrix, i, j, act[j]);
                    }
                    
                    // Copy other data
                    ts.batch_old_log_probs[i] = buffer.log_probs[idx];
                    ts.batch_advantages[i] = buffer.normalized_advantages[idx];
                    ts.batch_returns[i] = buffer.returns[idx];
                    ts.batch_old_values[i] = buffer.values[idx];
                }
                
                // Forward pass through actor (stores intermediate activations for backward)
                rlt::forward(ts.device, ts.actor_critic.actor, ts.batch_obs_matrix);
                rlt::copy(ts.device, ts.device, rlt::output(ts.actor_critic.actor), ts.batch_actor_output);
                
                // Forward pass through critic (stores intermediate activations for backward)
                rlt::forward(ts.device, ts.actor_critic.critic, ts.batch_obs_matrix);
                rlt::copy(ts.device, ts.device, rlt::output(ts.actor_critic.critic), ts.batch_value_matrix);
                
                // Compute losses for each sample and accumulate gradients
                T total_policy_loss = 0;
                T total_value_loss = 0;
                T total_entropy = 0;
                T clip_count = 0;
                T total_approx_kl = 0;
                
                // Zero gradients
                rlt::zero_gradient(ts.device, ts.actor_critic.actor);
                rlt::zero_gradient(ts.device, ts.actor_critic.critic);
                for (TI j = 0; j < ACTION_DIM; j++) {
                    ts.actor_critic.log_std_gradient[j] = 0;
                }
                
                // Initialize gradient matrices
                rlt::set_all(ts.device, ts.actor_output_grad, (T)0);
                rlt::set_all(ts.device, ts.critic_output_grad, (T)0);
                
                // Compute per-sample losses and gradients
                for (TI i = 0; i < BATCH_SIZE; i++) {
                    // Get action mean for this sample
                    T action_mean[ACTION_DIM];
                    T action[ACTION_DIM];
                    for (TI j = 0; j < ACTION_DIM; j++) {
                        action_mean[j] = rlt::get(ts.batch_actor_output, i, j);
                        action[j] = rlt::get(ts.batch_action_matrix, i, j);
                    }
                    
                    // Compute new log probability
                    T log_prob_new = compute_log_prob(
                        action_mean,
                        ts.actor_critic.log_std,
                        action,
                        ACTION_DIM
                    );
                    
                    T log_prob_old = ts.batch_old_log_probs[i];
                    T advantage = ts.batch_advantages[i];
                    
                    // Policy loss (clipped surrogate)
                    T ratio = std::exp(log_prob_new - log_prob_old);
                    T clipped_ratio = std::max((T)(1 - PARAMS::CLIP_EPSILON), 
                                               std::min((T)(1 + PARAMS::CLIP_EPSILON), ratio));
                    
                    T grad_ratio;
                    if (ratio * advantage < clipped_ratio * advantage) {
                        total_policy_loss += -ratio * advantage;
                        grad_ratio = -advantage;
                    } else {
                        total_policy_loss += -clipped_ratio * advantage;
                        grad_ratio = 0;  // Clipped, no gradient
                        clip_count += 1;
                    }
                    
                    // Value loss
                    T value_pred = rlt::get(ts.batch_value_matrix, i, 0);
                    T value_target = ts.batch_returns[i];
                    T value_old = ts.batch_old_values[i];
                    T value_loss = compute_value_loss(
                        value_pred,
                        value_old,
                        value_target,
                        PARAMS::VALUE_CLIP_RANGE,
                        PARAMS::CLIP_VALUE_LOSS
                    );
                    total_value_loss += value_loss;
                    
                    // Entropy
                    T entropy = compute_entropy(ts.actor_critic.log_std, ACTION_DIM);
                    total_entropy += entropy;
                    
                    // Approximate KL for logging
                    T approx_kl = 0.5 * (log_prob_old - log_prob_new) * (log_prob_old - log_prob_new);
                    total_approx_kl += approx_kl;
                    
                    // Compute gradient for actor output
                    // d(loss)/d(action_mean) = d(loss)/d(log_prob) * d(log_prob)/d(action_mean)
                    // d(log_prob)/d(mu_i) = (a_i - mu_i) / sigma_i^2
                    T grad_log_prob = grad_ratio * ratio;
                    for (TI j = 0; j < ACTION_DIM; j++) {
                        T std = std::exp(ts.actor_critic.log_std[j]);
                        T d_log_prob_d_mu = (action[j] - action_mean[j]) / (std * std);
                        T actor_grad = grad_log_prob * d_log_prob_d_mu / BATCH_SIZE;
                        rlt::set(ts.actor_output_grad, i, j, actor_grad);
                    }
                    
                    // Compute gradient for critic output
                    // d(MSE)/d(pred) = 2 * (pred - target)
                    T critic_grad = 2 * PARAMS::VALUE_LOSS_COEFFICIENT * (value_pred - value_target) / BATCH_SIZE;
                    rlt::set(ts.critic_output_grad, i, 0, critic_grad);
                    
                    // Accumulate gradient for log_std
                    T log_prob_grad[ACTION_DIM];
                    T entropy_grad[ACTION_DIM];
                    compute_log_prob_gradient_log_std(action_mean, ts.actor_critic.log_std, action, 
                                                      log_prob_grad, ACTION_DIM);
                    compute_entropy_gradient_log_std(entropy_grad, ACTION_DIM);
                    
                    for (TI j = 0; j < ACTION_DIM; j++) {
                        ts.actor_critic.log_std_gradient[j] += 
                            (grad_log_prob * log_prob_grad[j] - PARAMS::ENTROPY_COEFFICIENT * entropy_grad[j]) / BATCH_SIZE;
                    }
                }
                
                // Average losses
                total_policy_loss /= BATCH_SIZE;
                total_value_loss /= BATCH_SIZE;
                total_entropy /= BATCH_SIZE;
                total_approx_kl /= BATCH_SIZE;
                
                // Backward pass for actor
                rlt::backward_full(ts.device, ts.actor_critic.actor, ts.batch_obs_matrix, ts.actor_output_grad, ts.actor_d_input, ts.actor_buffer);
                
                // Backward pass for critic
                rlt::backward_full(ts.device, ts.actor_critic.critic, ts.batch_obs_matrix, ts.critic_output_grad, ts.critic_d_input, ts.critic_buffer);
                
                // Update networks
                rlt::step(ts.device, ts.actor_critic.actor_optimizer, ts.actor_critic.actor);
                rlt::step(ts.device, ts.actor_critic.critic_optimizer, ts.actor_critic.critic);
                
                // Update log_std
                ts.ppo_update_count++;
                adam_update_log_std(
                    ts.actor_critic.log_std,
                    ts.actor_critic.log_std_gradient,
                    ts.actor_critic.log_std_first_moment,
                    ts.actor_critic.log_std_second_moment,
                    ACTION_DIM,
                    PARAMS::ACTOR_LEARNING_RATE,
                    (T)0.9, (T)0.999, (T)1e-8,
                    ts.ppo_update_count,
                    PARAMS::MIN_LOG_STD,
                    PARAMS::MAX_LOG_STD
                );
                
                // Accumulate statistics
                ts.avg_policy_loss += total_policy_loss;
                ts.avg_value_loss += total_value_loss;
                ts.avg_entropy += total_entropy;
                ts.avg_clip_fraction += clip_count / BATCH_SIZE;
                ts.avg_approx_kl += total_approx_kl;
                update_count++;
            }
        }
        
        // Average statistics
        if (update_count > 0) {
            ts.avg_policy_loss /= update_count;
            ts.avg_value_loss /= update_count;
            ts.avg_entropy /= update_count;
            ts.avg_clip_fraction /= update_count;
            ts.avg_approx_kl /= update_count;
        }
        
        // Log statistics
        rlt::add_scalar(ts.device, ts.device.logger, "ppo/policy_loss", ts.avg_policy_loss);
        rlt::add_scalar(ts.device, ts.device.logger, "ppo/value_loss", ts.avg_value_loss);
        rlt::add_scalar(ts.device, ts.device.logger, "ppo/entropy", ts.avg_entropy);
        rlt::add_scalar(ts.device, ts.device.logger, "ppo/clip_fraction", ts.avg_clip_fraction);
        rlt::add_scalar(ts.device, ts.device.logger, "ppo/approx_kl", ts.avg_approx_kl);
        rlt::add_scalar(ts.device, ts.device.logger, "ppo/log_std", ts.actor_critic.log_std[0]);
    }

    /**
     * @brief Single PPO step: collect rollout + update
     */
    template<typename CONFIG>
    void step(TrainingState<CONFIG>& ts) {
        // Collect full rollout
        collect_rollout(ts);
        
        // Perform PPO update
        ppo_update(ts);
        
        // Increment step counter (counts PPO updates, not env steps)
        ts.step++;
    }

    /**
     * @brief Cleanup PPO training state
     */
    template<typename CONFIG>
    void destroy(TrainingState<CONFIG>& ts) {
        using TI = typename CONFIG::TI;
        
        rlt::free(ts.device, ts.actor_critic.actor);
        rlt::free(ts.device, ts.actor_critic.critic);
        
        rlt::free(ts.device, ts.single_actor_buffer);
        rlt::free(ts.device, ts.single_critic_buffer);
        rlt::free(ts.device, ts.actor_buffer);
        rlt::free(ts.device, ts.critic_buffer);
        
        rlt::free(ts.device, ts.single_obs_matrix);
        rlt::free(ts.device, ts.single_action_matrix);
        rlt::free(ts.device, ts.single_value_matrix);
        rlt::free(ts.device, ts.batch_obs_matrix);
        rlt::free(ts.device, ts.batch_action_matrix);
        rlt::free(ts.device, ts.batch_actor_output);
        rlt::free(ts.device, ts.batch_value_matrix);
        rlt::free(ts.device, ts.actor_output_grad);
        rlt::free(ts.device, ts.critic_output_grad);
        rlt::free(ts.device, ts.actor_d_input);
        rlt::free(ts.device, ts.critic_d_input);
        
        // Environments don't need explicit free in this architecture
    }

} // namespace rl_tools::rl::algorithms::ppo::loop

#endif // LEARNING_TO_FLY_PPO_LOOP_H
