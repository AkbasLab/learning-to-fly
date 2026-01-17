/**
 * @file rollout_buffer.h
 * @brief On-policy rollout buffer for PPO
 * 
 * This file implements the rollout buffer used to collect trajectories
 * for PPO training. Unlike TD3's replay buffer, this is cleared after
 * each PPO update (on-policy learning).
 * 
 * The buffer stores:
 * - Observations (states)
 * - Actions taken
 * - Action log probabilities (from old policy)
 * - Rewards received
 * - Value estimates V(s)
 * - Terminal flags
 * - Computed advantages and returns (after GAE)
 * 
 * MEMORY LAYOUT:
 * - Static allocation with fixed maximum size
 * - All arrays are contiguous for efficient batch sampling
 * - Compatible with STM32 constraints (no dynamic allocation)
 */

#ifndef LEARNING_TO_FLY_PPO_ROLLOUT_BUFFER_H
#define LEARNING_TO_FLY_PPO_ROLLOUT_BUFFER_H

// Static arrays only - no matrix includes needed

namespace rl_tools::rl::algorithms::ppo {

    /**
     * @brief Rollout buffer specification
     */
    template<typename T, typename TI, TI T_OBSERVATION_DIM, TI T_ACTION_DIM, TI T_CAPACITY, TI T_N_ENVIRONMENTS>
    struct RolloutBufferSpecification {
        using SCALAR = T;
        using INDEX = TI;
        static constexpr TI OBSERVATION_DIM = T_OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = T_ACTION_DIM;
        static constexpr TI CAPACITY = T_CAPACITY;
        static constexpr TI N_ENVIRONMENTS = T_N_ENVIRONMENTS;
        // Total number of transitions stored
        static constexpr TI TOTAL_CAPACITY = CAPACITY * N_ENVIRONMENTS;
    };

    /**
     * @brief On-policy rollout buffer for PPO
     * 
     * Stores transitions collected during rollout for PPO update.
     * All data is stored in row-major order with shape [TOTAL_CAPACITY, DIM].
     */
    template<typename T_SPEC>
    struct RolloutBuffer {
        using SPEC = T_SPEC;
        using T = typename SPEC::SCALAR;
        using TI = typename SPEC::INDEX;
        
        static constexpr TI OBSERVATION_DIM = SPEC::OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = SPEC::ACTION_DIM;
        static constexpr TI CAPACITY = SPEC::CAPACITY;
        static constexpr TI N_ENVIRONMENTS = SPEC::N_ENVIRONMENTS;
        static constexpr TI TOTAL_CAPACITY = SPEC::TOTAL_CAPACITY;
        
        // Current position in buffer (per environment)
        TI position;
        
        // Whether buffer is full (ready for PPO update)
        bool full;
        
        // Observations at each timestep: [TOTAL_CAPACITY, OBSERVATION_DIM]
        T observations[TOTAL_CAPACITY * OBSERVATION_DIM];
        
        // Actions taken: [TOTAL_CAPACITY, ACTION_DIM]
        T actions[TOTAL_CAPACITY * ACTION_DIM];
        
        // Log probability of actions under old policy: [TOTAL_CAPACITY]
        T log_probs[TOTAL_CAPACITY];
        
        // Rewards received: [TOTAL_CAPACITY]
        T rewards[TOTAL_CAPACITY];
        
        // Value estimates V(s): [TOTAL_CAPACITY]
        T values[TOTAL_CAPACITY];
        
        // Terminal flags (1 if episode ended): [TOTAL_CAPACITY]
        T dones[TOTAL_CAPACITY];
        
        // Next observations (for bootstrapping): [TOTAL_CAPACITY, OBSERVATION_DIM]
        T next_observations[TOTAL_CAPACITY * OBSERVATION_DIM];
        
        // ============================================================
        // The following are computed after rollout, before PPO update
        // ============================================================
        
        // Advantage estimates (from GAE): [TOTAL_CAPACITY]
        T advantages[TOTAL_CAPACITY];
        
        // Returns (advantage + value): [TOTAL_CAPACITY]
        T returns[TOTAL_CAPACITY];
        
        // Normalized advantages (if enabled): [TOTAL_CAPACITY]
        T normalized_advantages[TOTAL_CAPACITY];
    };

    // ================================================================
    // Helper functions for rollout buffer access
    // ================================================================

    /**
     * @brief Get observation at given index
     */
    template<typename SPEC>
    inline typename SPEC::SCALAR* get_observation(RolloutBuffer<SPEC>& buffer, typename SPEC::INDEX idx) {
        return &buffer.observations[idx * SPEC::OBSERVATION_DIM];
    }

    template<typename SPEC>
    inline const typename SPEC::SCALAR* get_observation(const RolloutBuffer<SPEC>& buffer, typename SPEC::INDEX idx) {
        return &buffer.observations[idx * SPEC::OBSERVATION_DIM];
    }

    /**
     * @brief Get action at given index
     */
    template<typename SPEC>
    inline typename SPEC::SCALAR* get_action(RolloutBuffer<SPEC>& buffer, typename SPEC::INDEX idx) {
        return &buffer.actions[idx * SPEC::ACTION_DIM];
    }

    template<typename SPEC>
    inline const typename SPEC::SCALAR* get_action(const RolloutBuffer<SPEC>& buffer, typename SPEC::INDEX idx) {
        return &buffer.actions[idx * SPEC::ACTION_DIM];
    }

    /**
     * @brief Get next observation at given index
     */
    template<typename SPEC>
    inline typename SPEC::SCALAR* get_next_observation(RolloutBuffer<SPEC>& buffer, typename SPEC::INDEX idx) {
        return &buffer.next_observations[idx * SPEC::OBSERVATION_DIM];
    }

    template<typename SPEC>
    inline const typename SPEC::SCALAR* get_next_observation(const RolloutBuffer<SPEC>& buffer, typename SPEC::INDEX idx) {
        return &buffer.next_observations[idx * SPEC::OBSERVATION_DIM];
    }

    /**
     * @brief Convert environment index and step to flat buffer index
     */
    template<typename SPEC>
    inline typename SPEC::INDEX flat_index(typename SPEC::INDEX env_idx, typename SPEC::INDEX step, typename SPEC::INDEX capacity) {
        // Layout: all steps for env 0, then all steps for env 1, etc.
        return env_idx * capacity + step;
    }

} // namespace rl_tools::rl::algorithms::ppo

#endif // LEARNING_TO_FLY_PPO_ROLLOUT_BUFFER_H
