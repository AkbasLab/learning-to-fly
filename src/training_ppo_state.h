/**
 * @file training_ppo_state.h
 * @brief PPO Training State
 * 
 * Extends the base PPO loop training state with learning-to-fly specific features.
 */

#ifndef LEARNING_TO_FLY_TRAINING_PPO_STATE_H
#define LEARNING_TO_FLY_TRAINING_PPO_STATE_H

#include "ppo/loop.h"

#include <queue>
#include <vector>
#include <mutex>
#include <limits>

namespace learning_to_fly {

    /**
     * @brief Extended PPO training state for learning-to-fly
     * 
     * Adds trajectory collection, validation, and logging features
     * on top of the base PPO training state.
     */
    template <typename T_CONFIG>
    struct PPOTrainingState : rlt::rl::algorithms::ppo::loop::TrainingState<T_CONFIG> {
        using CONFIG = T_CONFIG;
        using T = typename CONFIG::T;
        using TI = typename CONFIG::TI;
        
        // Run identification
        std::string run_name;
        
        // Training completion flag (for UI)
        bool finished = false;
        
        // Trajectory collection for visualization
        std::queue<std::vector<typename CONFIG::ENVIRONMENT::State>> trajectories;
        std::mutex trajectories_mutex;
        std::vector<typename CONFIG::ENVIRONMENT::State> current_episode;
        
        // Validation environments
        typename CONFIG::ENVIRONMENT validation_envs[CONFIG::VALIDATION_N_EPISODES];
        typename CONFIG::ACTOR_TYPE::template DoubleBuffer<CONFIG::VALIDATION_N_EPISODES> validation_actor_buffers;
        
        // Best return tracking for checkpoint-on-best
        T best_validation_return = -std::numeric_limits<T>::infinity();
        TI best_return_step = 0;
    };

} // namespace learning_to_fly

#endif // LEARNING_TO_FLY_TRAINING_PPO_STATE_H
