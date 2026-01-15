/**
 * @file ppo.h
 * @brief Proximal Policy Optimization (PPO) algorithm structures
 * 
 * This file defines the core PPO algorithm specification and actor-critic structure.
 * 
 * PPO Overview:
 * - On-policy algorithm using clipped surrogate objective
 * - Separate policy (actor) and value (critic) networks
 * - Actor outputs mean of action distribution; std is a learnable parameter
 * - Critic estimates state value V(s) for advantage computation
 * 
 * DEPLOYMENT NOTE:
 * - Only the ACTOR network is deployed to STM32
 * - At inference, we use the mean action (deterministic)
 * - Value network, log_std, and all PPO-specific logic are training-only
 */

#ifndef LEARNING_TO_FLY_PPO_PPO_H
#define LEARNING_TO_FLY_PPO_PPO_H

namespace rl_tools::rl::algorithms::ppo {

    /**
     * @brief Default PPO hyperparameters
     * 
     * These values are tuned for quadrotor control with high-frequency (100Hz) control loops.
     */
    template<typename T, typename TI>
    struct DefaultParameters {
        // PPO clipping parameter (epsilon)
        static constexpr T CLIP_EPSILON = 0.2;
        
        // Value function loss coefficient
        static constexpr T VALUE_LOSS_COEFFICIENT = 0.5;
        
        // Entropy bonus coefficient (encourages exploration)
        static constexpr T ENTROPY_COEFFICIENT = 0.01;
        
        // Discount factor (gamma)
        static constexpr T GAMMA = 0.99;
        
        // GAE lambda parameter
        static constexpr T GAE_LAMBDA = 0.95;
        
        // Number of epochs per PPO update
        static constexpr TI N_EPOCHS = 10;
        
        // Mini-batch size for PPO updates
        static constexpr TI BATCH_SIZE = 64;
        
        // Rollout buffer size (steps per environment before update)
        static constexpr TI ROLLOUT_STEPS = 2048;
        
        // Number of parallel environments
        static constexpr TI N_ENVIRONMENTS = 1;
        
        // Maximum gradient norm for clipping
        static constexpr T MAX_GRAD_NORM = 0.5;
        
        // Initial action standard deviation (log space)
        static constexpr T INITIAL_LOG_STD = -0.5;
        
        // Minimum action standard deviation (log space)
        static constexpr T MIN_LOG_STD = -2.0;
        
        // Maximum action standard deviation (log space)
        static constexpr T MAX_LOG_STD = 0.5;
        
        // Whether to normalize advantages
        static constexpr bool NORMALIZE_ADVANTAGE = true;
        
        // Whether to clip value function loss
        static constexpr bool CLIP_VALUE_LOSS = true;
        
        // Value function clipping range
        static constexpr T VALUE_CLIP_RANGE = 0.2;
        
        // Learning rate for actor
        static constexpr T ACTOR_LEARNING_RATE = 3e-4;
        
        // Learning rate for critic
        static constexpr T CRITIC_LEARNING_RATE = 3e-4;
        
        // Whether to use orthogonal initialization
        static constexpr bool ORTHOGONAL_INIT = true;
        
        // Orthogonal init scale for hidden layers
        static constexpr T ORTHOGONAL_INIT_SCALE = 1.41421356237; // sqrt(2)
        
        // Orthogonal init scale for output layers
        static constexpr T ORTHOGONAL_INIT_OUTPUT_SCALE = 0.01;
    };

    /**
     * @brief PPO Algorithm Specification
     * 
     * @tparam T Floating point type
     * @tparam TI Index type
     * @tparam T_ENVIRONMENT Environment type
     * @tparam T_ACTOR_TYPE Actor network type (deployed to STM32)
     * @tparam T_CRITIC_TYPE Value network type (training only)
     * @tparam T_ACTOR_OPTIMIZER Actor optimizer type
     * @tparam T_CRITIC_OPTIMIZER Critic optimizer type  
     * @tparam T_PARAMETERS PPO hyperparameters
     */
    template<
        typename T,
        typename TI,
        typename T_ENVIRONMENT,
        typename T_ACTOR_TYPE,
        typename T_CRITIC_TYPE,
        typename T_ACTOR_OPTIMIZER,
        typename T_CRITIC_OPTIMIZER,
        typename T_PARAMETERS
    >
    struct Specification {
        using ENVIRONMENT = T_ENVIRONMENT;
        using ACTOR_TYPE = T_ACTOR_TYPE;
        using CRITIC_TYPE = T_CRITIC_TYPE;
        using ACTOR_OPTIMIZER = T_ACTOR_OPTIMIZER;
        using CRITIC_OPTIMIZER = T_CRITIC_OPTIMIZER;
        using PARAMETERS = T_PARAMETERS;
        
        // Derived dimensions
        static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
        static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        static constexpr TI ROLLOUT_STEPS = PARAMETERS::ROLLOUT_STEPS;
        static constexpr TI N_ENVIRONMENTS = PARAMETERS::N_ENVIRONMENTS;
        static constexpr TI BATCH_SIZE = PARAMETERS::BATCH_SIZE;
    };

    /**
     * @brief PPO Actor-Critic structure
     * 
     * Contains the actor (policy) and critic (value) networks along with
     * action distribution parameters.
     * 
     * DEPLOYMENT NOTE:
     * - Only 'actor' is used at inference time on STM32
     * - log_std, critic, and optimizers are training-only
     */
    template<typename T_SPEC>
    struct ActorCritic {
        using SPEC = T_SPEC;
        using T = typename SPEC::ENVIRONMENT::T;
        using TI = typename SPEC::ENVIRONMENT::TI;
        
        // Actor network (policy mean) - THIS IS DEPLOYED TO STM32
        typename SPEC::ACTOR_TYPE actor;
        
        // Critic network (value function) - TRAINING ONLY
        typename SPEC::CRITIC_TYPE critic;
        
        // Action log standard deviation (learnable) - TRAINING ONLY
        // Shape: [ACTION_DIM] - one std per action dimension
        T log_std[SPEC::ACTION_DIM];
        
        // Optimizers - TRAINING ONLY
        typename SPEC::ACTOR_OPTIMIZER actor_optimizer;
        typename SPEC::CRITIC_OPTIMIZER critic_optimizer;
        
        // Gradient accumulator for log_std - TRAINING ONLY
        T log_std_gradient[SPEC::ACTION_DIM];
        
        // Adam optimizer state for log_std - TRAINING ONLY
        T log_std_first_moment[SPEC::ACTION_DIM];
        T log_std_second_moment[SPEC::ACTION_DIM];
    };

} // namespace rl_tools::rl::algorithms::ppo

#endif // LEARNING_TO_FLY_PPO_PPO_H
