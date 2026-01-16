/**
 * @file ppo_config.h
 * @brief PPO Configuration for Quadrotor Control
 * 
 * This file defines the complete PPO configuration, including:
 * - Network architectures (actor and critic)
 * - PPO hyperparameters
 * - Environment settings
 * - Training parameters
 * 
 * DEPLOYMENT NOTE:
 * The actor network defined here has the SAME architecture as the TD3 actor.
 * This ensures that:
 * - Checkpoints are exported in the same format
 * - STM32 inference code works without modification
 * - The deployment pipeline is unchanged
 */

#ifndef LEARNING_TO_FLY_CONFIG_PPO_CONFIG_H
#define LEARNING_TO_FLY_CONFIG_PPO_CONFIG_H

#include "parameters.h"
#include "ablation.h"

#include "../ppo/ppo.h"
#include "../ppo/rollout_buffer.h"

#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

namespace learning_to_fly {
    namespace config {
        namespace ppo {

            using namespace rlt::nn_models::sequential::interface;

            /**
             * @brief PPO-specific hyperparameters for quadrotor control
             * 
             * These parameters are tuned for:
             * - High-frequency control (100Hz)
             * - Stable hover and trajectory tracking
             * - Sample-efficient learning
             */
            template<typename T, typename TI>
            struct PPOParameters : rlt::rl::algorithms::ppo::DefaultParameters<T, TI> {
                // Override defaults for quadrotor control
                
                // Clipping epsilon - lower for more conservative updates
                static constexpr T CLIP_EPSILON = 0.1;
                
                // Higher value loss coefficient for accurate value estimation
                static constexpr T VALUE_LOSS_COEFFICIENT = 0.5;
                
                // Entropy coefficient for exploration
                static constexpr T ENTROPY_COEFFICIENT = 0.01;
                
                // Discount factor for 100Hz control
                static constexpr T GAMMA = 0.99;
                
                // GAE lambda
                static constexpr T GAE_LAMBDA = 0.95;
                
                // Number of PPO epochs per update
                static constexpr TI N_EPOCHS = 15;
                
                // Mini-batch size
                static constexpr TI BATCH_SIZE = 64;
                
                // Rollout length (steps per PPO update)
                static constexpr TI ROLLOUT_STEPS = 2048;
                
                // Number of parallel environments
                static constexpr TI N_ENVIRONMENTS = 8;
                
                // Gradient clipping
                static constexpr T MAX_GRAD_NORM = 0.5;
                
                // Action distribution parameters
                static constexpr T INITIAL_LOG_STD = -0.5;
                static constexpr T MIN_LOG_STD = -2.0;
                static constexpr T MAX_LOG_STD = 0.5;
                
                // Advantage normalization
                static constexpr bool NORMALIZE_ADVANTAGE = true;
                
                // Value function clipping
                static constexpr bool CLIP_VALUE_LOSS = true;
                static constexpr T VALUE_CLIP_RANGE = 0.2;
                
                // Learning rates
                static constexpr T ACTOR_LEARNING_RATE = 5e-5;
                static constexpr T CRITIC_LEARNING_RATE = 1e-4;
            };

            /**
             * @brief Actor and Critic network definitions for PPO
             * 
             * CRITICAL: Actor architecture matches TD3 exactly for deployment compatibility
             */
            template <typename T, typename TI, typename ENVIRONMENT, typename PPO_PARAMS>
            struct ActorCriticNetworks {
                static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
                static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
                
                // ================================================================
                // ACTOR NETWORK (DEPLOYED TO STM32)
                // Architecture: obs -> 64 -> 64 -> action_dim
                // Activation: FAST_TANH (same as TD3)
                // Output: Action mean in [-1, 1]
                // ================================================================
                template <typename PARAMETER_TYPE>
                struct ACTOR {
                    static constexpr TI HIDDEN_DIM = 64;
                    static constexpr TI BATCH_SIZE = PPO_PARAMS::BATCH_SIZE;
                    static constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::FAST_TANH;
                    
                    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, 
                        PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Input>;
                    using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                    
                    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, 
                        PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Normal>;
                    using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                    
                    // Output layer uses FAST_TANH to bound actions to [-1, 1]
                    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        HIDDEN_DIM, ACTION_DIM, rlt::nn::activation_functions::FAST_TANH, 
                        PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Output>;
                    using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
                    
                    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
                };
                
                /**
                 * @brief Checkpoint actor type (for export to STM32)
                 * 
                 * Uses Plain parameters (no optimizer state) for minimal memory footprint.
                 */
                template <typename ACTOR>
                struct ACTOR_CHECKPOINT {
                    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        OBSERVATION_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, 
                        rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Input>;
                    using LAYER_1 = rlt::nn::layers::dense::Layer<LAYER_1_SPEC>;
                    
                    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        ACTOR::HIDDEN_DIM, ACTOR::HIDDEN_DIM, ACTOR::ACTIVATION_FUNCTION, 
                        rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Normal>;
                    using LAYER_2 = rlt::nn::layers::dense::Layer<LAYER_2_SPEC>;
                    
                    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        ACTOR::HIDDEN_DIM, ACTION_DIM, rlt::nn::activation_functions::FAST_TANH, 
                        rlt::nn::parameters::Plain, 1, rlt::nn::parameters::groups::Output>;
                    using LAYER_3 = rlt::nn::layers::dense::Layer<LAYER_3_SPEC>;
                    
                    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
                };
                
                // ================================================================
                // CRITIC NETWORK (VALUE FUNCTION - TRAINING ONLY)
                // Architecture: obs -> 64 -> 64 -> 1
                // Output: State value V(s)
                // NOT deployed to STM32
                // ================================================================
                template <typename PARAMETER_TYPE>
                struct CRITIC {
                    static constexpr TI HIDDEN_DIM = 64;
                    static constexpr TI BATCH_SIZE = PPO_PARAMS::BATCH_SIZE;
                    static constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::FAST_TANH;
                    
                    // Note: Critic takes only observation (not obs+action like TD3 Q-function)
                    using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, 
                        PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Input>;
                    using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                    
                    using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, 
                        PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Normal>;
                    using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                    
                    // Output: Single scalar value
                    using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, 
                        HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, 
                        PARAMETER_TYPE, BATCH_SIZE, rlt::nn::parameters::groups::Output>;
                    using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
                    
                    using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
                };
                
                // Optimizer
                struct OPTIMIZER_PARAMETERS : rlt::nn::optimizers::adam::DefaultParameters<T, TI> {
                    static constexpr T WEIGHT_DECAY = 0.0;
                    static constexpr T WEIGHT_DECAY_INPUT = 0.0;
                    static constexpr T WEIGHT_DECAY_OUTPUT = 0.0;
                    static constexpr T BIAS_LR_FACTOR = 1;
                };
                using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
                
                // Type aliases
                using ACTOR_TYPE = typename ACTOR<rlt::nn::parameters::Adam>::MODEL;
                using ACTOR_CHECKPOINT_TYPE = typename ACTOR_CHECKPOINT<ACTOR<rlt::nn::parameters::Plain>>::MODEL;
                using CRITIC_TYPE = typename CRITIC<rlt::nn::parameters::Adam>::MODEL;
            };

            /**
             * @brief Complete PPO configuration
             */
            template <typename T_ABLATION_SPEC>
            struct PPOConfig {
#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
                static constexpr bool BENCHMARK = true;
#else
                static constexpr bool BENCHMARK = false;
#endif
                using ABLATION_SPEC = T_ABLATION_SPEC;
                using LOGGER = rlt::LOGGER_FACTORY<>;
                using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
                using DEVICE = rlt::DEVICE_FACTORY<DEV_SPEC>;
                using T = float;
                using TI = typename DEVICE::index_t;
                using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), (TI)0));
                
                // Environment
                using ENVIRONMENT = typename parameters::environment<T, TI, ABLATION_SPEC>::ENVIRONMENT;
                using ABLATION_SPEC_EVAL_INSTANCE = ABLATION_SPEC_EVAL<ABLATION_SPEC>;
                using ENVIRONMENT_EVALUATION = typename parameters::environment<T, TI, ABLATION_SPEC_EVAL_INSTANCE>::ENVIRONMENT;
                
                // PPO parameters
                using PPO_PARAMS = PPOParameters<T, TI>;
                
                // Networks
                using NETWORKS = ActorCriticNetworks<T, TI, ENVIRONMENT, PPO_PARAMS>;
                using ACTOR_TYPE = typename NETWORKS::ACTOR_TYPE;
                using ACTOR_CHECKPOINT_TYPE = typename NETWORKS::ACTOR_CHECKPOINT_TYPE;
                using CRITIC_TYPE = typename NETWORKS::CRITIC_TYPE;
                using OPTIMIZER = typename NETWORKS::OPTIMIZER;
                
                // PPO specification
                using PPO_SPEC = rlt::rl::algorithms::ppo::Specification<
                    T, TI, ENVIRONMENT,
                    ACTOR_TYPE, CRITIC_TYPE,
                    OPTIMIZER, OPTIMIZER,
                    PPO_PARAMS
                >;
                
                // Rollout buffer specification
                using ROLLOUT_BUFFER_SPEC = rlt::rl::algorithms::ppo::RolloutBufferSpecification<
                    T, TI,
                    ENVIRONMENT::OBSERVATION_DIM,
                    ENVIRONMENT::ACTION_DIM,
                    PPO_PARAMS::ROLLOUT_STEPS,
                    PPO_PARAMS::N_ENVIRONMENTS
                >;
                
                // Dimensions
                static constexpr TI OBSERVATION_DIM = ENVIRONMENT::OBSERVATION_DIM;
                static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
                static constexpr TI BATCH_SIZE = PPO_PARAMS::BATCH_SIZE;
                static constexpr TI ROLLOUT_STEPS = PPO_PARAMS::ROLLOUT_STEPS;
                static constexpr TI N_ENVIRONMENTS = PPO_PARAMS::N_ENVIRONMENTS;
                
                // Training limits
                static constexpr TI STEP_LIMIT = 10000;  // Number of PPO updates
                static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;  // Max steps per episode
                static constexpr TI BASE_SEED = 0;
                
                // Checkpointing
                static constexpr bool ACTOR_ENABLE_CHECKPOINTS = !BENCHMARK;
                static constexpr TI ACTOR_CHECKPOINT_INTERVAL = 50;  // Every N PPO updates
                
                // Validation
                static constexpr TI VALIDATION_N_EPISODES = 10;
                static constexpr TI VALIDATION_MAX_EPISODE_LENGTH = ENVIRONMENT_STEP_LIMIT;
                static constexpr TI EVALUATION_INTERVAL = 10;  // Every N PPO updates
            };

        } // namespace ppo
    } // namespace config
} // namespace learning_to_fly

#endif // LEARNING_TO_FLY_CONFIG_PPO_CONFIG_H
