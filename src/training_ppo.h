/**
 * @file training_ppo.h
 * @brief PPO Training Functions for Quadrotor Control
 * 
 * This file implements the main PPO training loop for learning to fly.
 * It provides init(), step(), and destroy() functions analogous to the
 * existing TD3 training code.
 * 
 * TRAINING PIPELINE:
 * 1. Initialize environment, networks, and rollout buffer
 * 2. For each PPO update:
 *    a. Collect trajectories (ROLLOUT_STEPS per environment)
 *    b. Compute GAE advantages
 *    c. Perform PPO updates for N_EPOCHS
 *    d. Clear rollout buffer
 *    e. Log metrics and save checkpoints
 * 
 * DEPLOYMENT:
 * - Checkpoints are saved in the same format as TD3
 * - Only the actor network is exported
 * - STM32 deployment code is unchanged
 */

#ifndef LEARNING_TO_FLY_TRAINING_PPO_H
#define LEARNING_TO_FLY_TRAINING_PPO_H

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/metrics.h>

#include "config/ppo_config.h"
#include "ppo/loop.h"
#include "ppo/observation_normalizer.h"
#include "ppo/deployment_safety.h"
#include "training_ppo_state.h"

// Checkpoint saving (reuse from TD3)
#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/containers/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#endif

#include <rl_tools/containers/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include "helpers.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>


namespace learning_to_fly {
    namespace ppo_training {

        /**
         * @brief Generate run name for PPO training
         */
        template <typename ABLATION_SPEC, typename CONFIG>
        std::string ppo_run_name(typename CONFIG::TI seed) {
            std::stringstream run_name_ss;
            auto now = std::chrono::system_clock::now();
            auto local_time = std::chrono::system_clock::to_time_t(now);
            std::tm *tm = std::localtime(&local_time);
            run_name_ss << std::put_time(tm, "%Y_%m_%d_%H_%M_%S");
            run_name_ss << "_PPO";  // Mark as PPO training
            if constexpr (CONFIG::BENCHMARK) {
                run_name_ss << "_BENCHMARK";
            }
            run_name_ss << "_" << helpers::ablation_name<ABLATION_SPEC>();
            run_name_ss << "_" << std::setw(3) << std::setfill('0') << seed;
            return run_name_ss.str();
        }

        /**
         * @brief Initialize PPO training state
         */
        template <typename T_CONFIG>
        void init(PPOTrainingState<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0) {
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            using ABLATION_SPEC = typename CONFIG::ABLATION_SPEC;
            
            // Get environment parameters
            auto env_parameters = parameters::environment<T, TI, ABLATION_SPEC>::parameters;
            
            // Set up environments
            for (auto& env : ts.envs) {
                env.parameters = env_parameters;
            }
            ts.env_eval.parameters = env_parameters;
            
            // Generate run name
            TI effective_seed = CONFIG::BASE_SEED + seed;
            ts.run_name = ppo_run_name<ABLATION_SPEC, CONFIG>(effective_seed);
            
            // Set up logger
            rlt::construct(ts.device, ts.device.logger, std::string("logs"), ts.run_name);
            rlt::set_step(ts.device, ts.device.logger, 0);
            rlt::add_scalar(ts.device, ts.device.logger, "loop/seed", effective_seed);
            
            // Initialize base PPO training state
            rlt::rl::algorithms::ppo::loop::init(ts, effective_seed);
            
            // Initialize validation environments
            for (typename CONFIG::ENVIRONMENT& env : ts.validation_envs) {
                env.parameters = env_parameters;
            }
            rlt::malloc(ts.device, ts.validation_actor_buffers);
            
            // Print environment info
            std::cout << "PPO Training - Environment Info:\n";
            std::cout << "\t" << "Observation dim: " << CONFIG::ENVIRONMENT::OBSERVATION_DIM << std::endl;
            std::cout << "\t" << "Action dim: " << CONFIG::ENVIRONMENT::ACTION_DIM << std::endl;
            std::cout << "\t" << "Rollout steps: " << CONFIG::ROLLOUT_STEPS << std::endl;
            std::cout << "\t" << "Batch size: " << CONFIG::BATCH_SIZE << std::endl;
            std::cout << "\t" << "N environments: " << CONFIG::N_ENVIRONMENTS << std::endl;
        }

        /**
         * @brief Save actor checkpoint (same format as TD3)
         * 
         * The checkpoint format is identical to TD3, ensuring STM32 deployment
         * code works without modification.
         * 
         * CRITICAL ADDITIONS FOR PPO:
         * 1. Export observation normalizer (mean, std) for consistent inference
         * 2. Export log_std values (even though we use mean-only inference)
         * 3. Export deployment safety header for STM32
         */
        template <typename T_CONFIG>
        void checkpoint(PPOTrainingState<T_CONFIG>& ts) {
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            
            if (CONFIG::ACTOR_ENABLE_CHECKPOINTS && 
                (ts.step % CONFIG::ACTOR_CHECKPOINT_INTERVAL == 0)) {
                
                // Use same checkpoint directory structure as TD3
                const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/multirotor_ppo";
                std::filesystem::path actor_output_dir = 
                    std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / ts.run_name;
                
                try {
                    std::filesystem::create_directories(actor_output_dir);
                } catch (std::exception& e) {
                    std::cerr << "Failed to create checkpoint directory: " << e.what() << std::endl;
                }
                
                std::stringstream checkpoint_name_ss;
                checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << ts.step;
                std::string checkpoint_name = checkpoint_name_ss.str();
                
#if defined(RL_TOOLS_ENABLE_HDF5) && !defined(RL_TOOLS_DISABLE_HDF5)
                // Save HDF5 checkpoint
                std::filesystem::path actor_output_path_hdf5 = 
                    actor_output_dir / (checkpoint_name + ".h5");
                std::cout << "Saving PPO actor checkpoint: " << actor_output_path_hdf5 << std::endl;
                try {
                    auto actor_file = HighFive::File(actor_output_path_hdf5.string(), 
                                                      HighFive::File::Overwrite);
                    rlt::save(ts.device, ts.actor_critic.actor, actor_file.createGroup("actor"));
                } catch (HighFive::Exception& e) {
                    std::cout << "Error saving actor: " << e.what() << std::endl;
                }
#endif
                
                // Save C++ header checkpoint (CRITICAL FOR STM32 DEPLOYMENT)
                {
                    typename CONFIG::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                    typename decltype(ts.actor_critic.actor)::template DoubleBuffer<1> actor_buffer;
                    typename decltype(actor_checkpoint)::template DoubleBuffer<1> actor_checkpoint_buffer;
                    
                    rlt::malloc(ts.device, actor_checkpoint);
                    rlt::malloc(ts.device, actor_buffer);
                    rlt::malloc(ts.device, actor_checkpoint_buffer);
                    
                    // Copy weights to checkpoint format
                    rlt::copy(ts.device, ts.device, ts.actor_critic.actor, actor_checkpoint);
                    
                    std::filesystem::path actor_output_path_code = 
                        actor_output_dir / (checkpoint_name + ".h");
                    
                    // Generate C++ code for weights
                    auto actor_weights = rlt::save_code(
                        ts.device, actor_checkpoint, 
                        std::string("rl_tools::checkpoint::actor"), true
                    );
                    
                    std::cout << "Saving PPO checkpoint: " << actor_output_path_code << std::endl;
                    std::ofstream actor_output_file(actor_output_path_code);
                    actor_output_file << actor_weights;
                    
                    // ================================================================
                    // PPO-SPECIFIC: Export log_std values
                    // ================================================================
                    actor_output_file << "\n// PPO log_std parameters (for reference, not used in deterministic inference)\n";
                    actor_output_file << "namespace rl_tools::checkpoint::ppo {\n";
                    actor_output_file << "    constexpr int ACTION_DIM = " << CONFIG::ACTION_DIM << ";\n";
                    actor_output_file << "    constexpr float log_std[ACTION_DIM] = {";
                    for (TI i = 0; i < CONFIG::ACTION_DIM; i++) {
                        actor_output_file << ts.actor_critic.log_std[i] << "f";
                        if (i < CONFIG::ACTION_DIM - 1) actor_output_file << ", ";
                    }
                    actor_output_file << "};\n";
                    actor_output_file << "    constexpr float std[ACTION_DIM] = {";
                    for (TI i = 0; i < CONFIG::ACTION_DIM; i++) {
                        actor_output_file << std::exp(ts.actor_critic.log_std[i]) << "f";
                        if (i < CONFIG::ACTION_DIM - 1) actor_output_file << ", ";
                    }
                    actor_output_file << "};\n";
                    actor_output_file << "}\n";
                    
                    // ================================================================
                    // PPO-SPECIFIC: Export observation normalizer
                    // ================================================================
                    actor_output_file << "\n// Observation normalization parameters - CRITICAL FOR DEPLOYMENT!\n";
                    actor_output_file << rlt::rl::algorithms::ppo::save_normalizer_code(
                        ts.obs_normalizer, 
                        "rl_tools::checkpoint"
                    );
                    
                    // Add test observation/action for verification
                    {
                        typename CONFIG::ENVIRONMENT::State state;
                        rlt::sample_initial_state(ts.device, ts.envs[0], state, ts.rng_eval);
                        
                        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::OBSERVATION_DIM>> observation;
                        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::ACTION_DIM>> action;
                        rlt::malloc(ts.device, observation);
                        rlt::malloc(ts.device, action);
                        
                        auto rng_copy = ts.rng_eval;
                        rlt::observe(ts.device, ts.env_eval, state, observation, rng_copy);
                        
                        // Normalize observation before inference (matching deployment)
                        ts.obs_normalizer.normalize_matrix(ts.device, observation);
                        
                        rlt::evaluate(ts.device, ts.actor_critic.actor, observation, action, actor_buffer);
                        rlt::evaluate(ts.device, actor_checkpoint, observation, action, actor_checkpoint_buffer);
                        
                        actor_output_file << "\n" << rlt::save_code(ts.device, observation, 
                            std::string("rl_tools::checkpoint::observation"), true);
                        actor_output_file << "\n" << rlt::save_code(ts.device, action, 
                            std::string("rl_tools::checkpoint::action"), true);
                        
                        // Metadata
                        actor_output_file << "\n" << "namespace rl_tools::checkpoint::meta{";
                        actor_output_file << "\n" << "   " << "char name[] = \"" 
                            << ts.run_name << "_" << checkpoint_name << "\";";
                        actor_output_file << "\n" << "   " << "char algorithm[] = \"PPO\";";
                        actor_output_file << "\n" << "   " << "char commit_hash[] = \"" 
                            << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                        actor_output_file << "\n" << "   " << "// DEPLOYMENT NOTE: Use deterministic inference (actor output = action mean)";
                        actor_output_file << "\n" << "   " << "// Actions are already bounded to [-1, 1] by tanh output activation";
                        actor_output_file << "\n" << "   " << "// Apply observation normalization BEFORE actor inference!";
                        actor_output_file << "\n" << "}";
                        
                        rlt::free(ts.device, observation);
                        rlt::free(ts.device, action);
                    }
                    
                    rlt::free(ts.device, actor_checkpoint);
                    rlt::free(ts.device, actor_buffer);
                    rlt::free(ts.device, actor_checkpoint_buffer);
                }
                
                // ================================================================
                // SAVE STANDALONE DEPLOYMENT SAFETY HEADER
                // ================================================================
                {
                    std::filesystem::path safety_header_path = 
                        actor_output_dir / "deployment_safety.h";
                    std::ofstream safety_file(safety_header_path);
                    safety_file << rlt::deployment::generate_deployment_safety_header();
                    std::cout << "Saved deployment safety header: " << safety_header_path << std::endl;
                }
            }
        }
        
        /**
         * @brief Save checkpoint when new best validation return is achieved
         * 
         * This saves the actor with a special "best" prefix, making it easy to
         * identify the best performing model for deployment.
         */
        template <typename T_CONFIG>
        void checkpoint_best(PPOTrainingState<T_CONFIG>& ts, typename T_CONFIG::T avg_return) {
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            
            const std::string ACTOR_CHECKPOINT_DIRECTORY = "checkpoints/multirotor_ppo";
            std::filesystem::path actor_output_dir = 
                std::filesystem::path(ACTOR_CHECKPOINT_DIRECTORY) / ts.run_name;
            
            try {
                std::filesystem::create_directories(actor_output_dir);
            } catch (std::exception& e) {
                std::cerr << "Failed to create checkpoint directory: " << e.what() << std::endl;
                return;
            }
            
            // Format return value for filename (replace decimal point with underscore)
            std::stringstream return_ss;
            return_ss << std::fixed << std::setprecision(2) << avg_return;
            std::string return_str = return_ss.str();
            std::replace(return_str.begin(), return_str.end(), '.', '_');
            std::replace(return_str.begin(), return_str.end(), '-', 'n');  // negative sign
            
            std::stringstream checkpoint_name_ss;
            checkpoint_name_ss << "BEST_step" << std::setw(8) << std::setfill('0') << ts.step
                               << "_return" << return_str;
            std::string checkpoint_name = checkpoint_name_ss.str();
            
            // Save C++ header checkpoint
            {
                typename CONFIG::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                typename decltype(ts.actor_critic.actor)::template DoubleBuffer<1> actor_buffer;
                typename decltype(actor_checkpoint)::template DoubleBuffer<1> actor_checkpoint_buffer;
                
                rlt::malloc(ts.device, actor_checkpoint);
                rlt::malloc(ts.device, actor_buffer);
                rlt::malloc(ts.device, actor_checkpoint_buffer);
                
                rlt::copy(ts.device, ts.device, ts.actor_critic.actor, actor_checkpoint);
                
                std::filesystem::path actor_output_path_code = 
                    actor_output_dir / (checkpoint_name + ".h");
                
                auto actor_weights = rlt::save_code(
                    ts.device, actor_checkpoint, 
                    std::string("rl_tools::checkpoint::actor"), true
                );
                
                std::cout << "*** Saving BEST checkpoint: " << actor_output_path_code << std::endl;
                std::ofstream actor_output_file(actor_output_path_code);
                actor_output_file << actor_weights;
                
                // Export PPO-specific parameters
                actor_output_file << "\n// BEST MODEL - Validation Return: " << avg_return << "\n";
                actor_output_file << "// Saved at step: " << ts.step << "\n\n";
                
                // Log std
                actor_output_file << "namespace rl_tools::checkpoint::ppo {\n";
                actor_output_file << "    constexpr int ACTION_DIM = " << CONFIG::ACTION_DIM << ";\n";
                actor_output_file << "    constexpr float log_std[ACTION_DIM] = {";
                for (TI i = 0; i < CONFIG::ACTION_DIM; i++) {
                    actor_output_file << ts.actor_critic.log_std[i] << "f";
                    if (i < CONFIG::ACTION_DIM - 1) actor_output_file << ", ";
                }
                actor_output_file << "};\n";
                actor_output_file << "    constexpr float std[ACTION_DIM] = {";
                for (TI i = 0; i < CONFIG::ACTION_DIM; i++) {
                    actor_output_file << std::exp(ts.actor_critic.log_std[i]) << "f";
                    if (i < CONFIG::ACTION_DIM - 1) actor_output_file << ", ";
                }
                actor_output_file << "};\n";
                actor_output_file << "    constexpr float best_validation_return = " << avg_return << "f;\n";
                actor_output_file << "    constexpr int best_step = " << ts.step << ";\n";
                actor_output_file << "}\n";
                
                // Observation normalizer
                actor_output_file << "\n// Observation normalization parameters - CRITICAL FOR DEPLOYMENT!\n";
                actor_output_file << rlt::rl::algorithms::ppo::save_normalizer_code(
                    ts.obs_normalizer, 
                    "rl_tools::checkpoint"
                );
                
                // Test observation/action
                {
                    typename CONFIG::ENVIRONMENT::State state;
                    rlt::sample_initial_state(ts.device, ts.envs[0], state, ts.rng_eval);
                    
                    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::OBSERVATION_DIM>> observation;
                    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::ACTION_DIM>> action;
                    rlt::malloc(ts.device, observation);
                    rlt::malloc(ts.device, action);
                    
                    auto rng_copy = ts.rng_eval;
                    rlt::observe(ts.device, ts.env_eval, state, observation, rng_copy);
                    ts.obs_normalizer.normalize_matrix(ts.device, observation);
                    rlt::evaluate(ts.device, actor_checkpoint, observation, action, actor_checkpoint_buffer);
                    
                    actor_output_file << "\n" << rlt::save_code(ts.device, observation, 
                        std::string("rl_tools::checkpoint::observation"), true);
                    actor_output_file << "\n" << rlt::save_code(ts.device, action, 
                        std::string("rl_tools::checkpoint::action"), true);
                    
                    // Metadata
                    actor_output_file << "\n" << "namespace rl_tools::checkpoint::meta{";
                    actor_output_file << "\n" << "   " << "char name[] = \"" 
                        << ts.run_name << "_" << checkpoint_name << "\";";
                    actor_output_file << "\n" << "   " << "char algorithm[] = \"PPO\";";
                    actor_output_file << "\n" << "   " << "bool is_best = true;";
                    actor_output_file << "\n" << "   " << "float validation_return = " << avg_return << "f;";
                    actor_output_file << "\n" << "   " << "char commit_hash[] = \"" 
                        << RL_TOOLS_STRINGIFY(RL_TOOLS_COMMIT_HASH) << "\";";
                    actor_output_file << "\n" << "}";
                    
                    rlt::free(ts.device, observation);
                    rlt::free(ts.device, action);
                }
                
                rlt::free(ts.device, actor_checkpoint);
                rlt::free(ts.device, actor_buffer);
                rlt::free(ts.device, actor_checkpoint_buffer);
            }
            
            // Also copy the deployment safety header to this checkpoint
            {
                std::filesystem::path safety_header_path = 
                    actor_output_dir / "deployment_safety.h";
                std::ofstream safety_file(safety_header_path);
                safety_file << rlt::deployment::generate_deployment_safety_header();
            }
        }

        /**
         * @brief Run validation episodes
         */
        template <typename CONFIG>
        void validation(PPOTrainingState<CONFIG>& ts) {
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            
            if (ts.step % CONFIG::EVALUATION_INTERVAL == 0) {
                T total_return = 0;
                T total_length = 0;
                
                // Allocate matrices once outside the loop
                rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::OBSERVATION_DIM>> obs_matrix;
                rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, CONFIG::ACTION_DIM>> action_matrix;
                typename CONFIG::ACTOR_TYPE::template DoubleBuffer<1> single_buffer;
                
                rlt::malloc(ts.device, obs_matrix);
                rlt::malloc(ts.device, action_matrix);
                rlt::malloc(ts.device, single_buffer);
                
                for (TI ep = 0; ep < CONFIG::VALIDATION_N_EPISODES; ep++) {
                    typename CONFIG::ENVIRONMENT::State state;
                    rlt::sample_initial_state(ts.device, ts.validation_envs[ep], state, ts.rng_eval);
                    
                    T episode_return = 0;
                    TI episode_length = 0;
                    
                    for (TI step_i = 0; step_i < CONFIG::VALIDATION_MAX_EPISODE_LENGTH; step_i++) {
                        // Get observation into properly allocated matrix
                        rlt::observe(ts.device, ts.validation_envs[ep], state, obs_matrix, ts.rng_eval);
                        
                        // Apply observation normalization (matching training)
                        ts.obs_normalizer.normalize_matrix(ts.device, obs_matrix);
                        
                        // Get action (deterministic - mean only, no sampling)
                        rlt::evaluate(ts.device, ts.actor_critic.actor, obs_matrix, action_matrix, single_buffer);
                        
                        // Step environment
                        typename CONFIG::ENVIRONMENT::State next_state;
                        rlt::step(ts.device, ts.validation_envs[ep], state, action_matrix, next_state, ts.rng_eval);
                        T reward = rlt::reward(ts.device, ts.validation_envs[ep], state, action_matrix, next_state, ts.rng_eval);
                        bool terminated = rlt::terminated(ts.device, ts.validation_envs[ep], next_state, ts.rng_eval);
                        
                        episode_return += reward;
                        episode_length++;
                        state = next_state;
                        
                        if (terminated) break;
                    }
                    
                    total_return += episode_return;
                    total_length += episode_length;
                }
                
                rlt::free(ts.device, obs_matrix);
                rlt::free(ts.device, action_matrix);
                rlt::free(ts.device, single_buffer);
                
                T avg_return = total_return / CONFIG::VALIDATION_N_EPISODES;
                T avg_length = total_length / CONFIG::VALIDATION_N_EPISODES;
                
                rlt::add_scalar(ts.device, ts.device.logger, "validation/avg_return", avg_return);
                rlt::add_scalar(ts.device, ts.device.logger, "validation/avg_length", avg_length);
                rlt::add_scalar(ts.device, ts.device.logger, "validation/best_return", ts.best_validation_return);
                
                std::cout << "Validation (step " << ts.step << "): avg_return=" << avg_return 
                          << ", avg_length=" << avg_length;
                
                // Check for new best return and save checkpoint if so
                if (avg_return > ts.best_validation_return) {
                    T improvement = avg_return - ts.best_validation_return;
                    ts.best_validation_return = avg_return;
                    ts.best_return_step = ts.step;
                    
                    std::cout << " *** NEW BEST! (improved by " << improvement << ")";
                    
                    // Save best checkpoint
                    checkpoint_best(ts, avg_return);
                }
                std::cout << std::endl;
            }
        }

        /**
         * @brief Single PPO training step
         */
        template <typename CONFIG>
        void step(PPOTrainingState<CONFIG>& ts) {
            using TI = typename CONFIG::TI;
            
            // Log progress
            if (ts.step % 10 == 0) {
                std::cout << "PPO Step: " << ts.step << " (total env steps: " << ts.total_steps << ")" << std::endl;
            }
            
            // Update logger step
            rlt::set_step(ts.device, ts.device.logger, ts.step);
            
            // Perform PPO step (rollout + update)
            rlt::rl::algorithms::ppo::loop::step(ts);
            
            // Validation
            validation(ts);
            
            // Checkpointing
            checkpoint(ts);
        }

        /**
         * @brief Cleanup PPO training state
         */
        template <typename CONFIG>
        void destroy(PPOTrainingState<CONFIG>& ts) {
            rlt::rl::algorithms::ppo::loop::destroy(ts);
            rlt::free(ts.device, ts.validation_actor_buffers);
        }

    } // namespace ppo_training
} // namespace learning_to_fly

#endif // LEARNING_TO_FLY_TRAINING_PPO_H
