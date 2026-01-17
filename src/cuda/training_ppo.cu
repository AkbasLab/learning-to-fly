/**
 * @file training_ppo.cu
 * @brief GPU-accelerated PPO Training for Quadrotor Control
 * 
 * This file implements PPO training with GPU acceleration using rl_tools CUDA backend.
 * It follows the exact pattern from rl_tools/src/rl/environments/mujoco/ant/ppo/cuda/training_ppo.cu
 * 
 * GPU ACCELERATION STRATEGY:
 * - Networks (actor/critic) duplicated on GPU for forward/backward passes
 * - collect_hybrid: GPU forward pass, CPU environment stepping
 * - train_hybrid: GPU training updates
 * - Environments run on CPU (physics simulation)
 * 
 * DEPLOYMENT:
 * The trained actor checkpoint is identical to CPU training.
 * STM32 deployment code works without modification.
 */

#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
// CUDA training operations
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

// Simulator
#include <learning_to_fly/simulator/operations_cpu.h>
#include <learning_to_fly/simulator/metrics.h>

// Config
#include "config/parameters.h"
#include "config/ablation.h"

// On-policy runner (CPU-side, with hybrid extensions for GPU inference)
#if defined(RL_TOOLS_BACKEND_ENABLE_MKL) && !defined(RL_TOOLS_BACKEND_DISABLE_BLAS)
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#else
#if defined(RL_TOOLS_BACKEND_ENABLE_ACCELERATE) && !defined(RL_TOOLS_BACKEND_DISABLE_BLAS)
#include <rl_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif
// Hybrid collection extension for GPU inference
#include <rl_tools/rl/components/on_policy_runner/operations_generic_extensions.h>

// PPO algorithm
#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
// Hybrid training extension for GPU updates
#include <rl_tools/rl/algorithms/ppo/operations_generic_extensions.h>

// Running normalizer (optional)
#include <rl_tools/rl/components/running_normalizer/operations_generic.h>

// Evaluation utilities
#include <rl_tools/rl/utils/evaluation.h>

// Checkpoint persistence
#include <rl_tools/containers/persist_code.h>
#include <rl_tools/nn/parameters/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#ifdef RL_TOOLS_ENABLE_HDF5
#include <rl_tools/containers/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <highfive/H5File.hpp>
#endif

#include "helpers.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <filesystem>


// ============================================================================
// Device and Logger Configuration
// ============================================================================

#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;
#else
using LOGGER = rlt::devices::logging::CPU;
#endif

using DEV_SPEC_SUPER = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using TI_GLOBAL = typename rlt::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;

// Execution hints for parallel on-policy runner
namespace execution_hints {
    struct HINTS : rlt::rl::components::on_policy_runner::ExecutionHints<TI_GLOBAL, 16> {};
}
struct DEV_SPEC : DEV_SPEC_SUPER {
    using EXECUTION_HINTS = execution_hints::HINTS;
};

// CPU device for environments and on-policy runner
using DEVICE = rlt::DEVICE_FACTORY<DEV_SPEC>;
// GPU device for network operations
using DEVICE_GPU = rlt::DEVICE_FACTORY_GPU<rlt::devices::DefaultCUDASpecification>;

using T = float;
using TI = typename DEVICE::index_t;


// ============================================================================
// PPO Parameters (matching ant/ppo/cuda parameters but for quadrotor)
// ============================================================================

template <typename T_ABLATION_SPEC>
struct Parameters {
    using ABLATION_SPEC = T_ABLATION_SPEC;
    
    template <typename T, typename TI>
    struct environment {
        using ENVIRONMENT = typename ::parameters::environment<T, TI, ABLATION_SPEC>::ENVIRONMENT;
        static constexpr auto parameters = ::parameters::environment<T, TI, ABLATION_SPEC>::parameters;
    };
    
    template <typename T_T, typename T_TI, typename T_ENVIRONMENT>
    struct rl {
        using T = T_T;
        using TI = T_TI;
        using ENVIRONMENT = T_ENVIRONMENT;
        
        // Batch size for PPO updates
        static constexpr TI BATCH_SIZE = 256;
        
        // Actor network: obs -> 64 -> 64 -> action (same as TD3 for deployment compatibility)
        using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<
            T, TI, 
            ENVIRONMENT::OBSERVATION_DIM, 
            ENVIRONMENT::ACTION_DIM, 
            3,  // 3 layers (input + 2 hidden + output = 3 weight matrices)
            64, // Hidden dim
            rlt::nn::activation_functions::ActivationFunction::FAST_TANH,  // Hidden activation
            rlt::nn::activation_functions::FAST_TANH,  // Output activation (bounded actions)
            BATCH_SIZE
        >;
        
        // Optimizer parameters
        struct ACTOR_OPTIMIZER_PARAMETERS : rlt::nn::optimizers::adam::DefaultParametersTorch<T, TI> {
            static constexpr T ALPHA = 3e-4;
        };
        struct CRITIC_OPTIMIZER_PARAMETERS : rlt::nn::optimizers::adam::DefaultParametersTorch<T, TI> {
            static constexpr T ALPHA = 3e-4;
        };
        using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_PARAMETERS>;
        using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_PARAMETERS>;
        
        // Actor types
        using ACTOR_SPEC = rlt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
        using ACTOR_TYPE_INFERENCE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<ACTOR_SPEC>;
        
        // Checkpoint type (inference-only MLP, no optimizer state)
        using ACTOR_CHECKPOINT_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<
            T, TI, 
            ENVIRONMENT::OBSERVATION_DIM, 
            ENVIRONMENT::ACTION_DIM, 
            3, 64,
            rlt::nn::activation_functions::ActivationFunction::FAST_TANH,
            rlt::nn::activation_functions::FAST_TANH,
            1  // Batch size 1 for inference
        >;
        using ACTOR_CHECKPOINT_SPEC = rlt::nn_models::mlp::InferenceSpecification<ACTOR_CHECKPOINT_STRUCTURE_SPEC>;
        using ACTOR_CHECKPOINT_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_CHECKPOINT_SPEC>;
        
        // Critic network: obs -> 64 -> 64 -> 1 (value function)
        using CRITIC_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<
            T, TI,
            ENVIRONMENT::OBSERVATION_DIM,
            1,  // Output: scalar value
            3,  // 3 layers
            64, // Hidden dim
            rlt::nn::activation_functions::ActivationFunction::FAST_TANH,
            rlt::nn::activation_functions::IDENTITY,  // No activation on value output
            BATCH_SIZE
        >;
        using CRITIC_SPEC = rlt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;
        
        // PPO hyperparameters
        struct PPO_PARAMETERS : rlt::rl::algorithms::ppo::DefaultParameters<T, TI> {
            static constexpr TI N_EPOCHS = 10;
            static constexpr bool LEARN_ACTION_STD = true;
            static constexpr T INITIAL_ACTION_STD = 0.5;
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
            static constexpr bool NORMALIZE_ADVANTAGE = true;
            static constexpr T GAMMA = 0.99;
            static constexpr bool ADAPTIVE_LEARNING_RATE = false;
            static constexpr bool NORMALIZE_OBSERVATIONS = false;  // Quadrotor obs already normalized
        };
        
        using PPO_SPEC = rlt::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
        using PPO_TYPE = rlt::rl::algorithms::PPO<PPO_SPEC>;
        using PPO_BUFFERS_TYPE = rlt::rl::algorithms::ppo::Buffers<PPO_SPEC>;
        
        // On-policy runner configuration
        static constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 500;  // Max steps per episode
        static constexpr TI N_ENVIRONMENTS = 64;  // Parallel environments
        using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
        using ON_POLICY_RUNNER_TYPE = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
        
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;  // Steps per PPO update per env
        using ON_POLICY_RUNNER_DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
        using ON_POLICY_RUNNER_DATASET_TYPE = rlt::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;
        
        // Buffers
        using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template DoubleBuffer<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
        using ACTOR_BUFFERS = typename ACTOR_TYPE::template DoubleBuffer<BATCH_SIZE>;
        using CRITIC_BUFFERS = typename CRITIC_TYPE::template DoubleBuffer<BATCH_SIZE>;
        using CRITIC_BUFFERS_GAE = typename CRITIC_TYPE::template DoubleBuffer<ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL>;
    };
};


// ============================================================================
// Training Constants
// ============================================================================

constexpr TI BASE_SEED = 0;
constexpr TI NUM_RUNS = 1;
constexpr TI NUM_PPO_STEPS = 2500;  // Total PPO updates
constexpr TI EVALUATION_INTERVAL = 100;  // Evaluate every N PPO steps
constexpr TI NUM_EVALUATION_EPISODES = 10;
constexpr TI CHECKPOINT_INTERVAL = 500;  // Save checkpoint every N PPO steps

#ifdef RL_TOOLS_ENABLE_HDF5
constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;
#else
constexpr bool ACTOR_ENABLE_CHECKPOINTS = true;  // Always save C++ header checkpoints
#endif


// ============================================================================
// Utility Functions
// ============================================================================

std::string sanitize_file_name(const std::string& input) {
    std::string output = input;
    const std::string invalid_chars = R"(<>:"/\|?*)";
    std::replace_if(output.begin(), output.end(), [&invalid_chars](const char& c) {
        return invalid_chars.find(c) != std::string::npos;
    }, '_');
    return output;
}

template <typename ABLATION_SPEC>
std::string generate_run_name(TI seed) {
    std::stringstream run_name_ss;
    auto now = std::chrono::system_clock::now();
    auto local_time = std::chrono::system_clock::to_time_t(now);
    std::tm* tm = std::localtime(&local_time);
    run_name_ss << std::put_time(tm, "%Y_%m_%d_%H_%M_%S");
    run_name_ss << "_PPO_CUDA";
    run_name_ss << "_" << learning_to_fly::helpers::ablation_name<ABLATION_SPEC>();
    run_name_ss << "_" << std::setw(3) << std::setfill('0') << seed;
    return sanitize_file_name(run_name_ss.str());
}


// ============================================================================
// Main Training Function
// ============================================================================

template <typename T_ABLATION_SPEC>
void run_ppo_cuda() {
    using PARAMS = Parameters<T_ABLATION_SPEC>;
    using penv = typename PARAMS::template environment<T, TI>;
    using prl = typename PARAMS::template rl<T, TI, typename penv::ENVIRONMENT>;
    
    // Hybrid buffer types for CUDA acceleration
    using ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE = rlt::rl::components::on_policy_runner::CollectionEvaluationBuffer<typename prl::ON_POLICY_RUNNER_SPEC>;
    using PPO_TRAINING_HYBRID_BUFFER_TYPE = rlt::rl::algorithms::ppo::TrainingBuffersHybrid<typename prl::PPO_SPEC>;
    
    std::string actor_checkpoints_dir = "checkpoints/multirotor_ppo_cuda";
    std::string logs_dir = "logs";
    
    std::cout << "========================================\n";
    std::cout << "PPO CUDA Training for Quadrotor Control\n";
    std::cout << "========================================\n\n";
    
    for (TI run_i = 0; run_i < NUM_RUNS; ++run_i) {
        TI seed = BASE_SEED + run_i;
        std::string run_name = generate_run_name<T_ABLATION_SPEC>(seed);
        
        std::cout << "Run " << run_i << " - Seed: " << seed << "\n";
        std::cout << "Run name: " << run_name << "\n";
        std::cout << "Environments: " << prl::N_ENVIRONMENTS << "\n";
        std::cout << "Steps per env per update: " << prl::ON_POLICY_RUNNER_STEPS_PER_ENV << "\n";
        std::cout << "Total steps per update: " << prl::ON_POLICY_RUNNER_STEPS_PER_ENV * prl::N_ENVIRONMENTS << "\n";
        std::cout << "Batch size: " << prl::BATCH_SIZE << "\n\n";
        
        // Devices
        DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        DEVICE_GPU device_gpu;
        
        // Optimizers
        typename prl::ACTOR_OPTIMIZER actor_optimizer;
        typename prl::CRITIC_OPTIMIZER critic_optimizer;
        
        // RNG
        auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), seed);
        auto evaluation_rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), seed + 1000);
        
        // PPO instances (CPU and GPU)
        typename prl::PPO_TYPE ppo;
        typename prl::PPO_TYPE ppo_gpu;
        
        // Buffers
        typename prl::PPO_BUFFERS_TYPE ppo_buffers;
        typename prl::ON_POLICY_RUNNER_TYPE on_policy_runner;
        typename prl::ON_POLICY_RUNNER_DATASET_TYPE on_policy_runner_dataset;
        
        // Hybrid buffers for GPU operations
        ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE on_policy_runner_collection_eval_buffer_gpu, on_policy_runner_collection_eval_buffer_cpu;
        PPO_TRAINING_HYBRID_BUFFER_TYPE ppo_training_hybrid_buffer_cpu, ppo_training_hybrid_buffer_gpu;
        
        // GAE buffers on GPU
        static constexpr TI GAE_ALL_ROWS = prl::ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, GAE_ALL_ROWS, prl::PPO_SPEC::ENVIRONMENT::OBSERVATION_DIM>> gae_all_observations;
        rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, GAE_ALL_ROWS, 1>> gae_all_values;
        
        // Evaluation buffers
        typename prl::ACTOR_EVAL_BUFFERS actor_eval_buffers, actor_eval_buffers_gpu;
        typename prl::PPO_TYPE::SPEC::ACTOR_TYPE::template DoubleBuffer<1> actor_deterministic_eval_buffers;
        
        // Training buffers (on GPU)
        typename prl::ACTOR_BUFFERS actor_buffers;
        typename prl::CRITIC_BUFFERS critic_buffers;
        typename prl::CRITIC_BUFFERS_GAE critic_buffers_gae;
        
        // Observation normalizer (optional - disabled for quadrotor)
        rlt::rl::components::RunningNormalizer<rlt::rl::components::running_normalizer::Specification<T, TI, penv::ENVIRONMENT::OBSERVATION_DIM>> observation_normalizer;
        
        // Environments
        typename penv::ENVIRONMENT envs[prl::N_ENVIRONMENTS];
        typename penv::ENVIRONMENT evaluation_env;
        bool ui = false;
        TI next_checkpoint_id = 0;
        TI next_evaluation_id = 0;
        
        // Initialize devices
        rlt::init(device);
        rlt::init(device_gpu);
        
        // CPU allocations
        rlt::malloc(device, ppo);
        rlt::malloc(device, ppo_buffers);
        rlt::malloc(device, on_policy_runner_dataset);
        rlt::malloc(device, on_policy_runner_collection_eval_buffer_cpu);
        rlt::malloc(device, ppo_training_hybrid_buffer_cpu);
        rlt::malloc(device, on_policy_runner);
        rlt::malloc(device, actor_eval_buffers);
        rlt::malloc(device, actor_deterministic_eval_buffers);
        rlt::malloc(device, observation_normalizer);
        
        for (auto& env : envs) {
            rlt::malloc(device, env);
            env.parameters = penv::parameters;
        }
        rlt::malloc(device, evaluation_env);
        evaluation_env.parameters = penv::parameters;
        
        // GPU allocations
        rlt::malloc(device_gpu, actor_buffers);
        rlt::malloc(device_gpu, critic_buffers);
        rlt::malloc(device_gpu, critic_buffers_gae);
        rlt::malloc(device_gpu, ppo_gpu);
        rlt::malloc(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
        rlt::malloc(device_gpu, ppo_training_hybrid_buffer_gpu);
        rlt::malloc(device_gpu, actor_eval_buffers_gpu);
        rlt::malloc(device_gpu, gae_all_observations);
        rlt::malloc(device_gpu, gae_all_values);
        
        // Initialize on-policy runner
        rlt::init(device, on_policy_runner, envs, rng);
        rlt::init(device, observation_normalizer);
        rlt::init(device, ppo, actor_optimizer, critic_optimizer, rng);
        
        // Copy PPO to GPU
        rlt::copy(device, device_gpu, ppo, ppo_gpu);
        
        // Setup logging
        rlt::construct(device, device.logger, logs_dir, run_name);
        
        auto training_start = std::chrono::high_resolution_clock::now();
        
        std::cout << "Starting training for " << NUM_PPO_STEPS << " PPO updates...\n\n";
        
        // Main training loop
        for (TI ppo_step_i = 0; ppo_step_i < NUM_PPO_STEPS; ppo_step_i++) {
            // Sync PPO from GPU to CPU (for collection)
            rlt::copy(device_gpu, device, ppo_gpu, ppo);
            
            // Checkpoint
            if (ACTOR_ENABLE_CHECKPOINTS && (on_policy_runner.step / CHECKPOINT_INTERVAL >= next_checkpoint_id)) {
                std::filesystem::path actor_output_dir = std::filesystem::path(actor_checkpoints_dir) / run_name;
                try {
                    std::filesystem::create_directories(actor_output_dir);
                } catch (std::exception& e) {
                    std::cerr << "Failed to create checkpoint dir: " << e.what() << "\n";
                }
                
                std::stringstream checkpoint_name_ss;
                checkpoint_name_ss << "actor_" << std::setw(15) << std::setfill('0') << next_checkpoint_id 
                                  << "_step_" << std::setw(10) << std::setfill('0') << on_policy_runner.step;
                std::string checkpoint_name = checkpoint_name_ss.str();
                
                // Save C++ header checkpoint (for STM32 deployment)
                // Create inference-only copy for saving
                {
                    typename prl::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
                    rlt::malloc(device, actor_checkpoint);
                    
                    // Copy weights from training actor to checkpoint actor
                    // (copies underlying MLP weights, ignores log_std and optimizer state)
                    rlt::copy(device, device, 
                        static_cast<const rlt::nn_models::mlp::NeuralNetworkAdam<typename prl::ACTOR_SPEC>&>(ppo.actor),
                        actor_checkpoint);
                    
                    std::filesystem::path actor_output_path_code = actor_output_dir / (checkpoint_name + ".h");
                    std::string actor_code = rlt::save_code(device, actor_checkpoint, "rl_tools::checkpoint::actor", true);
                    std::ofstream actor_file(actor_output_path_code);
                    actor_file << actor_code;
                    std::cout << "Saved C++ checkpoint: " << actor_output_path_code << "\n";
                    
                    rlt::free(device, actor_checkpoint);
                }
                
                next_checkpoint_id++;
            }
            
            // Evaluation
            if (on_policy_runner.step / EVALUATION_INTERVAL >= next_evaluation_id) {
                auto result = rlt::evaluate(device, evaluation_env, ui, ppo.actor, 
                    rlt::rl::utils::evaluation::Specification<NUM_EVALUATION_EPISODES, prl::ON_POLICY_RUNNER_STEP_LIMIT>(), 
                    observation_normalizer.mean, observation_normalizer.std, 
                    actor_deterministic_eval_buffers, evaluation_rng);
                
                std::cout << "Evaluation @ step " << on_policy_runner.step 
                          << " - Return mean: " << result.returns_mean 
                          << " (std: " << result.returns_std << ")\n";
                
                rlt::add_histogram(device, device.logger, "evaluation/return", result.returns, decltype(result)::N_EPISODES);
                next_evaluation_id++;
            }
            
            rlt::set_step(device, device.logger, on_policy_runner.step);
            
            auto step_start = std::chrono::high_resolution_clock::now();
            
            // Collect rollout (hybrid: GPU inference, CPU environment)
            {
                rlt::collect_hybrid(device, device_gpu, on_policy_runner_dataset, on_policy_runner, 
                    ppo.actor, ppo_gpu.actor, actor_eval_buffers_gpu, 
                    on_policy_runner_collection_eval_buffer_cpu, on_policy_runner_collection_eval_buffer_gpu, 
                    observation_normalizer.mean, observation_normalizer.std, rng);
            }
            
            // Compute GAE (GPU critic forward pass)
            {
                // Note: We're not using observation normalization for quadrotor
                // since the observations are already well-scaled
                rlt::copy(device, device_gpu, on_policy_runner_dataset.all_observations, gae_all_observations);
                rlt::evaluate(device_gpu, ppo_gpu.critic, gae_all_observations, gae_all_values, critic_buffers_gae);
                rlt::copy(device_gpu, device, gae_all_values, on_policy_runner_dataset.all_values);
                rlt::estimate_generalized_advantages(device, on_policy_runner_dataset, typename prl::PPO_TYPE::SPEC::PARAMETERS{});
            }
            
            // PPO update (hybrid: GPU training)
            {
                rlt::train_hybrid(device, device_gpu, ppo, ppo_gpu, on_policy_runner_dataset, 
                    actor_optimizer, critic_optimizer, ppo_buffers, ppo_training_hybrid_buffer_gpu, 
                    actor_buffers, critic_buffers, rng);
            }
            
            // Logging
            if (ppo_step_i % 10 == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<T> training_elapsed = now - training_start;
                std::chrono::duration<T> step_elapsed = now - step_start;
                T steps_per_second_lifetime = on_policy_runner.step / training_elapsed.count();
                T steps_per_second_current = prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS * prl::ON_POLICY_RUNNER_STEPS_PER_ENV / step_elapsed.count();
                
                std::cout << "PPO step: " << std::setw(6) << ppo_step_i 
                          << " | Env step: " << std::setw(10) << on_policy_runner.step 
                          << " | Elapsed: " << std::setw(8) << std::setprecision(1) << std::fixed << training_elapsed.count() << "s"
                          << " | Steps/s (lifetime): " << std::setw(8) << std::setprecision(0) << steps_per_second_lifetime
                          << " | Steps/s (current): " << std::setw(8) << std::setprecision(0) << steps_per_second_current
                          << "\n";
            }
        }
        
        // Final checkpoint
        {
            std::filesystem::path actor_output_dir = std::filesystem::path(actor_checkpoints_dir) / run_name;
            std::filesystem::create_directories(actor_output_dir);
            
            typename prl::ACTOR_CHECKPOINT_TYPE actor_checkpoint;
            rlt::malloc(device, actor_checkpoint);
            rlt::copy(device, device, 
                static_cast<const rlt::nn_models::mlp::NeuralNetworkAdam<typename prl::ACTOR_SPEC>&>(ppo.actor),
                actor_checkpoint);
            
            std::filesystem::path actor_output_path_code = actor_output_dir / "actor_final.h";
            std::string actor_code = rlt::save_code(device, actor_checkpoint, "rl_tools::checkpoint::actor", true);
            std::ofstream actor_file(actor_output_path_code);
            actor_file << actor_code;
            std::cout << "\nSaved final checkpoint: " << actor_output_path_code << "\n";
            
            rlt::free(device, actor_checkpoint);
        }
        
        auto training_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> total_time = training_end - training_start;
        
        std::cout << "\n========================================\n";
        std::cout << "Training completed!\n";
        std::cout << "Total time: " << total_time.count() << "s\n";
        std::cout << "Total environment steps: " << on_policy_runner.step << "\n";
        std::cout << "Average steps/s: " << on_policy_runner.step / total_time.count() << "\n";
        std::cout << "========================================\n";
        
        // Cleanup CPU
        rlt::free(device, ppo);
        rlt::free(device, ppo_buffers);
        rlt::free(device, on_policy_runner_dataset);
        rlt::free(device, on_policy_runner_collection_eval_buffer_cpu);
        rlt::free(device, ppo_training_hybrid_buffer_cpu);
        rlt::free(device, on_policy_runner);
        rlt::free(device, actor_eval_buffers);
        rlt::free(device, observation_normalizer);
        // Note: Multirotor environments don't need explicit free (no dynamic allocation)
        
        // Cleanup GPU
        rlt::free(device_gpu, actor_buffers);
        rlt::free(device_gpu, critic_buffers);
        rlt::free(device_gpu, critic_buffers_gae);
        rlt::free(device_gpu, ppo_gpu);
        rlt::free(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
        rlt::free(device_gpu, ppo_training_hybrid_buffer_gpu);
        rlt::free(device_gpu, actor_eval_buffers_gpu);
        rlt::free(device_gpu, gae_all_observations);
        rlt::free(device_gpu, gae_all_values);
    }
}


// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "PPO CUDA Training for Learning to Fly\n";
    std::cout << "Using rl_tools hybrid CPU/GPU execution\n\n";
    
    run_ppo_cuda<learning_to_fly::config::DEFAULT_ABLATION_SPEC>();
    
    return 0;
}
