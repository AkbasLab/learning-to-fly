/**
 * @file training_ppo.cpp
 * @brief PPO Training Entry Point for Quadrotor Control
 * 
 * This file provides the main() function for PPO-based training.
 * It can be used as a drop-in replacement for training.cpp.
 * 
 * USAGE:
 *   ./training_ppo_headless  # Train with PPO
 * 
 * OUTPUT:
 *   Checkpoints saved to: checkpoints/multirotor_ppo/<run_name>/
 *   Logs saved to: logs/<run_name>/
 * 
 * DEPLOYMENT:
 *   The trained actor checkpoint (.h file) can be used directly with
 *   the STM32 firmware. The checkpoint format is identical to TD3.
 */

#include "training_ppo.h"

#include <chrono>
#include <iostream>


/**
 * @brief Run PPO training with given ablation specification
 */
template <typename T_ABLATION_SPEC>
void run_ppo() {
    using namespace learning_to_fly::config::ppo;
    
    using CONFIG = PPOConfig<T_ABLATION_SPEC>;
    using TI = typename CONFIG::TI;
    
#ifdef LEARNING_TO_FLY_IN_SECONDS_BENCHMARK
    constexpr TI NUM_RUNS = 1;
#else
    constexpr TI NUM_RUNS = 1;
#endif
    
    for (TI run_i = 0; run_i < NUM_RUNS; run_i++) {
        std::cout << "========================================\n";
        std::cout << "PPO Training Run " << run_i << "\n";
        std::cout << "========================================\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Initialize training state (on heap due to large size with many envs)
        auto* ts = new learning_to_fly::PPOTrainingState<CONFIG>();
        learning_to_fly::ppo_training::init(*ts, run_i);
        
        std::cout << "Training for " << CONFIG::STEP_LIMIT << " PPO updates...\n";
        std::cout << "Total environment steps per update: " 
                  << CONFIG::ROLLOUT_STEPS * CONFIG::N_ENVIRONMENTS << "\n\n";
        
        // Training loop
        for (TI step_i = 0; step_i < CONFIG::STEP_LIMIT; step_i++) {
            learning_to_fly::ppo_training::step(*ts);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        
        std::cout << "\n========================================\n";
        std::cout << "PPO Training completed!\n";
        std::cout << "Total time: " << duration.count() << "s\n";
        std::cout << "Total environment steps: " << ts->total_steps << "\n";
        std::cout << "Steps per second: " << ts->total_steps / duration.count() << "\n";
        std::cout << "========================================\n\n";
        
        // Cleanup
        learning_to_fly::ppo_training::destroy(*ts);
        delete ts;
    }
}


int main() {
    std::cout << "================================================\n";
    std::cout << "Learning to Fly - PPO Training\n";
    std::cout << "================================================\n\n";
    
    std::cout << "PPO (Proximal Policy Optimization) training for\n";
    std::cout << "quadrotor flight control.\n\n";
    
    std::cout << "Trained policy will be saved as C++ header files\n";
    std::cout << "compatible with STM32 deployment.\n\n";
    
    // Run with default ablation spec
    run_ppo<learning_to_fly::config::DEFAULT_ABLATION_SPEC>();
    
    return 0;
}
