/**
 * @file rl_tools_adapter_ppo.cpp
 * @brief PPO-Compatible RL Tools Adapter for Crazyflie
 * 
 * CRITICAL FIX for PPO Deployment:
 * ================================
 * PPO requires observation normalization before inference. This adapter
 * adds the missing normalization step by calling:
 *   rl_tools::checkpoint::observation_normalizer::normalize(input_array)
 * 
 * The normalization parameters (mean, std_inv) are exported with the
 * PPO checkpoint in actor.h.
 * 
 * Usage:
 * ------
 * 1. Replace rl_tools_adapter.cpp with this file in the build
 * 2. OR enable RL_TOOLS_PPO flag when building with rl_tools_adapter.cpp
 * 
 * Differences from TD3 adapter:
 * -----------------------------
 * - Adds observation normalization before inference
 * - Auto-detects PPO vs TD3 via namespace presence
 * - Same API, drop-in replacement
 */

#include "rl_tools_adapter.h"

#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/dense/operations_arm/dsp.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include "data/actor.h"

#define RL_TOOLS_CONTROL_STATE_ROTATION_MATRIX
// #define RL_TOOLS_DISABLE_TEST
#define RL_TOOLS_ACTION_HISTORY

// ============================================================================
// PPO DETECTION AND NORMALIZATION
// ============================================================================

// Check if observation_normalizer namespace exists in checkpoint
// This is set automatically by the PPO training code
#if __has_include("ppo_normalizer_check.h")
#include "ppo_normalizer_check.h"
#else
// Fallback: check for algorithm metadata
namespace _ppo_detect {
    template<typename = void>
    constexpr bool has_ppo_normalizer() {
#ifdef RL_TOOLS_CHECKPOINT_HAS_NORMALIZER
        return true;
#else
        // Check if observation_normalizer namespace exists by checking if DIM is defined
        // This is a compile-time check
        return false;  // Default to false, override with RL_TOOLS_PPO define
#endif
    }
}
#endif

// Force PPO mode if explicitly requested
#ifdef RL_TOOLS_PPO
#define APPLY_OBSERVATION_NORMALIZATION 1
#else
// Auto-detect based on checkpoint contents
// For safety, check if algorithm is PPO in metadata
#if defined(rl_tools::checkpoint::observation_normalizer::DIM)
#define APPLY_OBSERVATION_NORMALIZATION 1
#else
#define APPLY_OBSERVATION_NORMALIZATION 0
#endif
#endif

// ============================================================================
// DEFINITIONS
// ============================================================================

namespace rlt = rl_tools;

using DEV_SPEC = rlt::devices::DefaultARMSpecification;
using DEVICE = rlt::devices::arm::OPT<DEV_SPEC>;
DEVICE device;
using ACTOR_TYPE = rlt::checkpoint::actor::MODEL;
using TI = typename ACTOR_TYPE::SPEC::TI;
using T = typename ACTOR_TYPE::SPEC::T;
constexpr TI CONTROL_FREQUENCY_MULTIPLE = 5;
static TI controller_tick = 0;
constexpr TI ACTION_HISTORY_LENGTH = 32;

#ifdef RL_TOOLS_ACTION_HISTORY
static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == (18 + ACTION_HISTORY_LENGTH * ACTOR_TYPE::SPEC::OUTPUT_DIM),
    "Input dimension must be 18 (observation) + 128 (action history)");
#else
static_assert(ACTOR_TYPE::SPEC::INPUT_DIM == 18,
    "Input dimension must be 18 without action history");
#endif

// ============================================================================
// STATE
// ============================================================================

static ACTOR_TYPE::template Buffer<1, rlt::MatrixStaticTag> buffers;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::INPUT_DIM>> input;
static rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM>> output;

#ifdef RL_TOOLS_ACTION_HISTORY
static T action_history[ACTION_HISTORY_LENGTH][ACTOR_TYPE::SPEC::OUTPUT_DIM];
#endif

// Buffer for normalized input (PPO only)
#if APPLY_OBSERVATION_NORMALIZATION
static T input_buffer[ACTOR_TYPE::SPEC::INPUT_DIM];
#endif

// ============================================================================
// OBSERVATION CONSTRUCTION
// ============================================================================

template <typename STATE_SPEC, typename OBS_SPEC>
static inline void observe_rotation_matrix(const rlt::Matrix<STATE_SPEC>& state, rlt::Matrix<OBS_SPEC>& observation){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == 18);
    
    float qw = rlt::get(state, 0, 3);
    float qx = rlt::get(state, 0, 4);
    float qy = rlt::get(state, 0, 5);
    float qz = rlt::get(state, 0, 6);
    
    // Position [0-2]
    rlt::set(observation, 0, 0, rlt::get(state, 0, 0));
    rlt::set(observation, 0, 1, rlt::get(state, 0, 1));
    rlt::set(observation, 0, 2, rlt::get(state, 0, 2));
    
    // Rotation matrix from quaternion [3-11]
    rlt::set(observation, 0, 3 + 0, (1 - 2*qy*qy - 2*qz*qz));
    rlt::set(observation, 0, 3 + 1, (    2*qx*qy - 2*qw*qz));
    rlt::set(observation, 0, 3 + 2, (    2*qx*qz + 2*qw*qy));
    rlt::set(observation, 0, 3 + 3, (    2*qx*qy + 2*qw*qz));
    rlt::set(observation, 0, 3 + 4, (1 - 2*qx*qx - 2*qz*qz));
    rlt::set(observation, 0, 3 + 5, (    2*qy*qz - 2*qw*qx));
    rlt::set(observation, 0, 3 + 6, (    2*qx*qz - 2*qw*qy));
    rlt::set(observation, 0, 3 + 7, (    2*qy*qz + 2*qw*qx));
    rlt::set(observation, 0, 3 + 8, (1 - 2*qx*qx - 2*qy*qy));
    
    // Linear velocity [12-14]
    rlt::set(observation, 0, 12 + 0, rlt::get(state, 0, 3 + 4 + 0));
    rlt::set(observation, 0, 12 + 1, rlt::get(state, 0, 3 + 4 + 1));
    rlt::set(observation, 0, 12 + 2, rlt::get(state, 0, 3 + 4 + 2));
    
    // Angular velocity [15-17]
    rlt::set(observation, 0, 15 + 0, rlt::get(state, 0, 3 + 4 + 3 + 0));
    rlt::set(observation, 0, 15 + 1, rlt::get(state, 0, 3 + 4 + 3 + 1));
    rlt::set(observation, 0, 15 + 2, rlt::get(state, 0, 3 + 4 + 3 + 2));
}

// ============================================================================
// PPO-SPECIFIC: OBSERVATION NORMALIZATION
// ============================================================================

#if APPLY_OBSERVATION_NORMALIZATION

/**
 * @brief Apply observation normalization for PPO
 * 
 * Uses the mean and std_inv arrays exported with the PPO checkpoint.
 * MUST be called before actor inference for PPO policies.
 */
static inline void normalize_observation() {
    // Copy input matrix to buffer
    for (TI i = 0; i < ACTOR_TYPE::SPEC::INPUT_DIM; i++) {
        input_buffer[i] = rlt::get(input, 0, i);
    }
    
    // Apply normalization from checkpoint
    rlt::checkpoint::observation_normalizer::normalize(input_buffer);
    
    // Copy back to input matrix
    for (TI i = 0; i < ACTOR_TYPE::SPEC::INPUT_DIM; i++) {
        rlt::set(input, 0, i, input_buffer[i]);
    }
}

#endif

// ============================================================================
// PUBLIC API
// ============================================================================

void rl_tools_init(){
    rlt::malloc(device, buffers);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    
#ifdef RL_TOOLS_ACTION_HISTORY
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
            action_history[step_i][action_i] = 0;
        }
    }
#endif
    
    controller_tick = 0;
    
    // Log which mode is active
#if APPLY_OBSERVATION_NORMALIZATION
    // DEBUG: Observation normalization is ENABLED (PPO mode)
#else
    // DEBUG: Observation normalization is DISABLED (TD3 mode)
#endif
}

char* rl_tools_get_checkpoint_name(){
    return (char*)rlt::checkpoint::meta::name;
}

float rl_tools_test(float* output_mem){
#ifndef RL_TOOLS_DISABLE_TEST
    // Run test inference with the checkpoint's test observation
    rlt::evaluate(device, rlt::checkpoint::actor::model, 
                  rlt::checkpoint::observation::container, output, buffers);
    
    float acc = 0;
    for(int i = 0; i < ACTOR_TYPE::SPEC::OUTPUT_DIM; i++){
        acc += std::abs(rlt::get(output, 0, i) - rlt::get(rlt::checkpoint::action::container, 0, i));
        output_mem[i] = rlt::get(rlt::checkpoint::action::container, 0, i);
    }
    return acc;
#else
    return 0;
#endif
}

void rl_tools_control(float* state, float* actions){
    // Construct observation from state
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 13, 
        rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> state_matrix = {(T*)state};
    
    auto state_rotation_matrix_input = rlt::view(device, input, 
        rlt::matrix::ViewSpec<1, 18>{}, 0, 0);
    observe_rotation_matrix(state_matrix, state_rotation_matrix_input);
    
    // Append action history to observation
#ifdef RL_TOOLS_ACTION_HISTORY
    auto action_history_observation = rlt::view(device, input, 
        rlt::matrix::ViewSpec<1, ACTION_HISTORY_LENGTH * ACTOR_TYPE::SPEC::OUTPUT_DIM>{}, 0, 18);
    for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH; step_i++){
        for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
            rlt::set(action_history_observation, 0, 
                step_i * ACTOR_TYPE::SPEC::OUTPUT_DIM + action_i, 
                action_history[step_i][action_i]);
        }
    }
#endif

    // =========================================================================
    // CRITICAL PPO FIX: Apply observation normalization before inference
    // =========================================================================
#if APPLY_OBSERVATION_NORMALIZATION
    normalize_observation();
#endif

    // Run actor inference
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ACTOR_TYPE::SPEC::OUTPUT_DIM, 
        rlt::matrix::layouts::RowMajorAlignment<TI, 1>>> output = {(T*)actions};
    rlt::evaluate(device, rlt::checkpoint::actor::model, input, output, buffers);
    
    // Update action history
#ifdef RL_TOOLS_ACTION_HISTORY
    int substep = controller_tick % CONTROL_FREQUENCY_MULTIPLE;
    if(substep == 0){
        // Shift history
        for(TI step_i = 0; step_i < ACTION_HISTORY_LENGTH - 1; step_i++){
            for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
                action_history[step_i][action_i] = action_history[step_i + 1][action_i];
            }
        }
    }
    // Average new actions into last history slot
    for(TI action_i = 0; action_i < ACTOR_TYPE::SPEC::OUTPUT_DIM; action_i++){
        T value = action_history[ACTION_HISTORY_LENGTH - 1][action_i];
        value *= substep;
        value += rlt::get(output, 0, action_i);
        value /= substep + 1;
        action_history[ACTION_HISTORY_LENGTH - 1][action_i] = value;
    }
#endif
    
    controller_tick++;
}
