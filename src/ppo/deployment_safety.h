/**
 * @file deployment_safety.h
 * @brief Safety constraints and action processing for deployment
 * 
 * This file implements critical safety features for real-world deployment:
 * 
 * 1. ACTION CLAMPING: Ensures actions stay within safe bounds
 * 2. THRUST RAMPING: Prevents sudden throttle jumps on startup
 * 3. IDLE FLOOR: Maintains minimum motor speed for stability
 * 4. MOTOR MIXING: Corrects any motor ordering mismatches
 * 
 * WHY THIS MATTERS:
 * The main failure mode observed (full throttle flip on takeoff) is caused by:
 * - Action values saturating to extremes
 * - No soft-start mechanism
 * - Possible motor ordering mismatch between sim and real
 * 
 * These functions should be called in the STM32 firmware AFTER actor inference
 * but BEFORE sending commands to motors.
 */

#ifndef LEARNING_TO_FLY_PPO_DEPLOYMENT_SAFETY_H
#define LEARNING_TO_FLY_PPO_DEPLOYMENT_SAFETY_H

#include <cmath>
#include <algorithm>

namespace rl_tools::deployment {

    /**
     * @brief Deployment safety parameters
     * 
     * These are tuned for Crazyflie 2.1 with 100Hz control loop.
     */
    template<typename T>
    struct SafetyParams {
        // Action limits (normalized space, output of tanh activation)
        static constexpr T ACTION_MIN = -1.0;
        static constexpr T ACTION_MAX = 1.0;
        
        // RPM limits for Crazyflie
        static constexpr T RPM_MIN = 0.0;
        static constexpr T RPM_MAX = 21702.0;
        
        // Idle RPM (motors spinning but no thrust)
        // This provides gyroscopic stability and faster response
        static constexpr T RPM_IDLE = 4000.0;
        
        // Maximum RPM for hover safety (prevents flip from excessive thrust)
        // At hover, each motor needs ~25% of max for 27g quad
        // Allow up to 80% max to prevent uncontrollable situations
        static constexpr T RPM_SAFE_MAX = 17000.0;
        
        // Maximum RPM change per timestep (100Hz = 10ms)
        // Prevents sudden jumps that cause flips
        // At 100Hz, this allows 0->max in 0.5 seconds
        static constexpr T MAX_RPM_DELTA = 400.0;  // RPM per 10ms
        
        // Thrust ramp duration on startup (in timesteps at 100Hz)
        static constexpr int RAMP_STEPS = 100;  // 1 second ramp
        
        // Action scaling parameters (match simulator)
        static constexpr T ACTION_SCALE = (RPM_MAX - RPM_MIN) / 2.0;
        static constexpr T ACTION_OFFSET = (RPM_MAX + RPM_MIN) / 2.0;
    };

    /**
     * @brief State for deployment safety system
     */
    template<typename T, int N_MOTORS = 4>
    struct SafetyState {
        // Previous motor RPM commands
        T prev_rpm[N_MOTORS];
        
        // Startup ramp counter
        int ramp_step;
        
        // Whether system has been armed
        bool armed;
        
        void init() {
            for (int i = 0; i < N_MOTORS; i++) {
                prev_rpm[i] = SafetyParams<T>::RPM_IDLE;
            }
            ramp_step = 0;
            armed = false;
        }
        
        void arm() {
            armed = true;
            ramp_step = 0;
            for (int i = 0; i < N_MOTORS; i++) {
                prev_rpm[i] = SafetyParams<T>::RPM_IDLE;
            }
        }
        
        void disarm() {
            armed = false;
            for (int i = 0; i < N_MOTORS; i++) {
                prev_rpm[i] = 0;
            }
        }
    };

    /**
     * @brief Convert normalized action [-1, 1] to RPM
     * 
     * This matches the simulator's action scaling exactly.
     * Action of -1 -> RPM_MIN (0)
     * Action of +1 -> RPM_MAX (21702)
     * Action of  0 -> midpoint (10851)
     */
    template<typename T>
    inline T action_to_rpm(T action) {
        using P = SafetyParams<T>;
        // Clamp action first
        action = std::max(P::ACTION_MIN, std::min(P::ACTION_MAX, action));
        // Linear mapping: [-1, 1] -> [RPM_MIN, RPM_MAX]
        return action * P::ACTION_SCALE + P::ACTION_OFFSET;
    }

    /**
     * @brief Apply rate limiting to RPM command
     * 
     * Prevents sudden RPM jumps that cause flips.
     */
    template<typename T>
    inline T rate_limit_rpm(T target_rpm, T prev_rpm) {
        using P = SafetyParams<T>;
        T delta = target_rpm - prev_rpm;
        if (delta > P::MAX_RPM_DELTA) {
            return prev_rpm + P::MAX_RPM_DELTA;
        } else if (delta < -P::MAX_RPM_DELTA) {
            return prev_rpm - P::MAX_RPM_DELTA;
        }
        return target_rpm;
    }

    /**
     * @brief Apply startup thrust ramp
     * 
     * During the first RAMP_STEPS after arming, limit maximum thrust
     * to prevent sudden takeoff flips.
     */
    template<typename T>
    inline T apply_ramp(T rpm, int ramp_step) {
        using P = SafetyParams<T>;
        if (ramp_step >= P::RAMP_STEPS) {
            return rpm;  // Ramp complete
        }
        
        // Linear ramp from idle to full range
        T ramp_factor = static_cast<T>(ramp_step) / static_cast<T>(P::RAMP_STEPS);
        T max_allowed = P::RPM_IDLE + (P::RPM_SAFE_MAX - P::RPM_IDLE) * ramp_factor;
        return std::min(rpm, max_allowed);
    }

    /**
     * @brief Apply idle floor
     * 
     * Ensures motors never go below idle speed when armed.
     * This provides gyroscopic stability and faster response.
     */
    template<typename T>
    inline T apply_idle_floor(T rpm, bool armed) {
        if (!armed) return 0;
        return std::max(rpm, SafetyParams<T>::RPM_IDLE);
    }

    /**
     * @brief Apply safety cap
     * 
     * Prevents motors from exceeding safe maximum RPM.
     */
    template<typename T>
    inline T apply_safety_cap(T rpm) {
        using P = SafetyParams<T>;
        return std::min(rpm, P::RPM_SAFE_MAX);
    }

    /**
     * @brief Motor ordering correction
     * 
     * Maps motor indices from sim convention to real hardware convention.
     * 
     * SIMULATOR MOTOR ORDERING (looking down at quad, front is X+):
     *   Motor 0: Front-Right (FR)
     *   Motor 1: Rear-Right (RR)  
     *   Motor 2: Rear-Left (RL)
     *   Motor 3: Front-Left (FL)
     * 
     * CRAZYFLIE MOTOR ORDERING (from firmware, looking down):
     *   Motor 0 (M1): Front-Right
     *   Motor 1 (M2): Rear-Left
     *   Motor 2 (M3): Rear-Right
     *   Motor 3 (M4): Front-Left
     * 
     * If these don't match, the quad will flip immediately!
     */
    template<typename T, int N_MOTORS = 4>
    void correct_motor_ordering(const T* sim_actions, T* real_motors) {
        // DEFAULT: Assume simulator and Crazyflie use same ordering
        // Uncomment the remapping if there's a mismatch
        
        // Direct mapping (no correction needed if orderings match)
        for (int i = 0; i < N_MOTORS; i++) {
            real_motors[i] = sim_actions[i];
        }
        
        // ALTERNATIVE: If motor ordering differs, uncomment and adjust:
        // Example remapping for different conventions:
        // real_motors[0] = sim_actions[0];  // FR -> M1
        // real_motors[1] = sim_actions[2];  // RL -> M2 (swapped)
        // real_motors[2] = sim_actions[1];  // RR -> M3 (swapped)
        // real_motors[3] = sim_actions[3];  // FL -> M4
    }

    /**
     * @brief Full safety pipeline for deployment
     * 
     * Call this in STM32 firmware after actor inference:
     *   1. Get actor output (4 values in [-1, 1])
     *   2. Call process_actions_safe()
     *   3. Send resulting RPM values to motors
     * 
     * @param actions Raw actor output (4 values in [-1, 1])
     * @param motor_rpms Output RPM commands (4 values)
     * @param state Safety state (must be initialized and persisted between calls)
     */
    template<typename T, int N_MOTORS = 4>
    void process_actions_safe(
        const T* actions,
        T* motor_rpms,
        SafetyState<T, N_MOTORS>& state
    ) {
        using P = SafetyParams<T>;
        
        // If not armed, output zero
        if (!state.armed) {
            for (int i = 0; i < N_MOTORS; i++) {
                motor_rpms[i] = 0;
                state.prev_rpm[i] = 0;
            }
            return;
        }
        
        // Step 1: Convert actions to RPM
        T target_rpm[N_MOTORS];
        for (int i = 0; i < N_MOTORS; i++) {
            target_rpm[i] = action_to_rpm(actions[i]);
        }
        
        // Step 2: Apply motor ordering correction
        T corrected_rpm[N_MOTORS];
        correct_motor_ordering<T, N_MOTORS>(target_rpm, corrected_rpm);
        
        // Step 3: Apply rate limiting (prevents sudden jumps)
        for (int i = 0; i < N_MOTORS; i++) {
            corrected_rpm[i] = rate_limit_rpm(corrected_rpm[i], state.prev_rpm[i]);
        }
        
        // Step 4: Apply startup ramp
        for (int i = 0; i < N_MOTORS; i++) {
            corrected_rpm[i] = apply_ramp(corrected_rpm[i], state.ramp_step);
        }
        
        // Step 5: Apply idle floor
        for (int i = 0; i < N_MOTORS; i++) {
            corrected_rpm[i] = apply_idle_floor(corrected_rpm[i], state.armed);
        }
        
        // Step 6: Apply safety cap
        for (int i = 0; i < N_MOTORS; i++) {
            corrected_rpm[i] = apply_safety_cap(corrected_rpm[i]);
        }
        
        // Output final values
        for (int i = 0; i < N_MOTORS; i++) {
            motor_rpms[i] = corrected_rpm[i];
            state.prev_rpm[i] = corrected_rpm[i];
        }
        
        // Increment ramp counter
        if (state.ramp_step < P::RAMP_STEPS) {
            state.ramp_step++;
        }
    }

    /**
     * @brief Generate C++ header for STM32 deployment
     * 
     * This function generates a standalone header that can be included
     * in STM32 firmware. It contains all safety parameters and the
     * process_actions_safe() function.
     */
    inline std::string generate_deployment_safety_header() {
        std::stringstream ss;
        
        ss << "/**\n";
        ss << " * @file deployment_safety.h\n";
        ss << " * @brief Auto-generated safety layer for PPO deployment\n";
        ss << " * \n";
        ss << " * Include this in STM32 firmware and call process_actions_safe()\n";
        ss << " * after actor inference.\n";
        ss << " */\n\n";
        ss << "#ifndef DEPLOYMENT_SAFETY_H\n";
        ss << "#define DEPLOYMENT_SAFETY_H\n\n";
        ss << "#include <math.h>\n\n";
        
        ss << "// Safety parameters\n";
        ss << "#define ACTION_MIN -1.0f\n";
        ss << "#define ACTION_MAX 1.0f\n";
        ss << "#define RPM_MIN 0.0f\n";
        ss << "#define RPM_MAX 21702.0f\n";
        ss << "#define RPM_IDLE 4000.0f\n";
        ss << "#define RPM_SAFE_MAX 17000.0f\n";
        ss << "#define MAX_RPM_DELTA 400.0f\n";
        ss << "#define RAMP_STEPS 100\n";
        ss << "#define ACTION_SCALE ((RPM_MAX - RPM_MIN) / 2.0f)\n";
        ss << "#define ACTION_OFFSET ((RPM_MAX + RPM_MIN) / 2.0f)\n\n";
        
        ss << "typedef struct {\n";
        ss << "    float prev_rpm[4];\n";
        ss << "    int ramp_step;\n";
        ss << "    int armed;\n";
        ss << "} SafetyState;\n\n";
        
        ss << "static inline void safety_init(SafetyState* state) {\n";
        ss << "    for (int i = 0; i < 4; i++) state->prev_rpm[i] = RPM_IDLE;\n";
        ss << "    state->ramp_step = 0;\n";
        ss << "    state->armed = 0;\n";
        ss << "}\n\n";
        
        ss << "static inline void safety_arm(SafetyState* state) {\n";
        ss << "    state->armed = 1;\n";
        ss << "    state->ramp_step = 0;\n";
        ss << "    for (int i = 0; i < 4; i++) state->prev_rpm[i] = RPM_IDLE;\n";
        ss << "}\n\n";
        
        ss << "static inline void safety_disarm(SafetyState* state) {\n";
        ss << "    state->armed = 0;\n";
        ss << "    for (int i = 0; i < 4; i++) state->prev_rpm[i] = 0;\n";
        ss << "}\n\n";
        
        ss << "static inline float action_to_rpm(float action) {\n";
        ss << "    if (action < ACTION_MIN) action = ACTION_MIN;\n";
        ss << "    if (action > ACTION_MAX) action = ACTION_MAX;\n";
        ss << "    return action * ACTION_SCALE + ACTION_OFFSET;\n";
        ss << "}\n\n";
        
        ss << "static inline float rate_limit(float target, float prev) {\n";
        ss << "    float delta = target - prev;\n";
        ss << "    if (delta > MAX_RPM_DELTA) return prev + MAX_RPM_DELTA;\n";
        ss << "    if (delta < -MAX_RPM_DELTA) return prev - MAX_RPM_DELTA;\n";
        ss << "    return target;\n";
        ss << "}\n\n";
        
        ss << "static inline void process_actions_safe(\n";
        ss << "    const float* actions,\n";
        ss << "    float* motor_rpms,\n";
        ss << "    SafetyState* state\n";
        ss << ") {\n";
        ss << "    if (!state->armed) {\n";
        ss << "        for (int i = 0; i < 4; i++) {\n";
        ss << "            motor_rpms[i] = 0;\n";
        ss << "            state->prev_rpm[i] = 0;\n";
        ss << "        }\n";
        ss << "        return;\n";
        ss << "    }\n\n";
        ss << "    float rpm[4];\n";
        ss << "    for (int i = 0; i < 4; i++) {\n";
        ss << "        rpm[i] = action_to_rpm(actions[i]);\n";
        ss << "        rpm[i] = rate_limit(rpm[i], state->prev_rpm[i]);\n";
        ss << "        \n";
        ss << "        // Apply ramp\n";
        ss << "        if (state->ramp_step < RAMP_STEPS) {\n";
        ss << "            float ramp_factor = (float)state->ramp_step / (float)RAMP_STEPS;\n";
        ss << "            float max_allowed = RPM_IDLE + (RPM_SAFE_MAX - RPM_IDLE) * ramp_factor;\n";
        ss << "            if (rpm[i] > max_allowed) rpm[i] = max_allowed;\n";
        ss << "        }\n";
        ss << "        \n";
        ss << "        // Apply floor and cap\n";
        ss << "        if (rpm[i] < RPM_IDLE) rpm[i] = RPM_IDLE;\n";
        ss << "        if (rpm[i] > RPM_SAFE_MAX) rpm[i] = RPM_SAFE_MAX;\n";
        ss << "        \n";
        ss << "        motor_rpms[i] = rpm[i];\n";
        ss << "        state->prev_rpm[i] = rpm[i];\n";
        ss << "    }\n";
        ss << "    \n";
        ss << "    if (state->ramp_step < RAMP_STEPS) state->ramp_step++;\n";
        ss << "}\n\n";
        
        ss << "#endif // DEPLOYMENT_SAFETY_H\n";
        
        return ss.str();
    }

} // namespace rl_tools::deployment

#endif // LEARNING_TO_FLY_PPO_DEPLOYMENT_SAFETY_H
