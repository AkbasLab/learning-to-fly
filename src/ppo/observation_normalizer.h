/**
 * @file observation_normalizer.h
 * @brief Running observation normalization for PPO
 * 
 * PPO is highly sensitive to observation scale. This normalizer maintains
 * running statistics (mean, variance) and normalizes observations during
 * both training and deployment.
 * 
 * CRITICAL FOR DEPLOYMENT:
 * The normalization parameters (mean, std) MUST be exported to the
 * STM32 firmware along with the actor weights. The firmware must apply
 * the SAME normalization before inference.
 * 
 * WHY THIS MATTERS:
 * Without matching normalization, the policy sees different input distributions
 * during training vs deployment, causing erratic behavior (e.g., full throttle).
 */

#ifndef LEARNING_TO_FLY_PPO_OBSERVATION_NORMALIZER_H
#define LEARNING_TO_FLY_PPO_OBSERVATION_NORMALIZER_H

#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace rl_tools::rl::algorithms::ppo {

    /**
     * @brief Running mean/variance estimator using Welford's algorithm
     * 
     * Numerically stable online algorithm for computing mean and variance.
     * Used to normalize observations for PPO training stability.
     */
    template<typename T, typename TI, TI DIM>
    struct ObservationNormalizer {
        // Running statistics
        T mean[DIM];
        T var[DIM];
        T count;
        
        // Clip value for normalized observations (prevents extreme values)
        static constexpr T CLIP_VALUE = 10.0;
        
        // Minimum variance to prevent division by zero
        static constexpr T MIN_VAR = 1e-8;
        
        // Whether normalizer is enabled
        bool enabled;
        
        /**
         * @brief Initialize normalizer with zero mean and unit variance
         */
        void init() {
            for (TI i = 0; i < DIM; i++) {
                mean[i] = 0;
                var[i] = 1;
            }
            count = 0;
            enabled = true;
        }
        
        /**
         * @brief Update running statistics with a new observation
         * 
         * Uses Welford's online algorithm for numerical stability.
         */
        void update(const T* observation) {
            count += 1;
            T delta[DIM];
            
            for (TI i = 0; i < DIM; i++) {
                delta[i] = observation[i] - mean[i];
                mean[i] += delta[i] / count;
                
                // Update variance (using Welford's algorithm)
                T delta2 = observation[i] - mean[i];
                var[i] += (delta[i] * delta2 - var[i]) / count;
            }
        }
        
        /**
         * @brief Update with batch of observations
         */
        template<typename MATRIX>
        void update_batch(const MATRIX& observations, TI batch_size) {
            for (TI b = 0; b < batch_size; b++) {
                T obs[DIM];
                for (TI i = 0; i < DIM; i++) {
                    obs[i] = rlt::get(observations, b, i);
                }
                update(obs);
            }
        }
        
        /**
         * @brief Normalize observation in-place
         * 
         * norm_obs = clip((obs - mean) / sqrt(var + epsilon), -CLIP_VALUE, CLIP_VALUE)
         */
        void normalize(T* observation) const {
            if (!enabled) return;
            
            for (TI i = 0; i < DIM; i++) {
                T std = std::sqrt(std::max(var[i], MIN_VAR));
                observation[i] = (observation[i] - mean[i]) / std;
                // Clip to prevent extreme values
                observation[i] = std::max(-CLIP_VALUE, std::min(CLIP_VALUE, observation[i]));
            }
        }
        
        /**
         * @brief Normalize observation matrix in-place (single row)
         */
        template<typename DEVICE, typename MATRIX>
        void normalize_matrix(DEVICE& device, MATRIX& observation) const {
            if (!enabled) return;
            
            for (TI i = 0; i < DIM; i++) {
                T val = rlt::get(observation, 0, i);
                T std = std::sqrt(std::max(var[i], MIN_VAR));
                val = (val - mean[i]) / std;
                val = std::max(-CLIP_VALUE, std::min(CLIP_VALUE, val));
                rlt::set(observation, 0, i, val);
            }
        }
        
        /**
         * @brief Normalize batch observation matrix in-place
         */
        template<typename DEVICE, typename MATRIX>
        void normalize_batch_matrix(DEVICE& device, MATRIX& observations, TI batch_size) const {
            if (!enabled) return;
            
            for (TI b = 0; b < batch_size; b++) {
                for (TI i = 0; i < DIM; i++) {
                    T val = rlt::get(observations, b, i);
                    T std = std::sqrt(std::max(var[i], MIN_VAR));
                    val = (val - mean[i]) / std;
                    val = std::max(-CLIP_VALUE, std::min(CLIP_VALUE, val));
                    rlt::set(observations, b, i, val);
                }
            }
        }
        
        /**
         * @brief Get standard deviation for dimension i
         */
        T get_std(TI i) const {
            return std::sqrt(std::max(var[i], MIN_VAR));
        }
    };

    /**
     * @brief Generate C++ code for observation normalizer export
     * 
     * This produces a header file that can be included in STM32 firmware
     * to apply the same normalization during inference.
     */
    template<typename T, typename TI, TI DIM>
    std::string save_normalizer_code(
        const ObservationNormalizer<T, TI, DIM>& normalizer,
        const std::string& name_prefix
    ) {
        std::stringstream ss;
        ss << std::setprecision(10) << std::fixed;  // Fixed-point ensures decimal point always present
        
        ss << "namespace " << name_prefix << "::observation_normalizer {\n";
        ss << "    constexpr int DIM = " << DIM << ";\n";
        ss << "    constexpr float CLIP_VALUE = " << ObservationNormalizer<T, TI, DIM>::CLIP_VALUE << "f;\n";
        ss << "    constexpr float count = " << normalizer.count << "f;\n\n";
        
        ss << "    constexpr float mean[DIM] = {\n        ";
        for (TI i = 0; i < DIM; i++) {
            ss << normalizer.mean[i] << "f";
            if (i < DIM - 1) ss << ", ";
            if ((i + 1) % 6 == 0 && i < DIM - 1) ss << "\n        ";
        }
        ss << "\n    };\n\n";
        
        ss << "    constexpr float std_inv[DIM] = {\n        ";
        for (TI i = 0; i < DIM; i++) {
            T std = std::sqrt(std::max(normalizer.var[i], ObservationNormalizer<T, TI, DIM>::MIN_VAR));
            ss << (1.0f / std) << "f";
            if (i < DIM - 1) ss << ", ";
            if ((i + 1) % 6 == 0 && i < DIM - 1) ss << "\n        ";
        }
        ss << "\n    };\n\n";
        
        // Helper function for firmware
        ss << "    inline void normalize(float* obs) {\n";
        ss << "        for (int i = 0; i < DIM; i++) {\n";
        ss << "            obs[i] = (obs[i] - mean[i]) * std_inv[i];\n";
        ss << "            if (obs[i] > CLIP_VALUE) obs[i] = CLIP_VALUE;\n";
        ss << "            if (obs[i] < -CLIP_VALUE) obs[i] = -CLIP_VALUE;\n";
        ss << "        }\n";
        ss << "    }\n";
        
        ss << "}\n";
        
        return ss.str();
    }

} // namespace rl_tools::rl::algorithms::ppo

#endif // LEARNING_TO_FLY_PPO_OBSERVATION_NORMALIZER_H
