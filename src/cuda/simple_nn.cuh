/**
 * @file simple_nn.cuh
 * @brief Simple neural network implementation for CPU-based PPO
 * 
 * This provides a lightweight, self-contained neural network implementation
 * that doesn't depend on rl_tools templates. Used for hybrid GPU physics +
 * CPU neural network training.
 */

#ifndef SIMPLE_NN_CUH
#define SIMPLE_NN_CUH

#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

namespace learning_to_fly {
namespace cuda {
namespace nn {

/**
 * @brief Fast tanh approximation (same as rl_tools FAST_TANH)
 */
inline float fast_tanh(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/**
 * @brief Dense layer with weights and biases
 */
template<int INPUT_DIM, int OUTPUT_DIM>
struct DenseLayer {
    float weights[OUTPUT_DIM][INPUT_DIM];
    float biases[OUTPUT_DIM];
    
    // Gradients
    float d_weights[OUTPUT_DIM][INPUT_DIM];
    float d_biases[OUTPUT_DIM];
    
    // Adam optimizer state
    float m_weights[OUTPUT_DIM][INPUT_DIM];
    float v_weights[OUTPUT_DIM][INPUT_DIM];
    float m_biases[OUTPUT_DIM];
    float v_biases[OUTPUT_DIM];
    
    // Cached activations for backprop
    float input_cache[INPUT_DIM];
    float output_cache[OUTPUT_DIM];
    
    void init(std::mt19937& rng) {
        // Xavier/Glorot initialization
        float std_dev = std::sqrt(2.0f / (INPUT_DIM + OUTPUT_DIM));
        std::normal_distribution<float> dist(0.0f, std_dev);
        
        for (int i = 0; i < OUTPUT_DIM; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                weights[i][j] = dist(rng);
                d_weights[i][j] = 0;
                m_weights[i][j] = 0;
                v_weights[i][j] = 0;
            }
            biases[i] = 0;
            d_biases[i] = 0;
            m_biases[i] = 0;
            v_biases[i] = 0;
        }
    }
    
    void forward(const float* input, float* output, bool use_tanh = true) {
        // Cache input for backprop
        memcpy(input_cache, input, INPUT_DIM * sizeof(float));
        
        for (int i = 0; i < OUTPUT_DIM; i++) {
            float sum = biases[i];
            for (int j = 0; j < INPUT_DIM; j++) {
                sum += weights[i][j] * input[j];
            }
            output_cache[i] = sum;
            output[i] = use_tanh ? fast_tanh(sum) : sum;
        }
    }
    
    void backward(const float* d_output, float* d_input, bool used_tanh = true) {
        // Zero input gradients
        if (d_input) {
            memset(d_input, 0, INPUT_DIM * sizeof(float));
        }
        
        for (int i = 0; i < OUTPUT_DIM; i++) {
            // Apply tanh derivative if used
            float grad = d_output[i];
            if (used_tanh) {
                float tanh_out = fast_tanh(output_cache[i]);
                grad *= (1.0f - tanh_out * tanh_out);
            }
            
            // Accumulate weight gradients
            for (int j = 0; j < INPUT_DIM; j++) {
                d_weights[i][j] += grad * input_cache[j];
                if (d_input) {
                    d_input[j] += grad * weights[i][j];
                }
            }
            d_biases[i] += grad;
        }
    }
    
    void zero_grad() {
        memset(d_weights, 0, sizeof(d_weights));
        memset(d_biases, 0, sizeof(d_biases));
    }
    
    void adam_update(float lr, float beta1, float beta2, float eps, int t) {
        float bc1 = 1.0f - std::pow(beta1, t);
        float bc2 = 1.0f - std::pow(beta2, t);
        
        for (int i = 0; i < OUTPUT_DIM; i++) {
            for (int j = 0; j < INPUT_DIM; j++) {
                m_weights[i][j] = beta1 * m_weights[i][j] + (1 - beta1) * d_weights[i][j];
                v_weights[i][j] = beta2 * v_weights[i][j] + (1 - beta2) * d_weights[i][j] * d_weights[i][j];
                float m_hat = m_weights[i][j] / bc1;
                float v_hat = v_weights[i][j] / bc2;
                weights[i][j] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
            m_biases[i] = beta1 * m_biases[i] + (1 - beta1) * d_biases[i];
            v_biases[i] = beta2 * v_biases[i] + (1 - beta2) * d_biases[i] * d_biases[i];
            float m_hat = m_biases[i] / bc1;
            float v_hat = v_biases[i] / bc2;
            biases[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
};

/**
 * @brief Actor network: observation -> action mean
 * Architecture: 13 -> 64 -> 64 -> 4 (same as original TD3/PPO)
 */
struct ActorNetwork {
    static constexpr int OBS_DIM = 13;
    static constexpr int HIDDEN_DIM = 64;
    static constexpr int ACTION_DIM = 4;
    
    DenseLayer<OBS_DIM, HIDDEN_DIM> layer1;
    DenseLayer<HIDDEN_DIM, HIDDEN_DIM> layer2;
    DenseLayer<HIDDEN_DIM, ACTION_DIM> layer3;
    
    // Learnable log standard deviation
    float log_std[ACTION_DIM];
    float d_log_std[ACTION_DIM];
    float m_log_std[ACTION_DIM];
    float v_log_std[ACTION_DIM];
    
    // Intermediate activations
    float h1[HIDDEN_DIM];
    float h2[HIDDEN_DIM];
    
    void init(std::mt19937& rng) {
        layer1.init(rng);
        layer2.init(rng);
        layer3.init(rng);
        
        // Initialize log_std to -1.0 (std ≈ 0.37)
        for (int i = 0; i < ACTION_DIM; i++) {
            log_std[i] = -1.0f;
            d_log_std[i] = 0;
            m_log_std[i] = 0;
            v_log_std[i] = 0;
        }
    }
    
    void forward(const float* obs, float* action_mean) {
        layer1.forward(obs, h1, true);
        layer2.forward(h1, h2, true);
        layer3.forward(h2, action_mean, true);  // tanh output
    }
    
    void sample_action(const float* obs, float* action, float* log_prob, std::mt19937& rng) {
        float action_mean[ACTION_DIM];
        forward(obs, action_mean);
        
        std::normal_distribution<float> dist(0.0f, 1.0f);
        *log_prob = 0;
        
        for (int i = 0; i < ACTION_DIM; i++) {
            float std = std::exp(log_std[i]);
            float noise = dist(rng);
            
            // Sample from Gaussian, then squash
            float u = action_mean[i] + std * noise;  // Pre-squash (but action_mean already squashed)
            
            // Since action_mean is already tanh'd, we sample around it
            // Proper implementation: sample u ~ N(mu, std), then a = tanh(u)
            // For simplicity here, add noise to squashed output and re-clamp
            float a = std::tanh(action_mean[i] + std * noise * 0.5f);  // Scale noise
            action[i] = a;
            
            // Log probability (simplified - assumes small noise)
            float log_p = -0.5f * noise * noise - log_std[i] - 0.5f * std::log(2.0f * M_PI);
            // Tanh squashing correction
            log_p -= std::log(1.0f - a * a + 1e-6f);
            *log_prob += log_p;
        }
    }
    
    float compute_log_prob(const float* obs, const float* action) {
        float action_mean[ACTION_DIM];
        forward(obs, action_mean);
        
        float log_prob = 0;
        for (int i = 0; i < ACTION_DIM; i++) {
            float std = std::exp(log_std[i]);
            // Inverse tanh to get pre-squash value
            float a_clamped = std::max(-0.999f, std::min(0.999f, action[i]));
            float u = 0.5f * std::log((1.0f + a_clamped) / (1.0f - a_clamped));  // atanh
            
            float diff = u - std::atanh(std::max(-0.999f, std::min(0.999f, action_mean[i])));
            float log_p = -0.5f * (diff * diff) / (std * std) - log_std[i] - 0.5f * std::log(2.0f * M_PI);
            log_p -= std::log(1.0f - a_clamped * a_clamped + 1e-6f);
            log_prob += log_p;
        }
        return log_prob;
    }
    
    void zero_grad() {
        layer1.zero_grad();
        layer2.zero_grad();
        layer3.zero_grad();
        memset(d_log_std, 0, sizeof(d_log_std));
    }
    
    void adam_update(float lr, int t) {
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        layer1.adam_update(lr, beta1, beta2, eps, t);
        layer2.adam_update(lr, beta1, beta2, eps, t);
        layer3.adam_update(lr, beta1, beta2, eps, t);
        
        // Update log_std
        float bc1 = 1.0f - std::pow(beta1, t);
        float bc2 = 1.0f - std::pow(beta2, t);
        for (int i = 0; i < ACTION_DIM; i++) {
            m_log_std[i] = beta1 * m_log_std[i] + (1 - beta1) * d_log_std[i];
            v_log_std[i] = beta2 * v_log_std[i] + (1 - beta2) * d_log_std[i] * d_log_std[i];
            float m_hat = m_log_std[i] / bc1;
            float v_hat = v_log_std[i] / bc2;
            log_std[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            // Clamp log_std
            log_std[i] = std::max(-3.0f, std::min(0.0f, log_std[i]));
        }
    }
};

/**
 * @brief Critic network: observation -> value
 * Architecture: 13 -> 64 -> 64 -> 1
 */
struct CriticNetwork {
    static constexpr int OBS_DIM = 13;
    static constexpr int HIDDEN_DIM = 64;
    
    DenseLayer<OBS_DIM, HIDDEN_DIM> layer1;
    DenseLayer<HIDDEN_DIM, HIDDEN_DIM> layer2;
    DenseLayer<HIDDEN_DIM, 1> layer3;
    
    float h1[HIDDEN_DIM];
    float h2[HIDDEN_DIM];
    
    void init(std::mt19937& rng) {
        layer1.init(rng);
        layer2.init(rng);
        layer3.init(rng);
    }
    
    float forward(const float* obs) {
        layer1.forward(obs, h1, true);
        layer2.forward(h1, h2, true);
        float value;
        layer3.forward(h2, &value, false);  // Linear output
        return value;
    }
    
    void backward(float d_value) {
        float d_h2[HIDDEN_DIM];
        float d_h1[HIDDEN_DIM];
        layer3.backward(&d_value, d_h2, false);
        layer2.backward(d_h2, d_h1, true);
        layer1.backward(d_h1, nullptr, true);
    }
    
    void zero_grad() {
        layer1.zero_grad();
        layer2.zero_grad();
        layer3.zero_grad();
    }
    
    void adam_update(float lr, int t) {
        float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
        layer1.adam_update(lr, beta1, beta2, eps, t);
        layer2.adam_update(lr, beta1, beta2, eps, t);
        layer3.adam_update(lr, beta1, beta2, eps, t);
    }
};

} // namespace nn
} // namespace cuda
} // namespace learning_to_fly

#endif // SIMPLE_NN_CUH
