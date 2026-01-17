/**
 * @file training_ppo_hybrid.cu
 * @brief Hybrid GPU/CPU PPO Training
 * 
 * GPU handles:
 * - Parallel physics simulation (all environments)
 * - Neural network forward pass for action sampling
 * - Rollout data collection
 * 
 * CPU handles:
 * - PPO update with proper gradient computation
 * - Adam optimizer
 * 
 * This approach gives us high throughput rollout collection (millions of steps/s)
 * while maintaining correct learning.
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>
#include <random>
#include <cstring>
#include <sys/stat.h>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>

// Configuration
namespace config {
    constexpr int N_ENVS = 512;
    constexpr int ROLLOUT_STEPS = 512;
    constexpr int OBS_DIM = 13;
    constexpr int ACTION_DIM = 4;
    constexpr int HIDDEN_DIM = 64;
    
    constexpr int PPO_EPOCHS = 10;
    constexpr int MINIBATCH_SIZE = 512;
    constexpr float GAMMA = 0.99f;
    constexpr float GAE_LAMBDA = 0.95f;
    constexpr float CLIP_EPS = 0.2f;
    constexpr float ACTOR_LR = 3e-4f;
    constexpr float CRITIC_LR = 1e-3f;
    constexpr float ENT_COEF = 0.01f;
    
    constexpr int N_UPDATES = 500;
    constexpr int LOG_INTERVAL = 10;
    constexpr int CHECKPOINT_INTERVAL = 50;
    
    // Physics
    constexpr float DT = 0.01f;
    constexpr int SUBSTEPS = 4;
    constexpr float MASS = 0.027f;
    constexpr float ARM_LENGTH = 0.046f;
    constexpr float MAX_THRUST = 0.06f;
    constexpr float G = 9.81f;
}

using namespace config;

// ============ Device Helpers ============

__device__ __forceinline__ float fast_tanh_d(float x) {
    if (x < -3.0f) return -1.0f;
    if (x > 3.0f) return 1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

// ============ GPU Quadrotor State ============

struct __align__(16) GPUQuadState {
    float pos[3];
    float quat[4];
    float vel[3];
    float omega[3];
    float rpm[4];
    float target_pos[3];
    bool terminated;
    int step_count;
};

// ============ Network Weights (GPU) ============

struct GPUWeights {
    float* actor_w1;
    float* actor_b1;
    float* actor_w2;
    float* actor_b2;
    float* actor_w3;
    float* actor_b3;
    float* actor_log_std;
    
    float* critic_w1;
    float* critic_b1;
    float* critic_w2;
    float* critic_b2;
    float* critic_w3;
    float* critic_b3;
};

// ============ Physics Kernels ============

__device__ void quat_mult(const float* q1, const float* q2, float* out) {
    out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3];
    out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2];
    out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1];
    out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0];
}

__device__ void rotate_vec(const float* q, const float* v, float* out) {
    float qv[4] = {0, v[0], v[1], v[2]};
    float q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
    float temp[4], result[4];
    quat_mult(q, qv, temp);
    quat_mult(temp, q_conj, result);
    out[0] = result[1];
    out[1] = result[2];
    out[2] = result[3];
}

__device__ void reset_env(GPUQuadState* state, curandState* rng) {
    state->pos[0] = curand_uniform(rng) * 2.0f - 1.0f;
    state->pos[1] = curand_uniform(rng) * 2.0f - 1.0f;
    state->pos[2] = curand_uniform(rng) * 2.0f - 1.0f;
    state->quat[0] = 1.0f;
    state->quat[1] = state->quat[2] = state->quat[3] = 0;
    state->vel[0] = state->vel[1] = state->vel[2] = 0;
    state->omega[0] = state->omega[1] = state->omega[2] = 0;
    float hover = sqrtf(MASS * G / (4.0f * MAX_THRUST)) * 2.0f;
    state->rpm[0] = state->rpm[1] = state->rpm[2] = state->rpm[3] = hover;
    state->target_pos[0] = state->target_pos[1] = state->target_pos[2] = 0;
    state->terminated = false;
    state->step_count = 0;
}

__device__ void get_obs(const GPUQuadState* s, float* obs) {
    obs[0] = s->pos[0] - s->target_pos[0];
    obs[1] = s->pos[1] - s->target_pos[1];
    obs[2] = s->pos[2] - s->target_pos[2];
    obs[3] = s->quat[0]; obs[4] = s->quat[1]; obs[5] = s->quat[2]; obs[6] = s->quat[3];
    obs[7] = s->vel[0]; obs[8] = s->vel[1]; obs[9] = s->vel[2];
    obs[10] = s->omega[0]; obs[11] = s->omega[1]; obs[12] = s->omega[2];
}

__device__ float step_physics(GPUQuadState* s, const float* action, curandState* rng) {
    float dt = DT / SUBSTEPS;
    float Jxx = 1.4e-5f, Jyy = 1.4e-5f, Jzz = 2.17e-5f;
    
    for (int sub = 0; sub < SUBSTEPS; sub++) {
        float rpm[4];
        for (int i = 0; i < 4; i++) {
            rpm[i] = (action[i] + 1.0f) * 0.5f;
            s->rpm[i] += 0.1f * (rpm[i] - s->rpm[i]);
        }
        
        float thrust[4], total = 0;
        for (int i = 0; i < 4; i++) {
            thrust[i] = MAX_THRUST * s->rpm[i] * s->rpm[i];
            total += thrust[i];
        }
        
        float tb[3] = {0, 0, total / MASS}, tw[3];
        rotate_vec(s->quat, tb, tw);
        
        s->vel[0] += tw[0] * dt;
        s->vel[1] += tw[1] * dt;
        s->vel[2] += (tw[2] - G) * dt;
        s->pos[0] += s->vel[0] * dt;
        s->pos[1] += s->vel[1] * dt;
        s->pos[2] += s->vel[2] * dt;
        
        s->omega[0] += ARM_LENGTH * (thrust[0] - thrust[1] - thrust[2] + thrust[3]) / Jxx * dt;
        s->omega[1] += ARM_LENGTH * (thrust[0] + thrust[1] - thrust[2] - thrust[3]) / Jyy * dt;
        s->omega[2] += 0.01f * (thrust[0] - thrust[1] + thrust[2] - thrust[3]) / Jzz * dt;
        
        float oq[4] = {0, s->omega[0]*dt*0.5f, s->omega[1]*dt*0.5f, s->omega[2]*dt*0.5f};
        float dq[4];
        quat_mult(s->quat, oq, dq);
        s->quat[0] += dq[0]; s->quat[1] += dq[1]; s->quat[2] += dq[2]; s->quat[3] += dq[3];
        float n = sqrtf(s->quat[0]*s->quat[0]+s->quat[1]*s->quat[1]+s->quat[2]*s->quat[2]+s->quat[3]*s->quat[3]);
        s->quat[0]/=n; s->quat[1]/=n; s->quat[2]/=n; s->quat[3]/=n;
    }
    
    s->step_count++;
    
    float dx = s->pos[0], dy = s->pos[1], dz = s->pos[2];
    float dist = dx*dx + dy*dy + dz*dz;
    float vel = s->vel[0]*s->vel[0]+s->vel[1]*s->vel[1]+s->vel[2]*s->vel[2];
    float omega = s->omega[0]*s->omega[0]+s->omega[1]*s->omega[1]+s->omega[2]*s->omega[2];
    
    float reward = -dist - 0.01f*vel - 0.001f*omega;
    
    float maxp = fmaxf(fabsf(s->pos[0]), fmaxf(fabsf(s->pos[1]), fabsf(s->pos[2])));
    if (maxp > 5.0f || s->step_count >= 500) {
        s->terminated = true;
        if (maxp > 5.0f) reward -= 10.0f;
    }
    
    return reward;
}

// ============ Rollout Step Kernel ============

__global__ void rollout_step_kernel(
    GPUQuadState* states, const GPUWeights w,
    float* obs_buf, float* act_buf, float* logp_buf,
    float* val_buf, float* rew_buf, float* done_buf,
    curandState* rngs, int step
) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= N_ENVS) return;
    
    GPUQuadState* s = &states[env];
    curandState rng = rngs[env];
    
    float obs[OBS_DIM];
    get_obs(s, obs);
    
    int off = step * N_ENVS + env;
    for (int i = 0; i < OBS_DIM; i++) obs_buf[off * OBS_DIM + i] = obs[i];
    
    // Actor forward
    float h1[HIDDEN_DIM], h2[HIDDEN_DIM], mean[ACTION_DIM];
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float sum = w.actor_b1[j];
        for (int i = 0; i < OBS_DIM; i++) sum += w.actor_w1[j*OBS_DIM+i] * obs[i];
        h1[j] = fast_tanh_d(sum);
    }
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float sum = w.actor_b2[j];
        for (int i = 0; i < HIDDEN_DIM; i++) sum += w.actor_w2[j*HIDDEN_DIM+i] * h1[i];
        h2[j] = fast_tanh_d(sum);
    }
    for (int j = 0; j < ACTION_DIM; j++) {
        float sum = w.actor_b3[j];
        for (int i = 0; i < HIDDEN_DIM; i++) sum += w.actor_w3[j*HIDDEN_DIM+i] * h2[i];
        mean[j] = fast_tanh_d(sum);
    }
    
    // Sample action
    float act[ACTION_DIM], logp = 0;
    for (int a = 0; a < ACTION_DIM; a++) {
        float std = expf(w.actor_log_std[a]);
        float noise = curand_normal(&rng);
        float mu_at = atanhf(fmaxf(-0.999f, fminf(0.999f, mean[a])));
        float u = mu_at + std * noise;
        act[a] = tanhf(u);
        float lp = -0.5f*noise*noise - w.actor_log_std[a] - 0.5f*logf(2.0f*M_PI);
        lp -= logf(1.0f - act[a]*act[a] + 1e-6f);
        logp += lp;
    }
    for (int i = 0; i < ACTION_DIM; i++) act_buf[off*ACTION_DIM+i] = act[i];
    logp_buf[off] = logp;
    
    // Critic forward
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float sum = w.critic_b1[j];
        for (int i = 0; i < OBS_DIM; i++) sum += w.critic_w1[j*OBS_DIM+i] * obs[i];
        h1[j] = fast_tanh_d(sum);
    }
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float sum = w.critic_b2[j];
        for (int i = 0; i < HIDDEN_DIM; i++) sum += w.critic_w2[j*HIDDEN_DIM+i] * h1[i];
        h2[j] = fast_tanh_d(sum);
    }
    float val = w.critic_b3[0];
    for (int i = 0; i < HIDDEN_DIM; i++) val += w.critic_w3[i] * h2[i];
    val_buf[off] = val;
    
    // Step physics
    float rew = step_physics(s, act, &rng);
    rew_buf[off] = rew;
    done_buf[off] = s->terminated ? 1.0f : 0.0f;
    
    if (s->terminated) reset_env(s, &rng);
    rngs[env] = rng;
}

__global__ void init_envs_kernel(GPUQuadState* states, curandState* rngs, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &rngs[idx]);
    reset_env(&states[idx], &rngs[idx]);
}

__global__ void compute_last_values_kernel(const GPUQuadState* states, const GPUWeights w, float* vals) {
    int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= N_ENVS) return;
    
    float obs[OBS_DIM], h1[HIDDEN_DIM], h2[HIDDEN_DIM];
    get_obs(&states[env], obs);
    
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float sum = w.critic_b1[j];
        for (int i = 0; i < OBS_DIM; i++) sum += w.critic_w1[j*OBS_DIM+i] * obs[i];
        h1[j] = fast_tanh_d(sum);
    }
    for (int j = 0; j < HIDDEN_DIM; j++) {
        float sum = w.critic_b2[j];
        for (int i = 0; i < HIDDEN_DIM; i++) sum += w.critic_w2[j*HIDDEN_DIM+i] * h1[i];
        h2[j] = fast_tanh_d(sum);
    }
    float val = w.critic_b3[0];
    for (int i = 0; i < HIDDEN_DIM; i++) val += w.critic_w3[i] * h2[i];
    vals[env] = val;
}

// ============ CPU Neural Network ============

template<int IN, int OUT>
struct DenseLayer {
    float w[OUT][IN];
    float b[OUT];
    float dw[OUT][IN];
    float db[OUT];
    float mw[OUT][IN], vw[OUT][IN];
    float mb[OUT], vb[OUT];
    
    float h[OUT], pre[OUT];  // Activations
    
    void init(std::mt19937& rng) {
        float std = sqrtf(2.0f / (IN + OUT));
        std::normal_distribution<float> d(0, std);
        for (int i = 0; i < OUT; i++) {
            for (int j = 0; j < IN; j++) {
                w[i][j] = d(rng);
                mw[i][j] = vw[i][j] = 0;
            }
            b[i] = 0;
            mb[i] = vb[i] = 0;
        }
    }
    
    void zero_grad() {
        memset(dw, 0, sizeof(dw));
        memset(db, 0, sizeof(db));
    }
    
    void forward_tanh(const float* in) {
        for (int i = 0; i < OUT; i++) {
            float sum = b[i];
            for (int j = 0; j < IN; j++) sum += w[i][j] * in[j];
            pre[i] = sum;
            if (sum < -3.0f) h[i] = -1.0f;
            else if (sum > 3.0f) h[i] = 1.0f;
            else h[i] = sum * (27.0f + sum*sum) / (27.0f + 9.0f * sum*sum);
        }
    }
    
    void forward_linear(const float* in) {
        for (int i = 0; i < OUT; i++) {
            float sum = b[i];
            for (int j = 0; j < IN; j++) sum += w[i][j] * in[j];
            h[i] = sum;
        }
    }
    
    void backward_tanh(const float* in, const float* dout, float* din) {
        for (int i = 0; i < OUT; i++) {
            float dtanh = 1.0f - h[i] * h[i];
            float d = dout[i] * dtanh;
            db[i] += d;
            for (int j = 0; j < IN; j++) {
                dw[i][j] += d * in[j];
                if (din) din[j] += d * w[i][j];
            }
        }
    }
    
    void backward_linear(const float* in, const float* dout, float* din) {
        for (int i = 0; i < OUT; i++) {
            db[i] += dout[i];
            for (int j = 0; j < IN; j++) {
                dw[i][j] += dout[i] * in[j];
                if (din) din[j] += dout[i] * w[i][j];
            }
        }
    }
    
    void adam_update(float lr, float bc1, float bc2) {
        for (int i = 0; i < OUT; i++) {
            for (int j = 0; j < IN; j++) {
                mw[i][j] = 0.9f * mw[i][j] + 0.1f * dw[i][j];
                vw[i][j] = 0.999f * vw[i][j] + 0.001f * dw[i][j] * dw[i][j];
                w[i][j] -= lr * (mw[i][j] / bc1) / (sqrtf(vw[i][j] / bc2) + 1e-8f);
            }
            mb[i] = 0.9f * mb[i] + 0.1f * db[i];
            vb[i] = 0.999f * vb[i] + 0.001f * db[i] * db[i];
            b[i] -= lr * (mb[i] / bc1) / (sqrtf(vb[i] / bc2) + 1e-8f);
        }
    }
};

struct CPUActor {
    DenseLayer<OBS_DIM, HIDDEN_DIM> l1;
    DenseLayer<HIDDEN_DIM, HIDDEN_DIM> l2;
    DenseLayer<HIDDEN_DIM, ACTION_DIM> l3;
    float log_std[ACTION_DIM];
    float dlog_std[ACTION_DIM];
    float m_log_std[ACTION_DIM], v_log_std[ACTION_DIM];
    
    void init(std::mt19937& rng) {
        l1.init(rng); l2.init(rng); l3.init(rng);
        for (int i = 0; i < ACTION_DIM; i++) {
            log_std[i] = -1.0f;
            m_log_std[i] = v_log_std[i] = 0;
        }
    }
    
    void zero_grad() {
        l1.zero_grad(); l2.zero_grad(); l3.zero_grad();
        memset(dlog_std, 0, sizeof(dlog_std));
    }
    
    void forward(const float* obs) {
        l1.forward_tanh(obs);
        l2.forward_tanh(l1.h);
        l3.forward_tanh(l2.h);
    }
    
    float compute_log_prob(const float* action) {
        float lp = 0;
        for (int a = 0; a < ACTION_DIM; a++) {
            float mean = l3.h[a];
            float std = expf(log_std[a]);
            float act = fmaxf(-0.999f, fminf(0.999f, action[a]));
            float mean_c = fmaxf(-0.999f, fminf(0.999f, mean));
            float u = atanhf(act);
            float mu = atanhf(mean_c);
            float diff = u - mu;
            lp += -0.5f * diff * diff / (std * std) - log_std[a] - 0.5f * logf(2.0f * M_PI);
            lp -= logf(1.0f - act * act + 1e-6f);
        }
        return lp;
    }
    
    void backward_policy(const float* obs, const float* action, float grad_scale) {
        // Simplified backward pass
        float dout[ACTION_DIM];
        for (int a = 0; a < ACTION_DIM; a++) {
            float mean = l3.h[a];
            float std = expf(log_std[a]);
            float act = fmaxf(-0.999f, fminf(0.999f, action[a]));
            float mean_c = fmaxf(-0.999f, fminf(0.999f, mean));
            float u = atanhf(act);
            float mu = atanhf(mean_c);
            float diff = u - mu;
            
            // Gradient of log prob w.r.t. mean
            float dmean = diff / (std * std) / (1.0f - mean_c * mean_c + 1e-6f);
            dout[a] = grad_scale * dmean;
            
            // Gradient w.r.t. log_std
            dlog_std[a] += grad_scale * (diff * diff / (std * std) - 1.0f);
        }
        
        float dh2[HIDDEN_DIM] = {0};
        l3.backward_tanh(l2.h, dout, dh2);
        
        float dh1[HIDDEN_DIM] = {0};
        l2.backward_tanh(l1.h, dh2, dh1);
        
        l1.backward_tanh(obs, dh1, nullptr);
    }
    
    void adam_update(float lr, float bc1, float bc2) {
        l1.adam_update(lr, bc1, bc2);
        l2.adam_update(lr, bc1, bc2);
        l3.adam_update(lr, bc1, bc2);
        
        for (int i = 0; i < ACTION_DIM; i++) {
            m_log_std[i] = 0.9f * m_log_std[i] + 0.1f * dlog_std[i];
            v_log_std[i] = 0.999f * v_log_std[i] + 0.001f * dlog_std[i] * dlog_std[i];
            log_std[i] -= lr * (m_log_std[i] / bc1) / (sqrtf(v_log_std[i] / bc2) + 1e-8f);
        }
    }
    
    void copy_to_gpu(GPUWeights& gw) {
        cudaMemcpy(gw.actor_w1, l1.w, sizeof(l1.w), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.actor_b1, l1.b, sizeof(l1.b), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.actor_w2, l2.w, sizeof(l2.w), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.actor_b2, l2.b, sizeof(l2.b), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.actor_w3, l3.w, sizeof(l3.w), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.actor_b3, l3.b, sizeof(l3.b), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.actor_log_std, log_std, sizeof(log_std), cudaMemcpyHostToDevice);
    }
};

struct CPUCritic {
    DenseLayer<OBS_DIM, HIDDEN_DIM> l1;
    DenseLayer<HIDDEN_DIM, HIDDEN_DIM> l2;
    DenseLayer<HIDDEN_DIM, 1> l3;
    
    void init(std::mt19937& rng) { l1.init(rng); l2.init(rng); l3.init(rng); }
    void zero_grad() { l1.zero_grad(); l2.zero_grad(); l3.zero_grad(); }
    
    float forward(const float* obs) {
        l1.forward_tanh(obs);
        l2.forward_tanh(l1.h);
        l3.forward_linear(l2.h);
        return l3.h[0];
    }
    
    void backward(const float* obs, float dvalue) {
        float dout[1] = {dvalue};
        float dh2[HIDDEN_DIM] = {0};
        l3.backward_linear(l2.h, dout, dh2);
        float dh1[HIDDEN_DIM] = {0};
        l2.backward_tanh(l1.h, dh2, dh1);
        l1.backward_tanh(obs, dh1, nullptr);
    }
    
    void adam_update(float lr, float bc1, float bc2) {
        l1.adam_update(lr, bc1, bc2);
        l2.adam_update(lr, bc1, bc2);
        l3.adam_update(lr, bc1, bc2);
    }
    
    void copy_to_gpu(GPUWeights& gw) {
        cudaMemcpy(gw.critic_w1, l1.w, sizeof(l1.w), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.critic_b1, l1.b, sizeof(l1.b), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.critic_w2, l2.w, sizeof(l2.w), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.critic_b2, l2.b, sizeof(l2.b), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.critic_w3, l3.w, sizeof(l3.w), cudaMemcpyHostToDevice);
        cudaMemcpy(gw.critic_b3, l3.b, sizeof(l3.b), cudaMemcpyHostToDevice);
    }
};

// ============ Training Class ============

class HybridTrainer {
public:
    // GPU
    GPUQuadState* d_states;
    GPUWeights d_weights;
    curandState* d_rngs;
    float *d_obs, *d_act, *d_logp, *d_val, *d_rew, *d_done, *d_last_val;
    
    // Host buffers
    float *h_obs, *h_act, *h_logp, *h_val, *h_rew, *h_done, *h_last_val;
    float *h_adv, *h_ret;
    
    // CPU networks
    CPUActor actor;
    CPUCritic critic;
    
    int adam_step;
    std::string checkpoint_dir;
    std::mt19937 rng;
    
    void init(unsigned int seed) {
        rng.seed(seed);
        adam_step = 0;
        
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::tm tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S");
        checkpoint_dir = "checkpoints/hybrid_ppo/" + oss.str();
        mkdir("checkpoints", 0755);
        mkdir("checkpoints/hybrid_ppo", 0755);
        mkdir(checkpoint_dir.c_str(), 0755);
        
        int buf_size = ROLLOUT_STEPS * N_ENVS;
        
        // GPU allocations
        cudaMalloc(&d_states, N_ENVS * sizeof(GPUQuadState));
        cudaMalloc(&d_rngs, N_ENVS * sizeof(curandState));
        cudaMalloc(&d_obs, buf_size * OBS_DIM * sizeof(float));
        cudaMalloc(&d_act, buf_size * ACTION_DIM * sizeof(float));
        cudaMalloc(&d_logp, buf_size * sizeof(float));
        cudaMalloc(&d_val, buf_size * sizeof(float));
        cudaMalloc(&d_rew, buf_size * sizeof(float));
        cudaMalloc(&d_done, buf_size * sizeof(float));
        cudaMalloc(&d_last_val, N_ENVS * sizeof(float));
        
        cudaMalloc(&d_weights.actor_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&d_weights.actor_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.actor_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.actor_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.actor_w3, ACTION_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.actor_b3, ACTION_DIM * sizeof(float));
        cudaMalloc(&d_weights.actor_log_std, ACTION_DIM * sizeof(float));
        
        cudaMalloc(&d_weights.critic_w1, HIDDEN_DIM * OBS_DIM * sizeof(float));
        cudaMalloc(&d_weights.critic_b1, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.critic_w2, HIDDEN_DIM * HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.critic_b2, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.critic_w3, HIDDEN_DIM * sizeof(float));
        cudaMalloc(&d_weights.critic_b3, sizeof(float));
        
        // Host allocations
        h_obs = new float[buf_size * OBS_DIM];
        h_act = new float[buf_size * ACTION_DIM];
        h_logp = new float[buf_size];
        h_val = new float[buf_size];
        h_rew = new float[buf_size];
        h_done = new float[buf_size];
        h_last_val = new float[N_ENVS];
        h_adv = new float[buf_size];
        h_ret = new float[buf_size];
        
        // Initialize networks
        actor.init(rng);
        critic.init(rng);
        
        // Copy to GPU
        actor.copy_to_gpu(d_weights);
        critic.copy_to_gpu(d_weights);
        
        // Initialize envs
        int blocks = (N_ENVS + 255) / 256;
        init_envs_kernel<<<blocks, 256>>>(d_states, d_rngs, seed, N_ENVS);
        cudaDeviceSynchronize();
        
        std::cout << "Hybrid PPO Trainer initialized\n";
        std::cout << "  N_ENVS: " << N_ENVS << "\n";
        std::cout << "  ROLLOUT_STEPS: " << ROLLOUT_STEPS << "\n";
        std::cout << "  Checkpoint: " << checkpoint_dir << "\n";
    }
    
    void collect_rollout() {
        int blocks = (N_ENVS + 255) / 256;
        
        for (int step = 0; step < ROLLOUT_STEPS; step++) {
            rollout_step_kernel<<<blocks, 256>>>(
                d_states, d_weights, d_obs, d_act, d_logp,
                d_val, d_rew, d_done, d_rngs, step);
        }
        
        compute_last_values_kernel<<<blocks, 256>>>(d_states, d_weights, d_last_val);
        cudaDeviceSynchronize();
        
        // Copy to host
        int buf_size = ROLLOUT_STEPS * N_ENVS;
        cudaMemcpy(h_obs, d_obs, buf_size * OBS_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_act, d_act, buf_size * ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_logp, d_logp, buf_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_val, d_val, buf_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rew, d_rew, buf_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_done, d_done, buf_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_last_val, d_last_val, N_ENVS * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    void compute_gae() {
        for (int env = 0; env < N_ENVS; env++) {
            float gae = 0;
            float next_val = h_last_val[env];
            float next_done = 0;
            
            for (int step = ROLLOUT_STEPS - 1; step >= 0; step--) {
                int idx = step * N_ENVS + env;
                float delta = h_rew[idx] + GAMMA * next_val * (1.0f - next_done) - h_val[idx];
                gae = delta + GAMMA * GAE_LAMBDA * (1.0f - next_done) * gae;
                h_adv[idx] = gae;
                h_ret[idx] = gae + h_val[idx];
                next_val = h_val[idx];
                next_done = h_done[idx];
            }
        }
        
        // Normalize advantages
        int n = ROLLOUT_STEPS * N_ENVS;
        float mean = 0, var = 0;
        for (int i = 0; i < n; i++) mean += h_adv[i];
        mean /= n;
        for (int i = 0; i < n; i++) var += (h_adv[i] - mean) * (h_adv[i] - mean);
        var /= n;
        float std = sqrtf(var);
        for (int i = 0; i < n; i++) h_adv[i] = (h_adv[i] - mean) / (std + 1e-8f);
    }
    
    void ppo_update() {
        int n_samples = ROLLOUT_STEPS * N_ENVS;
        int n_batches = n_samples / MINIBATCH_SIZE;
        
        std::vector<int> indices(n_samples);
        for (int i = 0; i < n_samples; i++) indices[i] = i;
        
        for (int epoch = 0; epoch < PPO_EPOCHS; epoch++) {
            std::shuffle(indices.begin(), indices.end(), rng);
            
            for (int batch = 0; batch < n_batches; batch++) {
                adam_step++;
                float bc1 = 1.0f - powf(0.9f, adam_step);
                float bc2 = 1.0f - powf(0.999f, adam_step);
                
                actor.zero_grad();
                critic.zero_grad();
                
                for (int i = 0; i < MINIBATCH_SIZE; i++) {
                    int idx = indices[batch * MINIBATCH_SIZE + i];
                    const float* obs = h_obs + idx * OBS_DIM;
                    const float* act = h_act + idx * ACTION_DIM;
                    float old_logp = h_logp[idx];
                    float adv = h_adv[idx];
                    float ret = h_ret[idx];
                    
                    // Actor update
                    actor.forward(obs);
                    float new_logp = actor.compute_log_prob(act);
                    float ratio = expf(new_logp - old_logp);
                    float clipped = fmaxf(1.0f - CLIP_EPS, fminf(1.0f + CLIP_EPS, ratio));
                    float surr1 = ratio * adv;
                    float surr2 = clipped * adv;
                    
                    float grad_scale = 0;
                    if (surr1 < surr2) {
                        grad_scale = -adv / MINIBATCH_SIZE;  // Gradient ascent
                    } else if (ratio >= 1.0f - CLIP_EPS && ratio <= 1.0f + CLIP_EPS) {
                        grad_scale = -adv / MINIBATCH_SIZE;
                    }
                    
                    if (grad_scale != 0) {
                        actor.backward_policy(obs, act, grad_scale);
                    }
                    
                    // Critic update
                    float val = critic.forward(obs);
                    float val_loss_grad = 2.0f * (val - ret) / MINIBATCH_SIZE;
                    critic.backward(obs, val_loss_grad);
                }
                
                actor.adam_update(ACTOR_LR, bc1, bc2);
                critic.adam_update(CRITIC_LR, bc1, bc2);
            }
        }
        
        // Sync weights to GPU
        actor.copy_to_gpu(d_weights);
        critic.copy_to_gpu(d_weights);
    }
    
    float compute_mean_reward() {
        int n = ROLLOUT_STEPS * N_ENVS;
        float sum = 0;
        for (int i = 0; i < n; i++) sum += h_rew[i];
        return sum / n;
    }
    
    void save_checkpoint(int update) {
        std::string path = checkpoint_dir + "/actor_" + std::to_string(update) + ".h";
        std::ofstream f(path);
        
        f << "// Hybrid PPO Actor - Update " << update << "\n";
        f << "#pragma once\n\nnamespace checkpoint {\n\n";
        
        f << "constexpr float L1_W[" << HIDDEN_DIM << "][" << OBS_DIM << "] = {\n";
        for (int i = 0; i < HIDDEN_DIM; i++) {
            f << "  {";
            for (int j = 0; j < OBS_DIM; j++) {
                f << actor.l1.w[i][j];
                if (j < OBS_DIM - 1) f << ", ";
            }
            f << "}" << (i < HIDDEN_DIM - 1 ? ",\n" : "\n");
        }
        f << "};\n\nconst float L1_B[" << HIDDEN_DIM << "] = {";
        for (int i = 0; i < HIDDEN_DIM; i++) f << actor.l1.b[i] << (i < HIDDEN_DIM - 1 ? ", " : "");
        f << "};\n\n";
        
        f << "constexpr float L2_W[" << HIDDEN_DIM << "][" << HIDDEN_DIM << "] = {\n";
        for (int i = 0; i < HIDDEN_DIM; i++) {
            f << "  {";
            for (int j = 0; j < HIDDEN_DIM; j++) {
                f << actor.l2.w[i][j];
                if (j < HIDDEN_DIM - 1) f << ", ";
            }
            f << "}" << (i < HIDDEN_DIM - 1 ? ",\n" : "\n");
        }
        f << "};\n\nconst float L2_B[" << HIDDEN_DIM << "] = {";
        for (int i = 0; i < HIDDEN_DIM; i++) f << actor.l2.b[i] << (i < HIDDEN_DIM - 1 ? ", " : "");
        f << "};\n\n";
        
        f << "constexpr float L3_W[" << ACTION_DIM << "][" << HIDDEN_DIM << "] = {\n";
        for (int i = 0; i < ACTION_DIM; i++) {
            f << "  {";
            for (int j = 0; j < HIDDEN_DIM; j++) {
                f << actor.l3.w[i][j];
                if (j < HIDDEN_DIM - 1) f << ", ";
            }
            f << "}" << (i < ACTION_DIM - 1 ? ",\n" : "\n");
        }
        f << "};\n\nconst float L3_B[" << ACTION_DIM << "] = {";
        for (int i = 0; i < ACTION_DIM; i++) f << actor.l3.b[i] << (i < ACTION_DIM - 1 ? ", " : "");
        f << "};\n\nconst float LOG_STD[" << ACTION_DIM << "] = {";
        for (int i = 0; i < ACTION_DIM; i++) f << actor.log_std[i] << (i < ACTION_DIM - 1 ? ", " : "");
        f << "};\n\n} // namespace checkpoint\n";
        
        f.close();
        std::cout << "  Saved: " << path << std::endl;
    }
    
    void train() {
        std::cout << "\n=== Starting Hybrid PPO Training ===\n\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        long long total_steps = 0;
        
        for (int update = 1; update <= N_UPDATES; update++) {
            collect_rollout();
            total_steps += ROLLOUT_STEPS * N_ENVS;
            
            compute_gae();
            ppo_update();
            
            if (update % LOG_INTERVAL == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - start).count();
                float mean_rew = compute_mean_reward();
                
                std::cout << "Update " << update << "/" << N_UPDATES
                          << " | Reward: " << std::fixed << std::setprecision(4) << mean_rew
                          << " | Steps/s: " << std::setprecision(0) << total_steps / elapsed
                          << std::endl;
            }
            
            if (update % CHECKPOINT_INTERVAL == 0) {
                save_checkpoint(update);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end - start).count();
        
        std::cout << "\n=== Training Complete ===\n";
        std::cout << "Time: " << total_time << "s, Steps: " << total_steps;
        std::cout << ", Steps/s: " << total_steps / total_time << "\n";
        
        save_checkpoint(N_UPDATES);
    }
    
    void cleanup() {
        cudaFree(d_states); cudaFree(d_rngs);
        cudaFree(d_obs); cudaFree(d_act); cudaFree(d_logp);
        cudaFree(d_val); cudaFree(d_rew); cudaFree(d_done);
        cudaFree(d_last_val);
        cudaFree(d_weights.actor_w1); cudaFree(d_weights.actor_b1);
        cudaFree(d_weights.actor_w2); cudaFree(d_weights.actor_b2);
        cudaFree(d_weights.actor_w3); cudaFree(d_weights.actor_b3);
        cudaFree(d_weights.actor_log_std);
        cudaFree(d_weights.critic_w1); cudaFree(d_weights.critic_b1);
        cudaFree(d_weights.critic_w2); cudaFree(d_weights.critic_b2);
        cudaFree(d_weights.critic_w3); cudaFree(d_weights.critic_b3);
        
        delete[] h_obs; delete[] h_act; delete[] h_logp;
        delete[] h_val; delete[] h_rew; delete[] h_done;
        delete[] h_last_val; delete[] h_adv; delete[] h_ret;
    }
};

int main() {
    std::cout << "Hybrid GPU/CPU PPO Training\n";
    std::cout << "===========================\n\n";
    
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << " (" << prop.multiProcessorCount << " SMs)\n\n";
    
    HybridTrainer trainer;
    trainer.init(42);
    trainer.train();
    trainer.cleanup();
    
    return 0;
}
