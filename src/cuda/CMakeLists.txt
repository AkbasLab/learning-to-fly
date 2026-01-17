cmake_minimum_required(VERSION 3.23)
project(multirotor_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

set(CMAKE_CUDA_ARCHITECTURES "native")

add_executable(cuda_benchmark benchmark.cu)

target_link_libraries(cuda_benchmark
        rl_tools::minimal
        rl_tools::optimizations_cuda
        learning_to_fly
)
RL_TOOLS_TAG_IS_CUDA(cuda_benchmark)

# ============================================================
# CUDA-accelerated PPO Training
# ============================================================
add_executable(training_ppo_cuda training_ppo_cuda.cu)
target_link_libraries(training_ppo_cuda
        rl_tools
        rl_tools::optimizations_cuda
        learning_to_fly
)
RL_TOOLS_TAG_IS_CUDA(training_ppo_cuda)
target_compile_definitions(training_ppo_cuda PRIVATE RL_TOOLS_BACKEND_ENABLE_CUDA)

# ============================================================
# Full GPU PPO Training (no CPU bottleneck)
# ============================================================
add_executable(training_ppo_fullgpu training_ppo_fullgpu.cu)
target_link_libraries(training_ppo_fullgpu
        rl_tools::minimal
        rl_tools::optimizations_cuda
)
RL_TOOLS_TAG_IS_CUDA(training_ppo_fullgpu)
target_compile_definitions(training_ppo_fullgpu PRIVATE RL_TOOLS_BACKEND_ENABLE_CUDA)

# Full GPU PPO v2 (optimized)
add_executable(training_ppo_fullgpu_v2 training_ppo_fullgpu_v2.cu)
target_link_libraries(training_ppo_fullgpu_v2
        rl_tools::minimal
        rl_tools::optimizations_cuda
)
RL_TOOLS_TAG_IS_CUDA(training_ppo_fullgpu_v2)
target_compile_definitions(training_ppo_fullgpu_v2 PRIVATE RL_TOOLS_BACKEND_ENABLE_CUDA)

# Hybrid GPU/CPU PPO (GPU rollout, CPU update)
add_executable(training_ppo_hybrid training_ppo_hybrid.cu)
target_link_libraries(training_ppo_hybrid
        rl_tools::minimal
        rl_tools::optimizations_cuda
)
RL_TOOLS_TAG_IS_CUDA(training_ppo_hybrid)
target_compile_definitions(training_ppo_hybrid PRIVATE RL_TOOLS_BACKEND_ENABLE_CUDA)
