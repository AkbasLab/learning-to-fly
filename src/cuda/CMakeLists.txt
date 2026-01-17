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
