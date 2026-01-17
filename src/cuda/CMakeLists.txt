cmake_minimum_required(VERSION 3.23)
project(multirotor_cuda CUDA CXX)

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
# PPO CUDA Training
# ============================================================
add_executable(training_ppo_cuda training_ppo.cu)
target_link_libraries(training_ppo_cuda
        rl_tools
        rl_tools::optimizations_cuda
        learning_to_fly
)
target_include_directories(training_ppo_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
RL_TOOLS_TAG_IS_CUDA(training_ppo_cuda)
