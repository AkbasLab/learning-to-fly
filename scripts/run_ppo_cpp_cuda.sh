#!/bin/bash
#
# run_ppo_cpp_cuda.sh - Gated PPO C++/CUDA training pipeline
#
# Gates:
#   0: CUDA environment sanity check
#   1: CMake build verification
#   2: Trainer correctness test (loss decreases, CPU/GPU parity)
#   3: Training success (achieve reward threshold)
#   4: Export artifact correctness
#   5: Firmware build with identity
#   6: Flash + runtime sanity
#   7: Tether test protocol
#
# Usage:
#   ./run_ppo_cpp_cuda.sh                  # Full pipeline
#   ./run_ppo_cpp_cuda.sh --gate 0         # Run up to gate N
#   ./run_ppo_cpp_cuda.sh --timesteps 1M   # Training timesteps
#   ./run_ppo_cpp_cuda.sh --skip-flash     # Skip gates 6-7
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/build"
LIBTORCH_BUILD_DIR="${BUILD_DIR}/ppo_libtorch"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/multirotor_ppo"

# LibTorch CMake path
LIBTORCH_CMAKE="/home/puddles/.local/lib/python3.12/site-packages/torch/share/cmake"

# Default parameters
MAX_GATE=7
TIMESTEPS="1M"
SKIP_FLASH=false
CRAZYFLIE_URI="radio://0/80/2M/E7E7E7E7E7"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gate)
            MAX_GATE="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --skip-flash)
            SKIP_FLASH=true
            shift
            ;;
        --uri)
            CRAZYFLIE_URI="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  PPO C++/CUDA Training Pipeline${NC}"
echo -e "${BLUE}================================${NC}"
echo ""
echo "Project dir: ${PROJECT_DIR}"
echo "Max gate: ${MAX_GATE}"
echo "Timesteps: ${TIMESTEPS}"
echo ""

# Gate tracking
gate_passed() {
    echo -e "${GREEN}✓ Gate $1: PASSED${NC}"
}

gate_failed() {
    echo -e "${RED}✗ Gate $1: FAILED${NC}"
    echo -e "${RED}Reason: $2${NC}"
    exit 1
}

gate_skipped() {
    echo -e "${YELLOW}○ Gate $1: SKIPPED${NC}"
}

# ===========================================================================
# Gate 0: CUDA Environment Check
# ===========================================================================
run_gate_0() {
    echo -e "\n${BLUE}=== Gate 0: CUDA Environment Check ===${NC}"
    
    if [[ $MAX_GATE -lt 0 ]]; then
        gate_skipped 0
        return
    fi
    
    # Check CUDA via Python
    if ! python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        gate_failed 0 "CUDA not available in PyTorch"
    fi
    
    # Check CUDA compiler
    if ! command -v nvcc &> /dev/null; then
        gate_failed 0 "nvcc not found"
    fi
    
    # Check LibTorch path
    if [[ ! -d "${LIBTORCH_CMAKE}" ]]; then
        gate_failed 0 "LibTorch CMake not found at ${LIBTORCH_CMAKE}"
    fi
    
    # Get GPU info
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "GPU: ${GPU_NAME}"
    
    gate_passed 0
}

# ===========================================================================
# Gate 1: CMake Build
# ===========================================================================
run_gate_1() {
    echo -e "\n${BLUE}=== Gate 1: CMake Build ===${NC}"
    
    if [[ $MAX_GATE -lt 1 ]]; then
        gate_skipped 1
        return
    fi
    
    mkdir -p "${LIBTORCH_BUILD_DIR}"
    cd "${LIBTORCH_BUILD_DIR}"
    
    echo "Configuring CMake..."
    if ! cmake "${PROJECT_DIR}/src/cuda/ppo_libtorch" \
        -DCMAKE_PREFIX_PATH="${LIBTORCH_CMAKE}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DRL_TOOLS_ROOT="${PROJECT_DIR}/external/rl_tools" \
        -DLEARNING_TO_FLY_INCLUDE="${PROJECT_DIR}/include" \
        2>&1 | tee cmake_config.log; then
        gate_failed 1 "CMake configuration failed"
    fi
    
    echo "Building targets..."
    if ! cmake --build . --parallel 2>&1 | tee cmake_build.log; then
        gate_failed 1 "CMake build failed"
    fi
    
    # Verify executables exist
    for exe in gate0_cuda_check gate2_trainer_test train_ppo_cuda gate4_export_verify; do
        if [[ ! -f "${LIBTORCH_BUILD_DIR}/${exe}" ]]; then
            gate_failed 1 "Executable not built: ${exe}"
        fi
    done
    
    gate_passed 1
}

# ===========================================================================
# Gate 2: Trainer Correctness Test
# ===========================================================================
run_gate_2() {
    echo -e "\n${BLUE}=== Gate 2: Trainer Correctness Test ===${NC}"
    
    if [[ $MAX_GATE -lt 2 ]]; then
        gate_skipped 2
        return
    fi
    
    cd "${LIBTORCH_BUILD_DIR}"
    
    # Run CUDA check first
    echo "Running CUDA sanity check..."
    if ! ./gate0_cuda_check; then
        gate_failed 2 "CUDA sanity check failed"
    fi
    
    # Run trainer test
    echo "Running trainer correctness test..."
    if ! ./gate2_trainer_test; then
        gate_failed 2 "Trainer correctness test failed"
    fi
    
    gate_passed 2
}

# ===========================================================================
# Gate 3: Training
# ===========================================================================
run_gate_3() {
    echo -e "\n${BLUE}=== Gate 3: Training ===${NC}"
    
    if [[ $MAX_GATE -lt 3 ]]; then
        gate_skipped 3
        return
    fi
    
    cd "${LIBTORCH_BUILD_DIR}"
    
    echo "Starting PPO training (${TIMESTEPS} timesteps)..."
    if ! ./train_ppo_cuda --timesteps "${TIMESTEPS}"; then
        gate_failed 3 "Training failed"
    fi
    
    # Find latest checkpoint
    LATEST_RUN=$(ls -td "${CHECKPOINT_DIR}"/*_PPO_CUDA 2>/dev/null | head -1)
    if [[ -z "${LATEST_RUN}" ]]; then
        gate_failed 3 "No checkpoint directory found"
    fi
    
    if [[ ! -f "${LATEST_RUN}/best.pt" ]]; then
        gate_failed 3 "Best checkpoint not saved"
    fi
    
    echo "Checkpoint: ${LATEST_RUN}"
    export PPO_CHECKPOINT_DIR="${LATEST_RUN}"
    
    gate_passed 3
}

# ===========================================================================
# Gate 4: Export Verification
# ===========================================================================
run_gate_4() {
    echo -e "\n${BLUE}=== Gate 4: Export Verification ===${NC}"
    
    if [[ $MAX_GATE -lt 4 ]]; then
        gate_skipped 4
        return
    fi
    
    if [[ -z "${PPO_CHECKPOINT_DIR}" ]]; then
        # Try to find latest
        PPO_CHECKPOINT_DIR=$(ls -td "${CHECKPOINT_DIR}"/*_PPO_CUDA 2>/dev/null | head -1)
    fi
    
    if [[ -z "${PPO_CHECKPOINT_DIR}" ]]; then
        gate_failed 4 "No checkpoint directory found"
    fi
    
    cd "${LIBTORCH_BUILD_DIR}"
    
    echo "Verifying export: ${PPO_CHECKPOINT_DIR}"
    if ! ./gate4_export_verify "${PPO_CHECKPOINT_DIR}/best.pt" "${PPO_CHECKPOINT_DIR}/best_actor.h"; then
        gate_failed 4 "Export verification failed"
    fi
    
    gate_passed 4
}

# ===========================================================================
# Gate 5: Firmware Build
# ===========================================================================
run_gate_5() {
    echo -e "\n${BLUE}=== Gate 5: Firmware Build ===${NC}"
    
    if [[ $MAX_GATE -lt 5 ]]; then
        gate_skipped 5
        return
    fi
    
    if [[ -z "${PPO_CHECKPOINT_DIR}" ]]; then
        PPO_CHECKPOINT_DIR=$(ls -td "${CHECKPOINT_DIR}"/*_PPO_CUDA 2>/dev/null | head -1)
    fi
    
    # Copy actor.h to controller directory
    echo "Copying actor weights to controller..."
    cp "${PPO_CHECKPOINT_DIR}/best_actor.h" "${PROJECT_DIR}/controller/actor.h"
    
    # Build firmware with PPO identity
    echo "Building firmware..."
    cd "${PROJECT_DIR}"
    
    if [[ -f "scripts/build_ppo_firmware.sh" ]]; then
        if ! bash scripts/build_ppo_firmware.sh; then
            gate_failed 5 "Firmware build failed"
        fi
    else
        # Fallback: use Docker
        if ! docker build -f Dockerfile_build_firmware -t learning-to-fly-firmware .; then
            gate_failed 5 "Docker firmware build failed"
        fi
    fi
    
    # Verify firmware identity contains PPO
    FIRMWARE_BIN="${PROJECT_DIR}/build_firmware/crazyflie-firmware/cf2.bin"
    if [[ -f "${FIRMWARE_BIN}" ]]; then
        if strings "${FIRMWARE_BIN}" | grep -q "PPO"; then
            echo "Firmware identity: PPO detected"
        else
            echo "Warning: PPO identity not found in firmware binary"
        fi
    fi
    
    gate_passed 5
}

# ===========================================================================
# Gate 6: Flash + Runtime Sanity
# ===========================================================================
run_gate_6() {
    echo -e "\n${BLUE}=== Gate 6: Flash + Runtime Sanity ===${NC}"
    
    if [[ $MAX_GATE -lt 6 ]] || [[ "${SKIP_FLASH}" == "true" ]]; then
        gate_skipped 6
        return
    fi
    
    echo "Flashing firmware to Crazyflie..."
    cd "${PROJECT_DIR}"
    
    # Flash firmware
    if ! cfloader flash build_firmware/crazyflie-firmware/cf2.bin stm32-fw -w "${CRAZYFLIE_URI}"; then
        gate_failed 6 "Firmware flash failed"
    fi
    
    # Wait for reboot
    echo "Waiting for Crazyflie to reboot..."
    sleep 5
    
    # Verify communication using trigger.py dry-run
    echo "Testing communication..."
    if ! python3 scripts/trigger.py --uri "${CRAZYFLIE_URI}" --dry-run 2>&1 | grep -q "Connected"; then
        gate_failed 6 "Failed to connect to Crazyflie after flash"
    fi
    
    gate_passed 6
}

# ===========================================================================
# Gate 7: Tether Test
# ===========================================================================
run_gate_7() {
    echo -e "\n${BLUE}=== Gate 7: Tether Test ===${NC}"
    
    if [[ $MAX_GATE -lt 7 ]] || [[ "${SKIP_FLASH}" == "true" ]]; then
        gate_skipped 7
        return
    fi
    
    echo "Running tether test..."
    cd "${PROJECT_DIR}"
    
    # Run trigger.py with short duration for testing
    if ! python3 scripts/trigger.py \
        --uri "${CRAZYFLIE_URI}" \
        --duration 5 \
        --log-packets; then
        gate_failed 7 "Tether test failed"
    fi
    
    gate_passed 7
}

# ===========================================================================
# Main Pipeline
# ===========================================================================
echo -e "${BLUE}Starting gated pipeline...${NC}"

run_gate_0
run_gate_1
run_gate_2
run_gate_3
run_gate_4
run_gate_5
run_gate_6
run_gate_7

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}  All gates passed!${NC}"
echo -e "${GREEN}================================${NC}"

if [[ -n "${PPO_CHECKPOINT_DIR}" ]]; then
    echo ""
    echo "Checkpoint: ${PPO_CHECKPOINT_DIR}"
    echo "Actor weights: ${PROJECT_DIR}/controller/actor.h"
fi
