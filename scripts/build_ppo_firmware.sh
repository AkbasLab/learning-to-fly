#!/bin/bash
# build_ppo_firmware.sh - Build Crazyflie firmware with PPO normalization support
#
# This script builds the firmware with the PPO observation normalization fix.
# It modifies the Makefile to add -DRL_TOOLS_PPO to enable normalization.
#
# Usage:
#   ./build_ppo_firmware.sh [checkpoint_path]
#
# Arguments:
#   checkpoint_path - Path to the PPO checkpoint directory containing actor.h
#                     Default: controller/actor.h (already in place)
#
# Output:
#   build_firmware/cf2.bin - The built firmware binary

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/build_firmware"

# Parse arguments
CHECKPOINT_PATH="${1:-$PROJECT_ROOT/controller/actor.h}"

echo "=== PPO Firmware Builder ==="
echo "Project root: $PROJECT_ROOT"
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output: $OUTPUT_DIR/cf2.bin"
echo ""

# Validate checkpoint
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

# Check if checkpoint has observation normalizer (PPO indicator)
if ! grep -q "observation_normalizer" "$CHECKPOINT_PATH"; then
    echo "WARNING: Checkpoint does not contain observation_normalizer namespace."
    echo "         This might be a TD3 checkpoint. PPO requires normalization."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check algorithm metadata
ALGO=$(grep -o 'algorithm\[\] = "[^"]*"' "$CHECKPOINT_PATH" | sed 's/.*= "\(.*\)"/\1/' || echo "unknown")
echo "Detected algorithm: $ALGO"

if [ "$ALGO" != "PPO" ]; then
    echo "WARNING: Algorithm metadata shows '$ALGO', not 'PPO'"
fi

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Build using Docker with PPO flag
echo ""
echo "Building firmware with PPO normalization enabled..."
echo ""

docker run -it --rm \
    -v "$CHECKPOINT_PATH":/controller/data/actor.h:ro \
    -v "$OUTPUT_DIR":/output \
    arpllab/learning_to_fly_build_firmware \
    bash -c '
        # Add PPO define to CFLAGS
        echo "Adding -DRL_TOOLS_PPO to build..."
        export EXTRA_CFLAGS="-DRL_TOOLS_PPO"
        
        # Modify Makefile to include the flag (if needed)
        # The cleaner approach is to use the existing CFLAGS mechanism
        sed -i "s/^CFLAGS = /CFLAGS = -DRL_TOOLS_PPO /" Makefile 2>/dev/null || true
        
        # Build
        make -j$(nproc) && cp build/cf2.bin /output/cf2.bin
        
        echo ""
        echo "Build complete! Output: /output/cf2.bin"
    '

if [ -f "$OUTPUT_DIR/cf2.bin" ]; then
    echo ""
    echo "=== SUCCESS ==="
    echo "PPO firmware built: $OUTPUT_DIR/cf2.bin"
    echo "Size: $(ls -lh "$OUTPUT_DIR/cf2.bin" | awk '{print $5}')"
    echo ""
    echo "Flash with:"
    echo "  cfloader flash $OUTPUT_DIR/cf2.bin stm32-fw"
else
    echo "ERROR: Build failed - cf2.bin not found"
    exit 1
fi
