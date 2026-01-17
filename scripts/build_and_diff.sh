#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/build_firmware"
DUMPS_DIR="$PROJECT_ROOT/dumps"

TD3_ACTOR="${1:-$PROJECT_ROOT/checkpoints/multirotor_td3/actor.h}"
PPO_ACTOR="${2:-$PROJECT_ROOT/controller/actor.h}"
URI="${3:-radio://0/80/2M/E7E7E7E7E7}"

mkdir -p "$OUTPUT_DIR" "$DUMPS_DIR"

GIT_HASH=$(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "nogit")

build_firmware() {
    local actor_path="$1"
    local algo="$2"
    local output_bin="$OUTPUT_DIR/cf2_${algo}.bin"
    
    echo "=== Building $algo firmware ==="
    echo "Actor: $actor_path"
    
    local extra_flags="-DRL_GIT_HASH=\\\"$GIT_HASH\\\""
    if [ "$algo" = "PPO" ]; then
        extra_flags="$extra_flags -DRL_TOOLS_PPO"
    fi
    
    docker run --rm \
        -v "$actor_path":/controller/data/actor.h:ro \
        -v "$PROJECT_ROOT/firmware_patches/rl_debug.h":/controller/src/rl_debug.h:ro \
        -v "$PROJECT_ROOT/firmware_patches/rl_firmware_version.h":/controller/src/rl_firmware_version.h:ro \
        -v "$OUTPUT_DIR":/output \
        arpllab/learning_to_fly_build_firmware \
        bash -c "sed -i 's/^CFLAGS = /CFLAGS = $extra_flags /' Makefile && make -j\$(nproc) && cp build/cf2.bin /output/cf2_${algo}.bin"
    
    echo "Built: $output_bin"
}

flash_firmware() {
    local algo="$1"
    local bin_path="$OUTPUT_DIR/cf2_${algo}.bin"
    
    echo "=== Flashing $algo firmware ==="
    cfloader flash "$bin_path" stm32-fw -w "$URI"
    sleep 3
}

run_dump() {
    local algo="$1"
    local dump_path="$DUMPS_DIR/${algo}_packets.jsonl"
    
    echo "=== Running $algo dump ==="
    python3 "$SCRIPT_DIR/trigger.py" \
        --uri "$URI" \
        --mode hover_learned \
        --verify-firmware \
        --expected-algo "$algo" \
        --dump "$dump_path" \
        --log-rate 100 \
        --height 0.0 &
    
    local pid=$!
    sleep 5
    kill $pid 2>/dev/null || true
    
    echo "Dump: $dump_path"
}

echo "TD3 actor: $TD3_ACTOR"
echo "PPO actor: $PPO_ACTOR"
echo "URI: $URI"
echo ""

if [ -f "$TD3_ACTOR" ]; then
    build_firmware "$TD3_ACTOR" "TD3"
else
    echo "WARNING: TD3 actor not found at $TD3_ACTOR, skipping TD3 build"
fi

if [ -f "$PPO_ACTOR" ]; then
    build_firmware "$PPO_ACTOR" "PPO"
else
    echo "WARNING: PPO actor not found at $PPO_ACTOR, skipping PPO build"
fi

echo ""
echo "=== Build complete ==="
echo "TD3 binary: $OUTPUT_DIR/cf2_TD3.bin"
echo "PPO binary: $OUTPUT_DIR/cf2_PPO.bin"
echo ""
echo "To flash and run dumps:"
echo "  cfloader flash $OUTPUT_DIR/cf2_TD3.bin stm32-fw -w $URI"
echo "  python3 scripts/trigger.py --verify-firmware --expected-algo TD3 --dump dumps/TD3.jsonl"
echo ""
echo "  cfloader flash $OUTPUT_DIR/cf2_PPO.bin stm32-fw -w $URI"
echo "  python3 scripts/trigger.py --verify-firmware --expected-algo PPO --dump dumps/PPO.jsonl"
echo ""
echo "  diff dumps/TD3.jsonl dumps/PPO.jsonl"
