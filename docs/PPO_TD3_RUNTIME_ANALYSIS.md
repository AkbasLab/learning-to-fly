# PPO vs TD3 Runtime Analysis Report

## Executive Summary

**ROOT CAUSE IDENTIFIED**: The PPO-trained policy fails in real flight because the STM32 firmware does not apply the observation normalization that PPO requires. Additionally, there is a potential observation dimension mismatch.

## 1. Runtime Contract: TD3 (Known Good)

### 1.1 Packet Format

**Trigger Packet (Learned Policy)**
- Port: `COMMANDER_GENERIC` (7)
- Channel: `META_COMMAND_CHANNEL` (1)
- Payload: 1 byte = `0x01`
- Meaning: Activate learned controller for one timestep

**Hover Packet (Original Controller)**
- Port: `COMMANDER_GENERIC` (7)
- Channel: `SET_SETPOINT_CHANNEL` (0)
- Payload: 17 bytes = `<Bffff>` (little-endian)
  - `B`: TYPE_HOVER (0x05)
  - `f`: vx (velocity X, m/s)
  - `f`: vy (velocity Y, m/s)
  - `f`: yawrate (rad/s)
  - `f`: height (m)

### 1.2 Firmware Interface (TD3)

**File**: `rl_tools_adapter.cpp`

**Input**: 13-element float array `state_input[]`:
- `[0-2]`: Position error (clipped)
- `[3-6]`: Quaternion (qw, qx, qy, qz)
- `[7-9]`: Velocity error (clipped)
- `[10-12]`: Angular velocity (rad/s)

**Observation Construction**: Converted to 18-element rotation matrix observation:
- `[0-2]`: Position
- `[3-11]`: 3x3 rotation matrix (from quaternion)
- `[12-14]`: Linear velocity
- `[15-17]`: Angular velocity

**With Action History** (32 steps × 4 actions = 128 elements):
- Total observation: 18 + 128 = **146 elements**

**Output**: 4-element float array `action_output[]`:
- Range: [-1, 1] (tanh output)
- Conversion: `motor_cmd = ((action + 1) / 2) * MAX_RPM / MAX_RPM * UINT16_MAX`

### 1.3 TD3 Specifics
- **No observation normalization** during training or deployment
- Observations are used directly
- Works because TD3 learns to handle raw input scales

---

## 2. PPO Critical Differences

### 2.1 Observation Normalization (CRITICAL)

PPO uses running mean/variance normalization during training:
```cpp
normalized_obs = (obs - mean) / std
normalized_obs = clip(normalized_obs, -10, 10)
```

**This normalization MUST be applied in firmware before inference.**

The PPO checkpoint includes normalization parameters in `actor.h`:
```cpp
namespace rl_tools::checkpoint::observation_normalizer {
    constexpr int DIM = 146;
    constexpr float mean[DIM] = {...};
    constexpr float std_inv[DIM] = {...};
    
    inline void normalize(float* obs) {
        for (int i = 0; i < DIM; i++) {
            obs[i] = (obs[i] - mean[i]) * std_inv[i];
            // clip to [-10, 10]
        }
    }
}
```

### 2.2 The Bug

**Current firmware (`rl_tools_adapter.cpp`)**:
```cpp
void rl_tools_control(float* state, float* actions){
    // Constructs observation
    observe_rotation_matrix(state_matrix, state_rotation_matrix_input);
    
    // ACTION HISTORY IS APPENDED (good)
    // ...
    
    // DIRECTLY CALLS EVALUATE WITHOUT NORMALIZATION (BAD!)
    rlt::evaluate(device, rlt::checkpoint::actor::model, input, output, buffers);
}
```

**What happens**:
1. PPO policy was trained with normalized inputs (mean≈0, std≈1)
2. Firmware provides raw inputs (e.g., z-position≈-0.15, not normalized)
3. Raw inputs pass through actor without normalization
4. Actor sees wildly different input distribution → erratic actions

---

## 3. Evidence

### 3.1 Observation Normalizer in actor.h

From `controller/actor.h`:
```cpp
namespace rl_tools::checkpoint::observation_normalizer {
    constexpr int DIM = 146;
    constexpr float CLIP_VALUE = 10.0f;
    constexpr float count = 245760.0f;  // ~245k samples used for statistics
    
    constexpr float mean[DIM] = {
        1.656406312e-05f, -0.00045450998f, -0.1560325772f, ...
    };
    
    constexpr float std_inv[DIM] = {
        4.929134846f, 5.000530243f, 4.897408485f, ...
    };
}
```

### 3.2 Algorithm Identification

From `controller/actor.h`:
```cpp
namespace rl_tools::checkpoint::meta{
   char name[] = "2026_01_16_22_44_57_PPO_...";
   char algorithm[] = "PPO";  // <-- PPO checkpoint!
}
```

### 3.3 Firmware Does Not Call Normalize

From GitHub `arplaboratory/learning_to_fly_controller` `rl_tools_adapter.cpp`:
- Line 105-137: `rl_tools_control()` constructs observation and calls evaluate
- **No call to observation normalization anywhere**

---

## 4. Required Fixes

### 4.1 Firmware Fix: Add Observation Normalization

The firmware must be modified to call the normalizer when PPO is detected.

**Option A (Compile-time)**:
```cpp
#ifdef RL_TOOLS_PPO
    // Apply normalization before inference
    rl_tools::checkpoint::observation_normalizer::normalize((float*)&input);
#endif
```

**Option B (Runtime detection)**:
```cpp
// Check if normalizer exists in checkpoint
#ifdef RL_TOOLS_CHECKPOINT_HAS_NORMALIZER
    rl_tools::checkpoint::observation_normalizer::normalize(input_array);
#endif
```

### 4.2 Minimal Patch (Recommended)

Add the following to `rl_tools_adapter.cpp` just before `rlt::evaluate()`:

```cpp
// PPO FIX: Apply observation normalization before inference
#ifdef RL_TOOLS_PPO
    {
        T input_buffer[ACTOR_TYPE::SPEC::INPUT_DIM];
        for (TI i = 0; i < ACTOR_TYPE::SPEC::INPUT_DIM; i++) {
            input_buffer[i] = rlt::get(input, 0, i);
        }
        rlt::checkpoint::observation_normalizer::normalize(input_buffer);
        for (TI i = 0; i < ACTOR_TYPE::SPEC::INPUT_DIM; i++) {
            rlt::set(input, 0, i, input_buffer[i]);
        }
    }
#endif
```

Then build with `-DRL_TOOLS_PPO` when deploying PPO policies.

**See**: `firmware_patches/PPO_NORMALIZATION_PATCH.txt` for complete patch.
**See**: `firmware_patches/rl_tools_adapter_ppo.cpp` for full drop-in replacement.

### 4.3 Firmware Version Tagging

Add compile-time version info to firmware:
```cpp
PARAM_GROUP_START(rltv)  // rl_tools version
PARAM_ADD(PARAM_UINT8, algo, &algorithm_id)  // 0=TD3, 1=PPO
PARAM_ADD(PARAM_FLOAT, major, &version_major)
PARAM_ADD(PARAM_FLOAT, minor, &version_minor)
PARAM_GROUP_STOP(rltv)
```

---

## 5. Verification Commands

### 5.1 Build and Flash

**TD3 Firmware**:
```bash
# Ensure actor.h is from TD3 checkpoint (no observation_normalizer namespace)
docker run -it --rm \
  -v $(pwd)/checkpoints/multirotor_td3/.../actor.h:/controller/data/actor.h:ro \
  -v $(pwd)/build_firmware:/output \
  arpllab/learning_to_fly_build_firmware

cfloader flash build_firmware/cf2.bin stm32-fw -w radio://0/80/2M
```

**PPO Firmware (with normalization fix)**:
```bash
# Option 1: Use build script (recommended)
./scripts/build_ppo_firmware.sh controller/actor.h

# Option 2: Manual with Docker + PPO flag
docker run -it --rm \
  -v $(pwd)/controller/actor.h:/controller/data/actor.h:ro \
  -v $(pwd)/build_firmware:/output \
  arpllab/learning_to_fly_build_firmware \
  bash -c 'sed -i "s/^CFLAGS = /CFLAGS = -DRL_TOOLS_PPO /" Makefile && make -j$(nproc) && cp build/cf2.bin /output/cf2.bin'

# Flash
cfloader flash build_firmware/cf2.bin stm32-fw -w radio://0/80/2M
```

### 5.2 Verify Firmware Identity

```bash
python scripts/trigger_instrumented.py --verify-firmware
```

### 5.3 Test with Logging

```bash
# TD3 test
python scripts/trigger_instrumented.py --mode hover_learned --log-packets

# PPO test  
python scripts/trigger_instrumented.py --mode hover_learned --log-packets

# Compare logs
diff packet_logs/packets_TD3_*.log packet_logs/packets_PPO_*.log
```

### 5.4 Dry Run (No Radio TX)

```bash
python scripts/trigger_instrumented.py --mode hover_learned --dry-run --log-packets
```

---

## 6. Quick Diagnostic Checklist

| Check | Expected TD3 | Expected PPO | How to Verify |
|-------|--------------|--------------|---------------|
| actor.h has `observation_normalizer` | NO | YES | `grep observation_normalizer controller/actor.h` |
| meta::algorithm | "TD3" | "PPO" | `grep "algorithm\[\]" controller/actor.h` |
| Layer 0 input size | 146 | 146 | Check actor model structure |
| Normalization called | N/A | Required | Firmware code inspection |

---

## 7. Summary

**The PPO policy failure is NOT a packet/radio issue.** Both TD3 and PPO use identical packet formats.

**The issue is in the firmware inference pipeline:**
1. PPO requires observation normalization before inference
2. The exported checkpoint contains the normalization parameters
3. The firmware never calls the normalization function
4. Without normalization, the policy sees wrong input distribution
5. This causes erratic/full-throttle behavior

**Fix**: Modify `rl_tools_adapter.cpp` to call `observation_normalizer::normalize()` when a PPO checkpoint is used.
