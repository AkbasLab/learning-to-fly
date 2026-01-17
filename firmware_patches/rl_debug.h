#ifndef RL_DEBUG_H
#define RL_DEBUG_H

#include <stdint.h>
#include <stdbool.h>

#ifndef RL_DEBUG_RX
#define RL_DEBUG_RX 0
#endif

#ifndef RL_DEBUG_RING_SIZE
#define RL_DEBUG_RING_SIZE 16
#endif

#ifndef RL_DEBUG_TELEMETRY_RATE_HZ
#define RL_DEBUG_TELEMETRY_RATE_HZ 2
#endif

#if RL_DEBUG_RX

typedef struct {
    uint32_t timestamp_ms;
    float actions[4];
    uint8_t packet_type;
} rl_debug_rx_entry_t;

typedef struct {
    rl_debug_rx_entry_t entries[RL_DEBUG_RING_SIZE];
    uint16_t head;
    uint16_t count;
    uint32_t total_rx_count;
    uint32_t last_telemetry_ms;
    float action_min[4];
    float action_max[4];
    uint32_t window_start_ms;
} rl_debug_rx_state_t;

static rl_debug_rx_state_t _rl_debug_state = {0};

static inline void rl_debug_init(void) {
    _rl_debug_state.head = 0;
    _rl_debug_state.count = 0;
    _rl_debug_state.total_rx_count = 0;
    _rl_debug_state.last_telemetry_ms = 0;
    _rl_debug_state.window_start_ms = 0;
    for (int i = 0; i < 4; i++) {
        _rl_debug_state.action_min[i] = 1e9f;
        _rl_debug_state.action_max[i] = -1e9f;
    }
}

static inline void rl_debug_log_rx(uint32_t timestamp_ms, const float* actions, uint8_t packet_type) {
    rl_debug_rx_entry_t* entry = &_rl_debug_state.entries[_rl_debug_state.head];
    entry->timestamp_ms = timestamp_ms;
    entry->packet_type = packet_type;
    for (int i = 0; i < 4; i++) {
        entry->actions[i] = actions[i];
        if (actions[i] < _rl_debug_state.action_min[i]) {
            _rl_debug_state.action_min[i] = actions[i];
        }
        if (actions[i] > _rl_debug_state.action_max[i]) {
            _rl_debug_state.action_max[i] = actions[i];
        }
    }
    _rl_debug_state.head = (_rl_debug_state.head + 1) % RL_DEBUG_RING_SIZE;
    if (_rl_debug_state.count < RL_DEBUG_RING_SIZE) {
        _rl_debug_state.count++;
    }
    _rl_debug_state.total_rx_count++;
}

static inline bool rl_debug_should_print_telemetry(uint32_t now_ms) {
    uint32_t interval_ms = 1000 / RL_DEBUG_TELEMETRY_RATE_HZ;
    if (now_ms - _rl_debug_state.last_telemetry_ms >= interval_ms) {
        _rl_debug_state.last_telemetry_ms = now_ms;
        return true;
    }
    return false;
}

static inline void rl_debug_get_stats(uint32_t now_ms, float* rx_rate_hz, float* last_actions, float* min_actions, float* max_actions) {
    uint32_t window_ms = now_ms - _rl_debug_state.window_start_ms;
    if (window_ms > 0) {
        *rx_rate_hz = (float)_rl_debug_state.total_rx_count * 1000.0f / (float)window_ms;
    } else {
        *rx_rate_hz = 0.0f;
    }
    
    if (_rl_debug_state.count > 0) {
        uint16_t last_idx = (_rl_debug_state.head + RL_DEBUG_RING_SIZE - 1) % RL_DEBUG_RING_SIZE;
        for (int i = 0; i < 4; i++) {
            last_actions[i] = _rl_debug_state.entries[last_idx].actions[i];
            min_actions[i] = _rl_debug_state.action_min[i];
            max_actions[i] = _rl_debug_state.action_max[i];
        }
    }
}

static inline void rl_debug_reset_window(uint32_t now_ms) {
    _rl_debug_state.total_rx_count = 0;
    _rl_debug_state.window_start_ms = now_ms;
    for (int i = 0; i < 4; i++) {
        _rl_debug_state.action_min[i] = 1e9f;
        _rl_debug_state.action_max[i] = -1e9f;
    }
}

#else

#define rl_debug_init() ((void)0)
#define rl_debug_log_rx(ts, actions, type) ((void)0)
#define rl_debug_should_print_telemetry(now) (false)
#define rl_debug_get_stats(now, rate, last, min, max) ((void)0)
#define rl_debug_reset_window(now) ((void)0)

#endif

#endif
