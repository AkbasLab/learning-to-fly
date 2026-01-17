#ifndef RL_FIRMWARE_VERSION_H
#define RL_FIRMWARE_VERSION_H

#ifndef RL_GIT_HASH
#define RL_GIT_HASH "unknown"
#endif

#ifdef RL_TOOLS_PPO
#define RL_ALGO_TAG "PPO"
#define RL_ALGO_ID 1
#else
#define RL_ALGO_TAG "TD3"
#define RL_ALGO_ID 0
#endif

#define RL_FIRMWARE_VERSION_STRING RL_ALGO_TAG "_" RL_GIT_HASH

#include "param.h"

static uint8_t _rl_algo_id = RL_ALGO_ID;
static uint32_t _rl_git_hash_int = 0;

PARAM_GROUP_START(rltv)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, algo, &_rl_algo_id)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash, &_rl_git_hash_int)
PARAM_GROUP_STOP(rltv)

static inline void rl_version_init(void) {
    const char* hash_str = RL_GIT_HASH;
    _rl_git_hash_int = 0;
    for (int i = 0; i < 8 && hash_str[i] != '\0'; i++) {
        char c = hash_str[i];
        uint8_t nibble = 0;
        if (c >= '0' && c <= '9') nibble = c - '0';
        else if (c >= 'a' && c <= 'f') nibble = 10 + (c - 'a');
        else if (c >= 'A' && c <= 'F') nibble = 10 + (c - 'A');
        _rl_git_hash_int = (_rl_git_hash_int << 4) | nibble;
    }
}

#endif
