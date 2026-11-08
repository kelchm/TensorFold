#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Four independent accumulator fragments, eight issues per fragment per round; only the final sum is stored.
#define RF_MMA(NAME, OP, TYPE, CONSTRAINT) \
extern "C" __global__ __launch_bounds__(256) \
void NAME(float* out, uint32_t rounds, uint32_t operand) { \
    TYPE c[4][4] = {}; \
    for (uint32_t i = 0; i < rounds; ++i) { \
        _Pragma("unroll") \
        for (int u = 0; u < 8; ++u) { \
            _Pragma("unroll") \
            for (int j = 0; j < 4; ++j) { \
                asm volatile(OP " {%0,%1,%2,%3}, {%4,%4,%4,%4}, {%4,%4}, {%0,%1,%2,%3};" \
                    : CONSTRAINT(c[j][0]), CONSTRAINT(c[j][1]), CONSTRAINT(c[j][2]), CONSTRAINT(c[j][3]) \
                    : "r"(operand)); \
            } \
        } \
    } \
    float sum = 0; \
    _Pragma("unroll") \
    for (int j = 0; j < 4; ++j) { \
        _Pragma("unroll") \
        for (int q = 0; q < 4; ++q) sum += float(c[j][q]); \
    } \
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum; \
}
