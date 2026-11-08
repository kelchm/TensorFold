// Synthetic hardware probes, called only by tf-cuda-test roofline; no model operators.
#include "roofline_mma.cuh"

__device__ __forceinline__ uint4 rf_load(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void rf_store(uint4* p, uint4 v) {
    asm volatile("st.global.wb.v4.u32 [%0], {%1,%2,%3,%4};"
        :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint4 rf_pattern(uint64_t i) {
    uint32_t x = uint32_t(i) ^ (uint32_t(i >> 32) + 0x9e3779b9u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return make_uint4(x, x ^ 0xa5a5a5a5u, x * 1664525u + 1013904223u, ~x);
}

extern "C" __global__ __launch_bounds__(256)
void tf_roof_read(const uint4* src, uint4*, uint64_t n, uint4* sink, uint32_t passes) {
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t i = tid; i < n; i += stride) {
            const uint4 v = rf_load(src + i);
            acc.x += v.x;
            acc.y += v.y;
            acc.z += v.z;
            acc.w += v.w;
        }
    for (int s = 16; s > 0; s >>= 1) {
        acc.x += __shfl_xor_sync(0xffffffffu, acc.x, s);
        acc.y += __shfl_xor_sync(0xffffffffu, acc.y, s);
        acc.z += __shfl_xor_sync(0xffffffffu, acc.z, s);
        acc.w += __shfl_xor_sync(0xffffffffu, acc.w, s);
    }
    if ((threadIdx.x & 31) == 0) sink[tid / 32] = acc;
}

extern "C" __global__ __launch_bounds__(256)
void tf_roof_write(const uint4*, uint4* dst, uint64_t n, uint4*, uint32_t passes) {
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t i = tid; i < n; i += stride) rf_store(dst + i, rf_pattern(i));
}

extern "C" __global__ __launch_bounds__(256)
void tf_roof_copy(const uint4* src, uint4* dst, uint64_t n, uint4*, uint32_t passes) {
    const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t stride = uint64_t(gridDim.x) * blockDim.x;
    for (uint32_t pass = 0; pass < passes; ++pass)
        for (uint64_t i = tid; i < n; i += stride) rf_store(dst + i, rf_load(src + i));
}

#if __CUDA_ARCH__ >= 800
RF_MMA(tf_roof_f16, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", float, "+f")
RF_MMA(tf_roof_bf16, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", float, "+f")
RF_MMA(tf_roof_i8, "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32", int32_t, "+r")
#endif
#if __CUDA_ARCH__ >= 890
RF_MMA(tf_roof_f8, "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32", float, "+f")
#endif

extern "C" __global__ __launch_bounds__(256)
void tf_roof_fma(float* out, uint32_t rounds, uint32_t) {
    float c[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (uint32_t i = 0; i < rounds; ++i) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(c[j]) : "f"(0.5f), "f"(0.25f));
        }
    }
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) sum += c[j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}
