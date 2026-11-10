// glm_mm4_128s2f, _128w and _128wb with B pairs' halves swapped: byte gates against glm_mm4_128s2 must catch them.
#pragma once

#define GLM_MM_DEVICE_ONLY
#include "glm_mm.cu"
#undef GLM_MM_DEVICE_ONLY

extern "C" __global__ void __launch_bounds__(256, 2)
    tf_ctl_glm_mm4_128s2f_swap(const __nv_bfloat16* __restrict__ x, long long ldx, const uint32_t* __restrict__ w,
                               const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, int f32,
                               int M, int N, int K, int SK) {
    mm_kernel<4, 128, 2, 2, false, true, false, 2>(x, ldx, w, s, out, part, f32, M, N, K, SK);
}

#define TF_CTL_GLM_MM_WIDE_SWAP(NAME, SHB)                                                                              \
    extern "C" __global__ void __launch_bounds__(256, 2)                                                               \
        NAME(const __nv_bfloat16* __restrict__ x, long long ldx, const uint32_t* __restrict__ w,                       \
             const __half* __restrict__ s, void* __restrict__ out, float* __restrict__ part, int f32, int M, int N,    \
             int K, int SK) {                                                                                          \
        mm_kernel<4, 128, 4, 2, false, true, false, 2, 2, SHB>(x, ldx, w, s, out, part, f32, M, N, K, SK);              \
    }
TF_CTL_GLM_MM_WIDE_SWAP(tf_ctl_glm_mm4_128w_swap, false)
TF_CTL_GLM_MM_WIDE_SWAP(tf_ctl_glm_mm4_128wb_swap, true)
