// glm_mm4_128s2f with each B pair's halves swapped: its byte gate against glm_mm4_128s2 must catch the order.
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
