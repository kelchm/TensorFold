// Gate 3: each opt-in latent GEMM reading the next head's weights.
#pragma once
#include "glm_glue_num.cuh"
extern "C" __global__ void __launch_bounds__(256)
tf_ctl_glm_mma_kn(const __nv_bfloat16* x, int64_t xr, const __nv_bfloat16* w, __nv_bfloat16* out,
                  int64_t orow, int64_t rows, int n, int k) {
  glm_num::latent_mma<true, 3, true>(x, xr, w, out, orow, rows, n, k);
}
extern "C" __global__ void __launch_bounds__(256)
tf_ctl_glm_mma_nk(const __nv_bfloat16* x, int64_t xr, const __nv_bfloat16* w, __nv_bfloat16* out,
                  int64_t orow, int64_t rows, int n, int k) {
  glm_num::latent_mma<false, 3, true>(x, xr, w, out, orow, rows, n, k);
}
