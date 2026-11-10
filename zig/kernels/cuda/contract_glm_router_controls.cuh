// TF_GLM_ROUTER_FUSE's launch as one unbroken K chain, not eight slices added in order: its byte gate must catch it.
#pragma once

#include "glm_glue_router_tile.cuh"

extern "C" __global__ void __launch_bounds__(128)
    tf_ctl_glm_router_chain(const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int64_t rows,
                            int64_t x_stride, int64_t d, int64_t ne, int ks) {
  glm_router_tile::tiles<true, true>(x, w, out, rows, x_stride, d, ne, ks);
}
