// GLM's TF_GLM_HC_FUSE launch with the warps' squares added (s0 + s1) + (s2 + s3): its byte gate must catch the order.
#pragma once

#include "glm_glue_hcfuse.cuh"

extern "C" __global__ void __launch_bounds__(512)
    tf_ctl_glm_hc_post_partial_order(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g, const float* post,
                                     const float* comb, const __nv_bfloat16* fn, float* part, int64_t rs, int world,
                                     int64_t rows) {
  glm_hc_fuse::post_partial<false>(x, xout, g, post, comb, fn, part, rs, world, rows);
}
