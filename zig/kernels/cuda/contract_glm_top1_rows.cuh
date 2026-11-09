// Gate 3: tf_glm_top1_rows with each row's words 2 apart instead of 4, so rows past the first land in padding.
#pragma once
#include "glm_top1.cuh"
extern "C" __global__ void __launch_bounds__(1024) tf_ctl_glm_top1_rows_words(const unsigned short* logits, int64_t n,
                                                                               unsigned lo, unsigned* out) {
  glm_top1<false>(logits + static_cast<int64_t>(blockIdx.x) * n, n, lo, out + static_cast<int64_t>(blockIdx.x) * 2);
}
