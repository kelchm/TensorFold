// Load eight packed E4M3 key codes into the BF16 shared tile. The row's exact scale is applied after MMA.
#pragma once
#include "common/fp_pairs.cuh"
namespace glm_packed_key {
__device__ __forceinline__ void stage(__nv_bfloat16* dst, float* scale, const __nv_bfloat16* base, int u, bool ok) {
  const unsigned char* row = reinterpret_cast<const unsigned char*>(base);
  const uint2 words = ok ? __ldcg(reinterpret_cast<const uint2*>(row) + u) : make_uint2(0, 0);
  if (u == 0) *scale = ok ? __ldcg(reinterpret_cast<const float*>(row + 128)) : 1.0f;
  *reinterpret_cast<uint4*>(dst) = make_uint4(tf_common::fp8pair(words.x), tf_common::fp8pair(words.x >> 16),
                                             tf_common::fp8pair(words.y), tf_common::fp8pair(words.y >> 16));
}
}
