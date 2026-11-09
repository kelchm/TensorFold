// GLM's greedy pick a row: tf_glm_top1's body, shared by the glue's launches and the contract's planted control.
#pragma once
#include <stdint.h>

// decode.sample_rows' greedy pick, as the host scan: first max of n bf16 logits, NaN never wins; out fp32 bits, lo+id.
template <bool sums>
__device__ __forceinline__ void glm_top1(const unsigned short* logits, int64_t n, unsigned lo, unsigned* out) {
  __shared__ float sv[1024];
  __shared__ unsigned si[1024];
  const int t = threadIdx.x;
  float top = -__int_as_float(0x7f800000);
  unsigned best = 0xffffffffu;
  for (int64_t i = t; i < n; i += 1024) {
    const float v = __int_as_float(static_cast<int>(static_cast<unsigned>(logits[i]) << 16));
    if (v > top) top = v, best = static_cast<unsigned>(i);
  }
  sv[t] = top, si[t] = best;
  __syncthreads();
  for (int h = 512; h > 0; h >>= 1) {
    if (t < h) {
      const float a = sv[t], b = sv[t + h];
      const unsigned ia = si[t], ib = si[t + h];
      if (b > a || (b == a && ib < ia)) sv[t] = b, si[t] = ib;
    }
    __syncthreads();
  }
  if (t == 0) {
    out[0] = __float_as_uint(sv[0]);
    out[1] = lo + (si[0] == 0xffffffffu ? 0u : si[0]);
  }
  if constexpr (sums) {
    // out[2]: the slice's sum of exp(logit - its max), fp32 bits, from which rank 0 reads the pick's chance
    const float m = sv[0];
    float s = 0.0f;
    for (int64_t i = t; i < n; i += 1024)
      s += __expf(__int_as_float(static_cast<int>(static_cast<unsigned>(logits[i]) << 16)) - m);
    __syncthreads();
    sv[t] = s;
    __syncthreads();
    for (int h = 512; h > 0; h >>= 1) {
      if (t < h) sv[t] += sv[t + h];
      __syncthreads();
    }
    if (t == 0) out[2] = __float_as_uint(sv[0]);
  }
}
