// TF_GLM_HC_FUSE: glue._hc_post, then the next site's glue._hc_partial of the streams it wrote, x read once.
#pragma once

#include "glm_glue_rows.cuh"

namespace glm_hc_fuse {

// Block (4 rows, quarter q of each stream), 512 threads: hc_post's ops, then 4 hc_partial blocks' ops on its values.
template <bool ORDER>  // the warps' squares as hc_partial adds them, (s0 + s2) + (s1 + s3); false: the control's
__device__ __forceinline__ void post_partial(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g,
                                             const float* post, const float* comb, const __nv_bfloat16* fn,
                                             float* part, int64_t rs, int world, int64_t rows) {
  constexpr int R = 4, S = 4, D = 4096, Q = 1024, NB = 16;  // rows a block, streams, width, a K block, K blocks
  constexpr int64_t W = S * D;
  __shared__ __align__(16) __nv_bfloat16 xs[R][S][Q];  // the bf16 values hc_partial would load back
  __shared__ float smem[S][R][4];
  const int64_t r0 = (int64_t)blockIdx.x * R;
  const int q = blockIdx.y, tid = threadIdx.x;
  {  // tf_glm_hc_post, a thread a row's 8 columns of every stream: it reads x before it writes them (x may be xout)
    const int rq = tid >> 7, at = (tid & 127) * 8;
    const int64_t r = r0 + rq, c = q * Q + at;
    float branch[8], xv[4][8];
    if (r < rows) {
      glm_glue::ranks8(g, rs, r * D + c, world, branch);
      for (int t = 0; t < 4; ++t) glm_glue::ld8(x + r * W + t * D + c, xv[t]);
    }
    for (int s = 0; s < S; ++s) {
      uint4 u = make_uint4(0, 0, 0, 0);  // hc_partial loads rows past `rows` as zeros
      if (r < rows) {
        const float* cm = comb + r * 16 + s;
        const float c0 = cm[0], c1 = cm[4], c2 = cm[8], c3 = cm[12], p = post[r * 4 + s];
        __nv_bfloat16* h = reinterpret_cast<__nv_bfloat16*>(&u);
        for (int j = 0; j < 8; ++j) {
          float m = __fmul_rn(xv[1][j], c1);
          m = __fmaf_rn(xv[0][j], c0, m);
          m = __fmaf_rn(xv[2][j], c2, m);
          m = __fmaf_rn(xv[3][j], c3, m);
          h[j] = __float2bfloat16_rn(__fmaf_rn(glm_triton::rbf(branch[j]), p, m));
        }
        *reinterpret_cast<uint4*>(xout + r * W + s * D + c) = u;
      }
      *reinterpret_cast<uint4*>(&xs[rq][s][at]) = u;
    }
  }
  __syncthreads();
  // 128-thread group s is tf_glm_hc_partial's block (4 rows, K block b = 4 s + q): x[row][b * Q + i] is xs[row][s][i]
  const int s = tid >> 7, t = tid & 127, lane = t & 31, warp = t >> 5, c = t & 15, m0 = (t >> 4) & 7;
  const int64_t b = s * 4 + q;
  float acc[R][3] = {}, ss[R] = {};
  for (int step = 0; step < Q / 128; ++step) {
    const int at = step * 128;
    float xv[R][8];
    for (int rq = 0; rq < R; ++rq) {
      const uint4 u = *reinterpret_cast<const uint4*>(&xs[rq][s][at + c * 8]);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[rq][j] = glm_triton::bf(h[j]);
    }
    for (int i = 0; i < 3; ++i) {
      const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + (int64_t)(m0 + 8 * i) * W + b * Q + at + c * 8));
      const __nv_bfloat16* hw = reinterpret_cast<const __nv_bfloat16*>(&u);
      float w[8];
      for (int j = 0; j < 8; ++j) w[j] = glm_triton::bf(hw[j]);
      for (int rq = 0; rq < R; ++rq) {
        float d = __fmul_rn(xv[rq][1], w[1]);
        d = __fmaf_rn(xv[rq][0], w[0], d);
        for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[rq][j], w[j], d);
        for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
        acc[rq][i] = acc[rq][i] + d;
      }
    }
    for (int rq = 0; rq < R; ++rq) {
      const float xt = glm_triton::bf(xs[rq][s][at + t]);
      ss[rq] = __fmaf_rn(xt, xt, ss[rq]);
    }
  }
  for (int rq = 0; rq < R; ++rq) {
    for (int o = 16; o > 0; o >>= 1) ss[rq] += __shfl_xor_sync(0xffffffffu, ss[rq], o);
    if (lane == 0) smem[s][rq][warp] = ss[rq];
  }
  __syncthreads();
  for (int rq = 0; rq < R; ++rq) {
    if (r0 + rq >= rows) break;
    float* out = part + ((r0 + rq) * NB + b) * 32;
    if (c == 0)
      for (int i = 0; i < 3; ++i) out[m0 + 8 * i] = acc[rq][i];
    const float* w4 = smem[s][rq];
    if (t == 0) out[24] = ORDER ? (w4[0] + w4[2]) + (w4[1] + w4[3]) : (w4[0] + w4[1]) + (w4[2] + w4[3]);
  }
}

}  // namespace glm_hc_fuse

// x, xout [rows][4][4096] (may alias), g [world][rows][4096] `rs` apart, fn [24][16384]; part as tf_glm_hc_partial's.
extern "C" __global__ void __launch_bounds__(512)
    tf_glm_hc_post_partial(const __nv_bfloat16* x, __nv_bfloat16* xout, const float* g, const float* post,
                           const float* comb, const __nv_bfloat16* fn, float* part, int64_t rs, int world,
                           int64_t rows) {
  glm_hc_fuse::post_partial<true>(x, xout, g, post, comb, fn, part, rs, world, rows);
}
