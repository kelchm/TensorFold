// TF_GLM_ROUTER_FUSE: aligned 64x32 router tiles adding all K slices in order inside one launch (slice_sum's bytes).
#pragma once

#include "glm_pdl.cuh"
#include "glm_glue_router.cuh"
#include "common/qmm_frag.cuh"

namespace glm_router_tile {
// 64 x 32 of 4 warps, K 64 in two stages, all ks slices added in order; CHAIN: one unbroken K chain (a control).
template <bool CHAIN = false>
__device__ __forceinline__ void tiles(const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int64_t rows,
                                      int64_t x_stride, int64_t d, int64_t ne, int ks) {
  constexpr int BM = 64, BE = 32, KS = 64, LD = KS + 8, A = BM * LD, B = BE * LD, STAGE = (A + B) * 2, RPI = 16;
  __shared__ __align__(16) __nv_bfloat16 sm[2][A + B];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, q = lane & 3;
  const int64_t m0 = (int64_t)blockIdx.y * BM, e0 = (int64_t)blockIdx.x * BE, span = d / ks;
  const int per = static_cast<int>(span / KS), steps = per * ks, lr = tid / 8, lc = (tid % 8) * 8;
  const __nv_bfloat16* ga[BM / RPI];
  const __nv_bfloat16* gb[BE / RPI];
  unsigned oka = 0, okb = 0;
#pragma unroll
  for (int i = 0; i < BM / RPI; ++i) {
    const int64_t row = m0 + lr + i * RPI;
    oka |= (row < rows ? 1u : 0u) << i;
    ga[i] = x + (row < rows ? row : 0) * x_stride + lc;
  }
#pragma unroll
  for (int i = 0; i < BE / RPI; ++i) {
    const int64_t e = e0 + lr + i * RPI;
    okb |= (e < ne ? 1u : 0u) << i;
    gb[i] = w + (e < ne ? e : 0) * d + lc;
  }
  const unsigned sbase = static_cast<unsigned>(__cvta_generic_to_shared(sm)), soff = (lr * LD + lc) * 2;
  auto load = [&](int st, int kt) {
    const unsigned a = sbase + st * STAGE + soff;
#pragma unroll
    for (int i = 0; i < BM / RPI; ++i) tf_common::cp16z(a + i * RPI * LD * 2, ga[i] + kt * KS, (oka >> i) & 1);
#pragma unroll
    for (int i = 0; i < BE / RPI; ++i) tf_common::cp16z(a + A * 2 + i * RPI * LD * 2, gb[i] + kt * KS, (okb >> i) & 1);
  };
  const unsigned fa = sbase + ((warp * 16 + (lane & 15)) * LD + (lane >> 4) * 8) * 2;
  const unsigned fb = sbase + A * 2 + (((lane >> 4) * 8 + (lane & 7)) * LD + ((lane >> 3) & 1) * 8) * 2;
  float c[BE / 8][4] = {}, t[BE / 8][4] = {};
  int done = 0;        // K steps of the current slice so far
  bool first = true;   // whether it is slice 0
  load(0, 0);
  tf_common::cp_commit();
  for (int kt = 0; kt < steps; ++kt) {
    if (kt + 1 < steps) load((kt + 1) & 1, kt + 1);
    tf_common::cp_commit();
    tf_common::cp_wait<1>();
    __syncthreads();
    const unsigned st = (kt & 1) * STAGE;
#pragma unroll
    for (int k16 = 0; k16 < KS; k16 += 16) {
      uint32_t a[4], b[BE / 8][2];
      qmm_frag::ldmatrix4(a, fa + st + k16 * 2);
#pragma unroll
      for (int j = 0; j < BE / 8; j += 2) {
        uint32_t r[4];
        qmm_frag::ldmatrix4(r, fb + st + (j * 8 * LD + k16) * 2);
        b[j][0] = r[0], b[j][1] = r[1], b[j + 1][0] = r[2], b[j + 1][1] = r[3];
      }
#pragma unroll
      for (int j = 0; j < BE / 8; ++j) glm_router::mma16816(c[j], a, b[j]);
    }
    if constexpr (!CHAIN) {
      if (++done == per) {  // a slice done: slice 0 as is, then t + c (slice_sum's order); the next chain from zero
#pragma unroll
        for (int j = 0; j < BE / 8; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) t[j][e] = first ? c[j][e] : t[j][e] + c[j][e], c[j][e] = 0.0f;
        done = 0, first = false;
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < BE / 8; ++j)
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const int64_t r = m0 + warp * 16 + g + h * 8, col = e0 + j * 8 + q * 2 + jj;
        if (r < rows && col < ne) out[r * ne + col] = CHAIN ? c[j][h * 2 + jj] : t[j][h * 2 + jj];
      }
}
}  // namespace glm_router_tile

// TF_GLM_ROUTER_FUSE: tf_glm_router_tile then tf_glm_slice_sum's bytes in out [m][e], no slices in memory.
extern "C" __global__ void __launch_bounds__(128)
    tf_glm_router_fused(const __nv_bfloat16* x, const __nv_bfloat16* w, float* out, int64_t rows, int64_t x_stride,
                        int64_t d, int64_t ne, int ks) {
  glm_pdl::enter();
  glm_router_tile::tiles<false>(x, w, out, rows, x_stride, d, ne, ks);
}
