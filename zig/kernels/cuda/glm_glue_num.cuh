// TF_GLM_ABSORB_MMA: head-batched BF16 GEMMs with FP32 MMA sums instead of the SIMT reduction order.
#pragma once

#include "glm_glue_router.cuh"
#include "common/cp_async.cuh"

namespace glm_num {

__device__ __forceinline__ void cp16z(void* dst, const void* src, bool ok) { tf_common::cp16z(dst, src, ok); }
__device__ __forceinline__ void commit() { tf_common::cp_commit(); }
template <int N>
__device__ __forceinline__ void wait() { tf_common::cp_wait<N>(); }
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) { qmm_frag::ldmatrix4(r, tf_common::smem(p)); }
__device__ __forceinline__ void ldm4t(uint32_t* r, const void* p) { qmm_frag::ldmatrix4t(r, tf_common::smem(p)); }
__device__ __forceinline__ void mma_bf16(float* d, const uint32_t* a, uint32_t b0, uint32_t b1) {
  const uint32_t b[2] = {b0, b1};
  glm_router::mma16816(d, a, b);
}

// B_KN reads [heads][K][N], otherwise [heads][N][K]; 128x128 blocks, K32 steps, padded cp.async stages.
constexpr int LBM = 128, LBN = 128, LBK = 32, LAP = LBK + 8, LBP = LBN + 8;
template <bool B_KN>
struct LatTile {
  static constexpr int A = LBM * LAP, B = B_KN ? LBK * LBP : LBN * LAP, STAGE = (A + B) * 2;
};

template <bool B_KN, int STAGES, bool BAD = false>
__device__ __forceinline__ void latent_mma(const __nv_bfloat16* __restrict__ x, int64_t xr,
                                           const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                           int64_t orow, int64_t rows, int N, int K) {
  using T = LatTile<B_KN>;
  constexpr int MT = 4, NT = 4;
  extern __shared__ __align__(16) unsigned char lat_smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp >> 2, wn = warp & 3;
  const int64_t n0 = (int64_t)blockIdx.x * LBN, m0 = (int64_t)blockIdx.y * LBM, h = blockIdx.z;
  const __nv_bfloat16* xa = x + h * K;
  const __nv_bfloat16* wb = w + (BAD ? (h + 1) % gridDim.z : h) * (int64_t)K * N;
  auto sa = [&](int st) { return reinterpret_cast<__nv_bfloat16*>(lat_smem + st * T::STAGE); };
  auto sb = [&](int st) { return reinterpret_cast<__nv_bfloat16*>(lat_smem + st * T::STAGE) + T::A; };
  auto load = [&](int st, int kt) {
    const int k0 = kt * LBK;
    __nv_bfloat16* a = sa(st);
    __nv_bfloat16* b = sb(st);
#pragma unroll
    for (int c = tid; c < LBM * (LBK / 8); c += 256) {  // 2 a thread
      const int r = c >> 2, ch = c & 3;
      const int64_t row = m0 + r;
      cp16z(a + r * LAP + ch * 8, xa + (row < rows ? row : 0) * xr + k0 + ch * 8, row < rows);
    }
    if constexpr (B_KN) {
#pragma unroll
      for (int c = tid; c < LBK * (LBN / 8); c += 256) {
        const int k = c >> 4, ch = c & 15;
        cp16z(b + k * LBP + ch * 8, wb + (int64_t)(k0 + k) * N + n0 + ch * 8, true);
      }
    } else {
#pragma unroll
      for (int c = tid; c < LBN * (LBK / 8); c += 256) {
        const int n = c >> 2, ch = c & 3;
        cp16z(b + n * LAP + ch * 8, wb + (n0 + n) * K + k0 + ch * 8, true);
      }
    }
  };
  float acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
  const int steps = K / LBK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < steps) load(s, s);
    commit();
  }
  for (int kt = 0; kt < steps; ++kt) {
    wait<STAGES - 2>();
    __syncthreads();
    if (kt + STAGES - 1 < steps) load((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
    commit();
    const __nv_bfloat16* a = sa(kt % STAGES);
    const __nv_bfloat16* b = sb(kt % STAGES);
#pragma unroll
    for (int ks = 0; ks < LBK / 16; ++ks) {
      uint32_t af[MT][4], bfr[NT][2];
#pragma unroll
      for (int i = 0; i < MT; ++i) ldm4(af[i], a + (wm * 64 + i * 16 + (lane & 15)) * LAP + ks * 16 + (lane >> 4) * 8);
#pragma unroll
      for (int j = 0; j < NT; j += 2) {
        uint32_t r[4];
        if constexpr (B_KN)
          ldm4t(r, b + (ks * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LBP + wn * 32 + j * 8 + (lane >> 4) * 8);
        else
          ldm4(r, b + (wn * 32 + j * 8 + (lane & 7) + (lane >> 4) * 8) * LAP + ks * 16 + ((lane >> 3) & 1) * 8);
        bfr[j][0] = r[0], bfr[j][1] = r[1], bfr[j + 1][0] = r[2], bfr[j + 1][1] = r[3];
      }
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) mma_bf16(acc[i][j], af[i], bfr[j][0], bfr[j][1]);
    }
  }
  wait<0>();
  const int g = lane >> 2, t = lane & 3;
  __nv_bfloat16* o = out + h * N;
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int64_t row = m0 + wm * 64 + i * 16 + g + hh * 8;
      if (row >= rows) continue;
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const int64_t col = n0 + wn * 32 + j * 8 + t * 2;
        *reinterpret_cast<__nv_bfloat162*>(o + row * orow + col) =
            __floats2bfloat162_rn(acc[i][j][2 * hh], acc[i][j][2 * hh + 1]);
      }
    }
}

}  // namespace glm_num

// absorb_q for prompt rows: qa [rows][heads][512] = q [rows][heads][256] . wk [heads][256][512] (B as [K][N]).
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_latent_mma_kn(const __nv_bfloat16* x, int64_t xr, const __nv_bfloat16* w, __nv_bfloat16* out, int64_t orow,
                         int64_t rows, int n, int k) {
  glm_num::latent_mma<true, 3>(x, xr, w, out, orow, rows, n, k);
}

// expand_v for prompt rows: out [rows][heads][256] = o_lat [rows][heads][512] . wv [heads][256][512] (B as [N][K]).
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_latent_mma_nk(const __nv_bfloat16* x, int64_t xr, const __nv_bfloat16* w, __nv_bfloat16* out, int64_t orow,
                         int64_t rows, int n, int k) {
  glm_num::latent_mma<false, 3>(x, xr, w, out, orow, rows, n, k);
}
