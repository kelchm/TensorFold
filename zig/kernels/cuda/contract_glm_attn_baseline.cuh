// Frozen e6f522d05 MLA loop: byte oracle for the FP8 tile-load replacement; BAD swaps each code pair.
#pragma once

#include "glm_glue_router.cuh"
#include <cuda_fp8.h>

namespace glm_attn_old {

constexpr int LW = 512, KT = 32, CH = 512, KPAD = LW + 8;

// p fp32 summed as Triton's tile does: a thread's two columns, the quad (xor 2, 1), then warps (w0 + w2) + (w1 + w3).
__device__ __forceinline__ float quad_sum(float v) {
  v = v + __shfl_xor_sync(0xffffffffu, v, 2);
  return v + __shfl_xor_sync(0xffffffffu, v, 1);
}
__device__ __forceinline__ float quad_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ void cp16(void* dst, const void* src, bool ok) {
  tf_common::cp16z(dst, src, ok);
}
__device__ __forceinline__ void commit() { tf_common::cp_commit(); }
template <int N>
__device__ __forceinline__ void wait() { tf_common::cp_wait<N>(); }
template <bool TRANS>
__device__ __forceinline__ void ldm4(uint32_t* r, const void* p) {
  const unsigned a = tf_common::smem(p);
  if (TRANS) qmm_frag::ldmatrix4t(r, a);
  else qmm_frag::ldmatrix4(r, a);
}

// A 512-wide bf16 row's 16-byte unit u in shared memory, swizzled by the row so ldmatrix's eight rows hit eight banks.
__device__ __forceinline__ int wide(int row, int u) { return row * LW + ((u ^ (row & 7)) << 3); }
// p's [32][32] rows (64 bytes): unit u of row r.
__device__ __forceinline__ int narrow(int row, int u) { return row * KT + ((u ^ ((row >> 1) & 3)) << 3); }

// latent._dense_chunks/_sparse_chunks (sparse 0/1), block (row, HB heads, chunk), 8 warps: online softmax by 32 keys.
template <int HB>
constexpr int wide_smem() {
  return (HB * LW + 2 * KT * LW + HB * KT) * 2 + (HB * 4 + 2 * HB) * 4;
}

template <int HB, bool FP8 = false, bool BAD = false>
__device__ void chunks_wide(const __nv_bfloat16* qa, const __nv_bfloat16* lc, const int* pos, const int* tok,
                            const int* cnt, float* po, float* pm, float* pl, int64_t rows, int64_t heads, int64_t w,
                            float scale, int sparse, const int* pt, int64_t pstep, const float* scales = nullptr) {
  constexpr int MT = HB / 16;
  extern __shared__ __align__(16) unsigned char smem[];
  __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem);
  __nv_bfloat16* kvs = qs + HB * LW;
  __nv_bfloat16* ps = kvs + 2 * KT * LW;
  float* red = reinterpret_cast<float*>(ps + HB * KT);
  float* stat = red + HB * 4;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5, g = lane >> 2, q = lane & 3;
  const int kg = warp & 3, hh = warp >> 2;
  const int64_t r = blockIdx.x, h0 = (int64_t)blockIdx.y * HB, c = blockIdx.z;
  const int64_t n = sparse ? cnt[r] : 0, limit = sparse ? 0 : (int64_t)pos[0] + r, start = c * CH;
  const bool live = sparse ? start < n : start <= limit;
  // a dense chunk's 512 keys are one page, its rows from that page's; a sparse key's row comes as the token list has it
  const __nv_bfloat16* keys_at = !FP8 && pt && !sparse && live ? lc + ((int64_t)pt[c] * (pstep / 2) - start * LW) : lc;
  auto load_kv = [&](int tile) {
    __nv_bfloat16* dst = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    for (int idx = t; idx < KT * (LW / 8); idx += blockDim.x) {
      const int key = idx / (LW / 8), u = idx % (LW / 8);
      const int64_t ki = first + key;
      const bool ok = sparse ? ki < n : ki <= limit;
      const int64_t src = ok ? (sparse ? (int64_t)tok[r * w + ki] : ki) : 0;
      if constexpr (FP8) {
        const unsigned char* codes = reinterpret_cast<const unsigned char*>(lc);
        const float row_scale = ok ? scales[src] : 1.0f;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          uint16_t bits = ok ? (uint16_t)codes[src * LW + u * 8 + j] |
                                        ((uint16_t)codes[src * LW + u * 8 + j + 1] << 8) : 0;
          if constexpr (BAD) bits = (bits >> 8) | (bits << 8);
          const float2 v = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(bits, __NV_E4M3)));
          dst[wide(key, u) + j] = __float2bfloat16_rn(__fmul_rn(v.x, row_scale));
          dst[wide(key, u) + j + 1] = __float2bfloat16_rn(__fmul_rn(v.y, row_scale));
        }
      } else {
        cp16(dst + wide(key, u), keys_at + src * LW + u * 8, ok);
      }
    }
  };
  float m[MT][2], l[MT][2], o[MT][8][4];
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 2; ++e) m[i][e] = -INFINITY, l[i][e] = 0.0f;
  for (int i = 0; i < MT; ++i)
    for (int j = 0; j < 8; ++j)
      for (int e = 0; e < 4; ++e) o[i][j][e] = 0.0f;
  constexpr int tiles = CH / KT;
  if (live) {
    for (int idx = t; idx < HB * (LW / 8); idx += blockDim.x) {
      const int hd = idx / (LW / 8), u = idx % (LW / 8);
      const bool ok = h0 + hd < heads;
      cp16(qs + wide(hd, u), qa + (r * heads + (ok ? h0 + hd : 0)) * LW + u * 8, ok);
    }
    load_kv(0);
    commit();
    load_kv(1);
    commit();
  }
  for (int tile = 0; live && tile < tiles; ++tile) {
    if constexpr (FP8) {
      wait<0>();
    } else {
      if (tile + 1 < tiles)
        wait<1>();
      else
        wait<0>();
    }
    __syncthreads();
    const __nv_bfloat16* kv = kvs + (tile & 1) * KT * LW;
    const int64_t first = start + tile * KT;
    float sc[4];
    if (hh < MT) {
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 4
      for (int kk = 0; kk < LW; kk += 32) {
        uint32_t a0[4], a1[4], b[4];
        ldm4<false>(a0, qs + wide(hh * 16 + (lane & 15), (kk >> 3) + (lane >> 4)));
        ldm4<false>(a1, qs + wide(hh * 16 + (lane & 15), ((kk + 16) >> 3) + (lane >> 4)));
        ldm4<false>(b, kv + wide(kg * 8 + (lane & 7), (kk >> 3) + (lane >> 3)));
        glm_router::mma16816(acc, a0, b);
        glm_router::mma16816(acc, a1, b + 2);
      }
      for (int e = 0; e < 2; ++e) {
        const int64_t ki = first + kg * 8 + q * 2 + e;
        const bool ok = sparse ? ki < n : ki <= limit;
        sc[e] = ok ? __fmul_rn(acc[e], scale) : -INFINITY;
        sc[e + 2] = ok ? __fmul_rn(acc[e + 2], scale) : -INFINITY;
      }
      for (int hf = 0; hf < 2; ++hf) {
        const float mx = quad_max(fmaxf(sc[hf * 2], sc[hf * 2 + 1]));
        if (q == 0) red[(hh * 16 + hf * 8 + g) * 4 + kg] = mx;
      }
    }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
      v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
      if ((t & 3) == 0) stat[t / 4] = v;
    }
    __syncthreads();
    float alpha[MT][2];
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) {
        const float tm = stat[i * 16 + hf * 8 + g];
        const bool active = tm != -INFINITY;
        const float nm = active ? fmaxf(m[i][hf], tm) : m[i][hf];
        alpha[i][hf] = active ? (m[i][hf] == -INFINITY ? 0.0f : glm_triton::exp_approx(m[i][hf] - nm)) : 1.0f;
        if (i == hh) {
          float p[2];
          for (int e = 0; e < 2; ++e) {
            const int64_t ki = first + kg * 8 + q * 2 + e;
            const bool ok = sparse ? ki < n : ki <= limit;
            p[e] = active && ok ? glm_triton::exp_approx(sc[hf * 2 + e] - nm) : 0.0f;
          }
          const int hr = i * 16 + hf * 8 + g, col = kg * 8 + q * 2;
          *reinterpret_cast<__nv_bfloat162*>(ps + narrow(hr, col >> 3) + (col & 7)) = __floats2bfloat162_rn(p[0], p[1]);
          const float s = quad_sum(p[0] + p[1]);
          if (q == 0) red[hr * 4 + kg] = s;
        }
        m[i][hf] = nm;
      }
    __syncthreads();
    if (t < HB * 4) {
      float v = red[t];
      v = v + __shfl_xor_sync(0xffffffffu, v, 2);
      v = v + __shfl_xor_sync(0xffffffffu, v, 1);
      if ((t & 3) == 0) stat[HB + t / 4] = v;
    }
    for (int i = 0; i < MT; ++i)
      for (int j = 0; j < 8; ++j)
        for (int e = 0; e < 4; ++e) o[i][j][e] = __fmul_rn(o[i][j][e], alpha[i][e / 2]);
#pragma unroll
    for (int k16 = 0; k16 < KT; k16 += 16) {
      uint32_t a[MT][4];
      for (int i = 0; i < MT; ++i) ldm4<false>(a[i], ps + narrow(i * 16 + (lane & 15), (k16 >> 3) + (lane >> 4)));
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        uint32_t b[4];
        ldm4<true>(b, kv + wide(k16 + (lane & 15), ((warp * 64 + j * 8) >> 3) + (lane >> 4)));
        for (int i = 0; i < MT; ++i) {
          glm_router::mma16816(o[i][j], a[i], b);
          glm_router::mma16816(o[i][j + 1], a[i], b + 2);
        }
      }
    }
    __syncthreads();
    for (int i = 0; i < MT; ++i)
      for (int hf = 0; hf < 2; ++hf) l[i][hf] = __fmaf_rn(l[i][hf], alpha[i][hf], stat[HB + i * 16 + hf * 8 + g]);
    if (tile + 2 < tiles) load_kv(tile + 2);
    commit();
  }
  for (int i = 0; i < MT; ++i)
    for (int e = 0; e < 4; ++e) {
      const int64_t hd = h0 + i * 16 + g + (e / 2) * 8;
      if (hd >= heads) continue;
      const int64_t base = (c * rows + r) * heads + hd;
      for (int j = 0; j < 8; ++j) po[base * LW + warp * 64 + j * 8 + q * 2 + e % 2] = o[i][j][e];
      if (warp == 0 && q == 0 && e % 2 == 0) {
        pm[base] = m[i][e / 2];
        pl[base] = l[i][e / 2];
      }
    }
}

}  // namespace glm_attn_old
extern "C" __global__ void __launch_bounds__(256)
tf_ctl_glm_chunks_16_fp8_old(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
       const int* tok, const int* cnt, float* po, float* pm, float* pl,
       int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn_old::chunks_wide<16, true, false>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
      po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}

extern "C" __global__ void __launch_bounds__(256)
tf_ctl_glm_chunks_16_fp8_bad(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
       const int* tok, const int* cnt, float* po, float* pm, float* pl,
       int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn_old::chunks_wide<16, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
      po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}

extern "C" __global__ void __launch_bounds__(256)
tf_ctl_glm_chunks_32_fp8_old(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
       const int* tok, const int* cnt, float* po, float* pm, float* pl,
       int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn_old::chunks_wide<32, true, false>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
      po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}

extern "C" __global__ void __launch_bounds__(256)
tf_ctl_glm_chunks_32_fp8_bad(const __nv_bfloat16* qa, const unsigned char* lc, const float* scales, const int* pos,
       const int* tok, const int* cnt, float* po, float* pm, float* pl,
       int64_t rows, int64_t heads, int64_t w, float scale, int sparse) {
  glm_attn_old::chunks_wide<32, true, true>(qa, reinterpret_cast<const __nv_bfloat16*>(lc), pos, tok, cnt,
      po, pm, pl, rows, heads, w, scale, sparse, nullptr, 0, scales);
}
