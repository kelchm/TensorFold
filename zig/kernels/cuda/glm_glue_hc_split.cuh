// hc_pre's decode kernels over more threads (Set.glue's hc_split, hc_part): tf_glm_hc_finish's and _partial8's bytes.
#pragma once

#include "glm_glue_hc.cuh"

namespace glm_hcp {
// tf_glm_hc_partial8's bytes over more blocks: block (row, K block, z) of 128 M threads, M dots; z = 24 / M the squares

template <int M>
__device__ __forceinline__ void partial_split(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part,
                                              int64_t wide, int nb) {
  static_assert(24 % M == 0, "whole dots a block");
  pdl_wait();  // dependent launch: both roles read x, an earlier grid's rows, so every thread waits before any read
  pdl_trigger();
  __shared__ float dots[M][8];
  __shared__ float smem[4];
  const int64_t r = blockIdx.x, b = blockIdx.y, kb = wide / nb;
  const int steps = static_cast<int>(kb / 128);  // 8 at hcPre's shapes, as many as tf_glm_hc_partial8 holds
  float* out = part + (r * nb + b) * 32;
  const int t = threadIdx.x;
  if (blockIdx.z == 24 / M) {  // tf_glm_hc_partial8's group 0: the squares
    if (t < 128) {
      float ss = 0.0f;
      for (int s = 0; s < steps; ++s) {
        const float xt = bf(x[r * wide + b * kb + s * 128 + t]);
        ss = __fmaf_rn(xt, xt, ss);
      }
      for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
      if ((t & 31) == 0) smem[t >> 5] = ss;
    }
    __syncthreads();
    if (t == 0) out[24] = (smem[0] + smem[2]) + (smem[1] + smem[3]);
    return;
  }
  // each (dot, step) on a 16-lane half-warp with partial8's lanes, loads, fma chain, butterfly and step sum from 0.0f
  const int k = t >> 7, s = (t >> 4) & 7, c = t & 15;
  const int64_t m = (int64_t)blockIdx.z * M + k;
  const bool live = s < steps;  // a half-warp's own step: its warp-mate holds another, so every lane shuffles
  float d = 0.0f;
  if (live) {
    const int64_t base = b * kb + s * 128;
    float xv[8];
    {
      const uint4 u = *reinterpret_cast<const uint4*>(x + r * wide + base + c * 8);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&u);
      for (int j = 0; j < 8; ++j) xv[j] = bf(h[j]);
    }
    const uint4 u = __ldg(reinterpret_cast<const uint4*>(fn + m * wide + base + c * 8));
    const __nv_bfloat16* w = reinterpret_cast<const __nv_bfloat16*>(&u);
    d = __fmul_rn(xv[1], bf(w[1]));
    d = __fmaf_rn(xv[0], bf(w[0]), d);
    for (int j = 2; j < 8; ++j) d = __fmaf_rn(xv[j], bf(w[j]), d);
  }
  for (int o = 8; o > 0; o >>= 1) d = d + __shfl_xor_sync(0xffffffffu, d, o);
  if (live && c == 0) dots[k][s] = d;
  __syncthreads();
  if (t < M) {
    float acc = 0.0f;
    for (int st = 0; st < steps; ++st) acc = acc + dots[t][st];
    out[(int64_t)blockIdx.z * M + t] = acc;
  }
}
}  // namespace glm_hcp

// Grid (rows, 16, 13) of 256 threads: two dots a block (cuda_glue.hcPre, decode windows to 16 rows).
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_partial8_o2(const __nv_bfloat16* x, const __nv_bfloat16* fn, float* part, int64_t wide, int nb) {
  glm_hcp::partial_split<2>(x, fn, part, wide, nb);
}

namespace glm_hc {

// finish's collapse of one thread's columns: its 16 collapsed values and their sum of squares, finish's ops in order.
__device__ __forceinline__ void collapse(const uint4 (&xu)[2][4], float p0, float p1, float p2, float p3,
                                         float (&c)[16], float& acc) {
  for (int k = 0; k < 16; ++k) {
    const __nv_bfloat16* h0 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][0]);
    const __nv_bfloat16* h1 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][1]);
    const __nv_bfloat16* h2 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][2]);
    const __nv_bfloat16* h3 = reinterpret_cast<const __nv_bfloat16*>(&xu[k / 8][3]);
    const float x0 = bf(h0[k % 8]), x1 = bf(h1[k % 8]), x2 = bf(h2[k % 8]), x3 = bf(h3[k % 8]);
    float m = k % 2 == 0 ? __fmaf_rn(p1, x1, __fmul_rn(p0, x0)) : __fmaf_rn(p0, x0, __fmul_rn(p1, x1));
    m = __fmaf_rn(p2, x2, m);
    c[k] = rbf(__fmaf_rn(p3, x3, m));
  }
  acc = __fmaf_rn(c[0], c[0], __fmul_rn(c[1], c[1]));
  for (int k = 2; k < 16; ++k) acc = k % 2 == 0 ? __fmaf_rn(c[k], c[k], acc) : fadd(fmul(c[k], c[k]), acc);
}

// finish's normed output and 64-sums of column thread `ct`'s values (the 8-lane groups of its warp).
template <typename W>
__device__ __forceinline__ void emit(const float (&c)[16], const Weights<W>& wu, float rinv, __nv_bfloat16* out,
                                     float* xs, int64_t r, int ct) {
  constexpr int D = 4096;
  for (int rep = 0; rep < 2; ++rep) {
    float y[8];
    uint4 ou;
    __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ou);
    for (int j = 0; j < 8; ++j) {
      const __nv_bfloat16 o = __float2bfloat16_rn(__fmul_rn(wu.at(rep, j), rbf(__fmul_rn(rinv, c[rep * 8 + j]))));
      oh[j] = o;
      y[j] = bf(o);
    }
    *reinterpret_cast<uint4*>(out + r * D + rep * 2048 + ct * 8) = ou;
    float g = y[0] + y[1];
    for (int j = 2; j < 8; ++j) g = y[j] + g;
    for (int o = 4; o > 0; o >>= 1) g = g + __shfl_xor_sync(0xffffffffu, g, o);
    if ((ct & 7) == 0) xs[r * (D / 64) + rep * 32 + ct / 8] = g;
  }
}

__device__ __forceinline__ void bar_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// finish's bytes with warp 0's Sinkhorn beside the collapse of warps 1..7 (warp 1 also takes warp 0's columns)
template <typename W, int NB>
__device__ __forceinline__ void finish_split(const __nv_bfloat16* x, const float* part, const float* base,
                                             const float* scale, const W* nw, __nv_bfloat16* out, float* xs,
                                             float* post, float* comb, float eps, float hc_eps, int iters) {
  constexpr int D = 4096, S = 4;
  __shared__ float smem[8];
  __shared__ float pre[4];
  __shared__ float tot;
  const int64_t r = blockIdx.x;
  const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
  const __nv_bfloat16* row = x + r * S * D;
  pdl_wait();  // dependent launch: warp 0 reads the partials, warps 1..7 x, both earlier grids' rows
  pdl_trigger();
  if (warp == 0) {
    const float* p = part + r * NB * 32;
    float mix = 0.0f, ss = 0.0f;
#pragma unroll
    for (int b = 0; b < NB; ++b) {
      mix = mix + p[b * 32 + lane];
      ss = ss + p[b * 32 + 24];
    }
    const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(ss, (float)(S * D))));
    mix = __fmul_rn(mix, rinv);
    const int k = lane & 3, idx = lane & 15;
    const float bpre = base[k], bpost = base[4 + k], bcomb = base[8 + idx];
    const float mpre = __shfl_sync(0xffffffffu, mix, k), mpost = __shfl_sync(0xffffffffu, mix, 4 + k);
    const float mcomb = __shfl_sync(0xffffffffu, mix, 8 + idx);
    const float pr = hc_eps + glm_hc::sigmoid(__fmaf_rn(mpre, scale[0], bpre));
    const float sp = glm_hc::sigmoid(__fmaf_rn(mpost, scale[1], bpost));
    if (lane < 4) {
      post[r * S + lane] = sp + sp;
      pre[lane] = pr;
    }
    bar_arrive(1, 256);  // the pre weights: warps 1..7 collapse while this warp runs Sinkhorn
    const float cl = __fmaf_rn(mcomb, scale[2], bcomb);
    float cmax = fmaxf(cl, __shfl_xor_sync(0xffffffffu, cl, 2));
    cmax = fmaxf(cmax, __shfl_xor_sync(0xffffffffu, cmax, 1));
    const float ce = exp_approx(cl - cmax);
    float cm = fadd(hc_eps, div_full(ce, glm_hc::row4(ce)));
    cm = div_full(cm, hc_eps + glm_hc::col4(cm));
    for (int it = 1; it < iters; ++it) {
      cm = div_full(cm, hc_eps + glm_hc::row4(cm));
      cm = div_full(cm, hc_eps + glm_hc::col4(cm));
    }
    if (lane < 16) comb[r * 16 + lane] = cm;
    return;
  }
  // warps 1..7: this thread's columns, and warp 1 warp 0's (thread `lane`'s)
  uint4 xu[2][S], xz[2][S];
  Weights<W> wu, wz;
  for (int rep = 0; rep < 2; ++rep) {
    for (int st = 0; st < S; ++st) xu[rep][st] = *reinterpret_cast<const uint4*>(row + st * D + rep * 2048 + t * 8);
    wu.load(nw, rep, t);
    if (warp == 1) {
      for (int st = 0; st < S; ++st)
        xz[rep][st] = *reinterpret_cast<const uint4*>(row + st * D + rep * 2048 + lane * 8);
      wz.load(nw, rep, lane);
    }
  }
  bar_sync(1, 256);
  const float p0 = pre[0], p1 = pre[1], p2 = pre[2], p3 = pre[3];
  float c[16], cz[16], acc, accz = 0.0f;
  collapse(xu, p0, p1, p2, p3, c, acc);
  if (warp == 1) collapse(xz, p0, p1, p2, p3, cz, accz);
  // block_sum<8>'s two butterflies on the same values in the same lanes: each warp's, then lanes 0..7's
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) smem[warp] = acc;
  if (warp == 1) {
    for (int o = 16; o > 0; o >>= 1) accz += __shfl_xor_sync(0xffffffffu, accz, o);
    if (lane == 0) smem[0] = accz;
  }
  bar_sync(2, 224);
  if (warp == 1) {
    float v = lane < 8 ? smem[lane] : 0.0f;
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) tot = v;
  }
  bar_sync(2, 224);
  const float rinv = div_full(1.0f, sqrt_approx(eps + div_full(tot, (float)D)));
  emit(c, wu, rinv, out, xs, r, t);
  if (warp == 1) emit(cz, wz, rinv, out, xs, r, lane);
}

}  // namespace glm_hc

// tf_glm_hc_finish with the split schedule (glm_hc::finish_split): the same outputs bit for bit; 16 K blocks.
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish_split(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                           const __nv_bfloat16* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                           float hc_eps, int nb, int iters) {
  if (nb != 16) __trap();  // the host launches it with glue.hc_blocks only
  glm_hc::finish_split<__nv_bfloat16, 16>(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, iters);
}

// The same with FP4-Local's float32 norm weight (tf_glm_hc_finish_f32w's bytes).
extern "C" __global__ void __launch_bounds__(256)
    tf_glm_hc_finish_f32w_split(const __nv_bfloat16* x, const float* part, const float* base, const float* scale,
                                const float* nw, __nv_bfloat16* out, float* xs, float* post, float* comb, float eps,
                                float hc_eps, int nb, int iters) {
  if (nb != 16) __trap();
  glm_hc::finish_split<float, 16>(x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, iters);
}
