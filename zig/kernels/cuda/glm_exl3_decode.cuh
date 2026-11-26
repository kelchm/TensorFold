// GLM TP2's four-bit routed experts for short verifier windows.
// Adapted from TensorFold 0.6.0 families/glm5_next/cuda/exl3.cu, Apache-2.0, Copyright 2026 TensorFold contributors.
// EXL3 format after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp. The non-caching vector load and staging
// follow jayleaton/glm53-tensorfold-spark patch 0580, Apache-2.0, Copyright 2026 Jay Leaton.
// This specialization keeps only production decode order 0, NC depth 1, gate/up split 4, down split 1.
// GLM's loader preserves the native bases of its contiguous uniform-width expert banks.
#pragma once
namespace tf_glm_exl3_decode {
using tf_exl3::mma16816;
using tf_exl3::fwht128;
using tf_exl3::HAD_SCALE;

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// This lane's eight values of a 4-bit tile (word = tile[lane]) as the B fragments of its two n8 halves.
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    uint32_t s = __funnelshift_r(w, p, 20);
    b0[0] = mcg2((s >> 8) & 0xffffu, (s >> 4) & 0xffffu);
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2((w >> 12) & 0xffffu, (w >> 8) & 0xffffu);
    b1[1] = mcg2((w >> 4) & 0xffffu, w & 0xffffu);
}

__device__ __forceinline__ uint32_t load_pair(const half* x, bool ok) {
    return ok ? *reinterpret_cast<const uint32_t*>(x) : 0u;
}

// 16 bytes of trellis words, read once a window (no L1 allocation); asm volatile keeps it where it stands relative
// to the (volatile) mma, so the ring depth is what the code says (patches/0580 of jayleaton/glm53-tensorfold-spark).
__device__ __forceinline__ uint4 ldg_nc_v4(const uint32_t* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

template <bool DOWN, int slots = 9>
__device__ __forceinline__ void projection(const half* __restrict__ X, const uint32_t* __restrict__ T0,
    const uint32_t* __restrict__ T1, const int* __restrict__ items, const int* __restrict__ counts,
    const int* __restrict__ members, float* __restrict__ Z, int P, int E, const half* __restrict__ svh_d,
    float* __restrict__ y) {
    constexpr int W = 4, NT = 8, STEPS = 16, SK = DOWN ? 1 : 4;
    constexpr int K = DOWN ? 1024 : 4096, N = DOWN ? 4096 : 1024;
    const int item = blockIdx.x, nblock = blockIdx.y, zz = blockIdx.z;
    const int total = counts[0];
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2];
    if (item >= total || e >= E) return;
    const int mat = zz / SK, split = zz % SK;
    const uint32_t* T = mat ? T1 : T0;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[16];
    __shared__ __align__(16) float red[W][16][NT * 16];
    constexpr int NV = NT / 4;                           // LD 1: 16-byte loads a lane a k step
    constexpr int PD = 1;
    uint4 rv[PD][NV];
    const uint32_t* tbase = T + (((size_t)e * KT + split * (KT / SK) + warp * STEPS) * NTILES + nblock * NT) * 32;
    const size_t kstep = (size_t)NTILES * 32;            // words from one k tile to the next
    {
#pragma unroll
        for (int d = 0; d < PD; ++d)
#pragma unroll
            for (int v = 0; v < NV; ++v) rv[d][v] = ldg_nc_v4(tbase + d * kstep + 128 * v + 4 * lane);
    }
    if (threadIdx.x < 16) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int r0 = rows_sh[g], r1 = rows_sh[g + 8];
    const int xr0 = r0 < 0 ? 0 : (!DOWN ? r0 / slots : r0), xr1 = r1 < 0 ? 0 : (!DOWN ? r1 / slots : r1);
    const half* x0 = X + (size_t)xr0 * K + 2 * t;
    const half* x1 = X + (size_t)xr1 * K + 2 * t;
    const int nt0 = nblock * NT;
    const int kt0 = split * (KT / SK) + warp * STEPS;

    float acc[NT][2][4];
#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][h][c] = 0.f;

    {
        uint32_t ra[PD][4];
#pragma unroll
        for (int d = 0; d < PD; ++d) {
            const int k = (kt0 + d) * 16;
            ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
            ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
        }
        uint32_t* stage = reinterpret_cast<uint32_t*>(&red[warp][0][0]);
        for (int kb = 0; kb < STEPS; kb += PD) {
#pragma unroll
            for (int d = 0; d < PD; ++d) {
                const int s = kb + d;
                const bool more = s + PD < STEPS;
                uint32_t a[4], words[NT];
#pragma unroll
                for (int c = 0; c < 4; ++c) a[c] = ra[d][c];
                __syncwarp();                            // every lane has read the previous step back
#pragma unroll
                for (int v = 0; v < NV; ++v) *reinterpret_cast<uint4*>(stage + 128 * v + 4 * lane) = rv[d][v];
                if (more) {                              // the slot's registers are free again
#pragma unroll
                    for (int v = 0; v < NV; ++v)
                        rv[d][v] = ldg_nc_v4(tbase + (size_t)(s + PD) * kstep + 128 * v + 4 * lane);
                    const int k = (kt0 + s + PD) * 16;
                    ra[d][0] = load_pair(x0 + k, r0 >= 0); ra[d][1] = load_pair(x1 + k, r1 >= 0);
                    ra[d][2] = load_pair(x0 + k + 8, r0 >= 0); ra[d][3] = load_pair(x1 + k + 8, r1 >= 0);
                }
                __syncwarp();
#pragma unroll
                for (int i = 0; i < NT; ++i) words[i] = stage[i * 32 + lane];
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(words[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        __syncwarp();                                    // the staging area becomes red[warp] below
    }

#pragma unroll
    for (int i = 0; i < NT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int col = i * 16 + h * 8 + 2 * t;
            red[warp][g][col] = acc[i][h][0];
            red[warp][g][col + 1] = acc[i][h][1];
            red[warp][g + 8][col] = acc[i][h][2];
            red[warp][g + 8][col + 1] = acc[i][h][3];
        }
    __syncthreads();

    if constexpr (DOWN) {
        // down_epilogue_kernel on this block's sums: v = 0 + (the warps' sums in warp order), rotated, times svh
        for (int i = warp; i < cnt; i += W) {
            const int p = rows_sh[i];
            const int n = nt0 * 16 + 4 * lane;
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = red[0][i][4 * lane + j];
#pragma unroll
                for (int w = 1; w < W; ++w) s += red[w][i][4 * lane + j];
                float z = 0.f;
                z += s;
                v[j] = z;
            }
            fwht128(v, lane);
            float* o = y + (size_t)p * N + n;
#pragma unroll
            for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(svh_d[(size_t)e * N + n + j]);
        }
        return;
    } else {
        constexpr int OUT = 16 * NT * 16 / (W * 32);
#pragma unroll
        for (int j = 0; j < OUT; ++j) {
            const int idx = threadIdx.x + j * W * 32;
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[row];
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            if (r >= 0) Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
}
}  // namespace tf_glm_exl3_decode

extern "C" __global__ void __launch_bounds__(128) tf_glm_exl3_decode_gateup(const half* x, const uint32_t* g,
        const uint32_t* u, const int* items, const int* counts, const int* members, float* z, int pairs, int experts) {
    tf_glm_exl3_decode::projection<false>(x, g, u, items, counts, members, z, pairs, experts, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) tf_glm_exl3_decode_gateup_flat(const half* x, const uint32_t* g,
        const uint32_t* u, const int* items, const int* counts, const int* members, float* z, int pairs, int experts) {
    tf_glm_exl3_decode::projection<false, 8>(x, g, u, items, counts, members, z, pairs, experts, nullptr, nullptr);
}
extern "C" __global__ void __launch_bounds__(128) tf_glm_exl3_decode_down(const half* x, const uint32_t* d,
        const int* items, const int* counts, const int* members, int pairs, int experts, const half* svh, float* out) {
    tf_glm_exl3_decode::projection<true>(x, d, d, items, counts, members, nullptr, pairs, experts, svh, out);
}
