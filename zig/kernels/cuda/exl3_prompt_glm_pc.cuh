// TF_EXL3_PROMPT_GU=pc: tf_exl3_prompt_gu_glm on 128-member items, warp-specialized (shared rotation only).
#pragma once
#include "exl3_prompt_glm.cuh"
namespace tf_exl3_prompt_glm {

constexpr int PM = 2 * TPM, PCW = 16, PDW = 3, PW = PCW + PDW + 1, PTH = PW * 32;   // members, consumer/decoder warps, threads
constexpr int PNX = 3, PND = 3, PCR = 4, PCD = PCR - 1;   // stages of rows (and barriers), of fragments, of compressed tiles
constexpr int PXB = PM * TXS * 2, PDB = TKT * 16 * 32 * 16, PCB = 16 * TKT * tile_words<8>() * 4;   // a stage's bytes of each
constexpr int POD = PNX * PXB, POC = POD + PND * PDB, POR = POC + PCR * PCB, POB = POR + PM * 4 + 16;   // X rows at 0; rows, e, cnt
constexpr int PSMEM = POB + 2 * PNX * 8;   // with full[PNX] then empty[PNX]
static_assert(2 * TPM * GU_ES * 4 <= POC && PND <= PNX && PSMEM <= 101376, "64 members' sums over the stages; sm_86+ opt-in");
static_assert(PSMEM == 96832 && PTH == 640, "exl3_prompt.zig's pc geometry");

__device__ __forceinline__ void mb_init(uint32_t a, int n) { asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" ::"r"(a), "r"(n) : "memory"); }
__device__ __forceinline__ void mb_arrive(uint32_t a) {
    asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared.b64 st, [%0];\n}\n" ::"r"(a) : "memory");
}
// The arrival once every cp.async this thread issued so far has landed (the barrier counts it at init).
__device__ __forceinline__ void mb_cp_arrive(uint32_t a) { asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" ::"r"(a) : "memory"); }
// Until the barrier's phase of this parity completes (acquire: its producers' writes are visible).
__device__ __forceinline__ void mb_wait(uint32_t a, uint32_t parity) {
    uint32_t done;
    do {
#if __CUDA_ARCH__ >= 900
        asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
#else
        asm volatile("{\n.reg .pred p;\nmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n" : "=r"(done) : "r"(a), "r"(parity) : "memory");
#endif
    } while (!done);
}
__device__ __forceinline__ void lds128(uint32_t (&r)[4], uint32_t a) {
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void sts128(uint32_t a, const uint32_t (&r)[4]) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(a), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]) : "memory");
}

// tgateup_n's row: one member's gate (g) and up (u) sums, 4 a lane, op for op (FWHT, scales, activation, FWHT, store).
__device__ __forceinline__ void pc_row(const float* g, const float* u, const half* __restrict__ svh_g, const half* __restrict__ svh_u,
        const half* __restrict__ suh_d, half* __restrict__ xd, int p, int e, int N, int n, float limit, int act_mode, int lane) {
    float gv[4], uv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        gv[j] = g[4 * lane + j];
        uv[j] = u[4 * lane + j];
    }
    fwht128f(gv, lane);
    fwht128f(uv, lane);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float gg = gv[j] * HAD_SCALE * __half2float(svh_g[(size_t)e * N + n + j]);
        float uu = uv[j] * HAD_SCALE * __half2float(svh_u[(size_t)e * N + n + j]);
        if (act_mode == 0) { gg = bf16r(gg); uu = bf16r(uu); }
        if (limit > 0) { gg = upTo(gg, limit); uu = within(uu, limit); }
        float act = gg / (1.f + expf(-gg));
        if (act_mode == 0) act = bf16r(act);
        act *= uu;
        if (act_mode == 0) act = bf16r(act);
        v[j] = act * __half2float(suh_d[(size_t)e * N + n + j]);
    }
    fwht128f(v, lane);
    const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
    const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
    uint2 packed;
    packed.x = *reinterpret_cast<const uint32_t*>(&o01);
    packed.y = *reinterpret_cast<const uint32_t*>(&o23);
    *reinterpret_cast<uint2*>(xd + (size_t)p * N + n) = packed;
}

// The kernel's epilogue arguments (parameters only: the item's expert and count are read back from shared memory).
struct PcEpi {
    const half* svh_g;
    const half* svh_u;
    const half* suh_d;
    half* xd;
    int N;
    float limit;
    int act_mode;
};
// Every warp; a pass spills members 32 hf.. of each half h as rows 32 h.. of [gate | up][64][GU_ES], then their rows.
template <class Spill>
__device__ __forceinline__ void pc_epilogue(const PcEpi& a, unsigned char* smem, Spill spill, int warp, int lane) {
    __syncthreads();   // every stage consumed and every copy landed: the stages become the sums
    float* E = reinterpret_cast<float*>(smem);
    const int* rows_sh = reinterpret_cast<const int*>(smem + POR);
    const volatile int* meta = rows_sh + PM;   // e, cnt: read where used, not held through the rows
    for (int hf = 0; hf < 2 && hf * TPH < meta[1]; ++hf) {
        if (hf) __syncthreads();   // the first pass's rows are read
        spill(E, hf);
        __syncthreads();
        for (int r = warp; r < TPM; r += PW) {
            const int m = (r / TPH) * TPM + hf * TPH + r % TPH;
            if (m < meta[1])
                pc_row(E + (size_t)r * GU_ES, E + (size_t)(TPM + r) * GU_ES, a.svh_g, a.svh_u, a.suh_d, a.xd, rows_sh[m], meta[0], a.N,
                       (int)blockIdx.x * 128 + 4 * lane, a.limit, a.act_mode, lane);
        }
    }
}

// Consumer warp: members 64 h.. of tiles 2 j, 2 j + 1 (tgateup_n's warp j), NBA blocks; tmain's mma chain per output.
template <int NBA>
__device__ __forceinline__ void pc_consume(const PcEpi& a, unsigned char* smem, uint32_t sb, int chunks, int warp, int lane) {
    const int h = warp >> 3, j = warp & 7;
    float acc[TMT][NBA][4];
#pragma unroll
    for (int i = 0; i < TMT; ++i)
#pragma unroll
        for (int b = 0; b < NBA; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][b][c] = 0.f;
    const uint32_t xlane = (uint32_t)(((h * TPM + (lane & 7) + ((lane >> 4) & 1) * 8) * TXS + ((lane >> 3) & 1) * 8) * 2);
    const uint32_t dlane = (uint32_t)(POD + (j * TMT * 32 + lane) * 16);   // tile 2 j's fragment of k tile 0
    static_assert(PND == PNX, "one slot index for rows, fragments and barriers");
    uint32_t xs = sb + xlane, ds = sb + dlane, bar = sb + POB, ph = 0;   // this stage's rows, fragments, full barrier; its parity
#pragma unroll 1
    for (int c = 0; c < chunks; ++c) {
        mb_wait(bar, ph);
#pragma unroll 1
        for (int kk = 0; kk < TKT; ++kk) {   // rolled: k tile 1's operands are not loaded early (96 registers, no spills)
            uint32_t A[TMT][4];
#pragma unroll
            for (int i = 0; i < TMT; ++i) lds128(A[i], ds + (uint32_t)((kk * 16 + i) * 32 * 16));
#pragma unroll
            for (int jp = 0; jp < (NBA + 1) / 2; ++jp) {
                const uint32_t xa = xs + (uint32_t)(kk * 32 + jp * 16 * TXS * 2);
                if (2 * jp + 1 < NBA) {
                    uint32_t r[4];
                    ldsm4(r, xa);
                    const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
                    for (int i = 0; i < TMT; ++i) {
                        mma16816(acc[i][2 * jp], A[i], bl);
                        mma16816(acc[i][2 * jp + 1], A[i], bh);
                    }
                } else {
                    uint32_t r[2];
                    ldsm2(r, xa);
#pragma unroll
                    for (int i = 0; i < TMT; ++i) mma16816(acc[i][2 * jp], A[i], r);
                }
            }
        }
        __syncwarp();
        if (lane == 0) mb_arrive(bar + PNX * 8);   // this warp is done with the stage
        if (bar == sb + POB + (PNX - 1) * 8) bar -= (PNX - 1) * 8, xs -= (PNX - 1) * PXB, ds -= (PNX - 1) * PDB, ph ^= 1;
        else bar += 8, xs += PXB, ds += PDB;
    }
    const int g = lane >> 2, t = lane & 3, mat = j >> 2, col0 = ((2 * j) & 7) * 16;
    pc_epilogue(a, smem, [&](float* E, int hf) {
#pragma unroll
        for (int i = 0; i < TMT; ++i)
#pragma unroll
            for (int b = 0; b < NBA; ++b)
                if (b / 4 == hf) {   // tspill's blocks of this pass
                    float* r = E + (size_t)mat * TPM * GU_ES + (size_t)(h * TPH + (b % 4) * 8 + 2 * t) * GU_ES + col0 + i * 16 + g;
                    r[0] = acc[i][b][0];
                    r[GU_ES] = acc[i][b][1];
                    r[8] = acc[i][b][2];
                    r[GU_ES + 8] = acc[i][b][3];
                }
    }, warp, lane);
}

// Decoder warp p: copies compressed tiles PCD stages ahead, decodes units p, p + PDW, ... (tile u % 16, k tile u / 16).
__device__ __forceinline__ void pc_decode(const uint32_t* T0, const uint32_t* T1, int NTILES, int cb, uint32_t sb, int chunks, int p, int lane) {
    constexpr int TW = tile_words<8>(), TB = TW * 4, PIECES = 16 * TKT * TW / 4, PPT = (PIECES + PDW * 32 - 1) / (PDW * 32);
    const int dt = p * 32 + lane;
    const uint32_t* src[PPT];
    uint32_t dst[PPT];
#pragma unroll
    for (int i = 0; i < PPT; ++i) {   // tmain's weight pieces: tile, k tile, part
        const int idx = dt + i * PDW * 32, q = (idx < PIECES ? idx : 0) / (TKT * 8), kk = (idx / 8) % TKT, part = idx % 8;
        int mat, nt;
        ttile<2>(q, cb, mat, nt);
        src[i] = (mat ? T1 : T0) + ((size_t)kk * NTILES + nt) * TW + part * 4;
        dst[i] = sb + (uint32_t)(POC + ((q * TKT + kk) * TW + part * 4) * 4);
    }
    const size_t wstep = (size_t)TKT * NTILES * TW;
    auto issue = [&](int c) {
#pragma unroll
        for (int i = 0; i < PPT; ++i)
            if (dt + i * PDW * 32 < PIECES) cp16s(dst[i] + (uint32_t)((c % PCR) * PCB), src[i] + (size_t)c * wstep, 16);
    };
#pragma unroll
    for (int s = 0; s < PCD; ++s) {
        if (s < chunks) issue(s);
        cp_commit();
    }
    const uint32_t wp = (uint32_t)(((lane + 31) & 31) * 4), wc = (uint32_t)(lane * 4);   // lane_start<8>: words lane - 1, lane
    for (int c = 0; c < chunks; ++c) {
        cp_wait<PCD - 1>();   // this thread's copies of stage c landed
        asm volatile("bar.sync 1, %0;\n" ::"n"(PDW * 32) : "memory");   // every decoder's, and stage c - 1's slot is read
        if (c + PCD < chunks) issue(c + PCD);   // into stage c - 1's slot
        cp_commit();
        if (c >= PND) mb_wait(sb + POB + (uint32_t)((PNX + (c - PND) % PNX) * 8), ((c - PND) / PNX) & 1);   // the fragment slot is free
        const uint32_t cs = sb + POC + (uint32_t)((c % PCR) * PCB), ds = sb + POD + (uint32_t)((c % PND) * PDB);
#pragma unroll
        for (int u = p; u < 16 * TKT; u += PDW) {
            const int q = u & 15, kk = u >> 4;
            const uint32_t tile = cs + (uint32_t)((q * TKT + kk) * TB);
            uint32_t A[4];
            decode_a8(lds32(tile + wc), lds32(tile + wp), A);
            sts128(ds + (uint32_t)((u * 32 + lane) * 16), A);
        }
        mb_arrive(sb + POB + (uint32_t)((c % PNX) * 8));
    }
    cp_wait<0>();
}

// Copier warp: the item's X rows as tmain stages them (rows past 8 nb untouched, invalid members zero), PNX - 1 ahead.
__device__ __forceinline__ void pc_copy(const half* X, const int* rows_sh, int K, int xdiv, int nb, uint32_t sb, int chunks, int lane) {
    uint32_t off[PM / 8], live = 0;   // member rows lane / 4 + 8 i, part lane % 4
#pragma unroll
    for (int i = 0; i < PM / 8; ++i) {
        const int r = rows_sh[(lane >> 2) + 8 * i];
        off[i] = (uint32_t)((r < 0 ? 0 : r / xdiv) * K + (lane & 3) * 8);
        live |= (r >= 0 ? 1u : 0u) << i;
    }
    const uint32_t xd0 = sb + (uint32_t)((((lane >> 2) * TXS) + (lane & 3) * 8) * 2);
    uint32_t sx = 0, ph = 1;
    for (int c = 0; c < chunks; ++c) {
        if (c >= PNX) mb_wait(sb + POB + (PNX + sx) * 8, ph);   // stage c - PNX's rows are free
#pragma unroll
        for (int i = 0; i < PM / 8; ++i)
            if (i < nb) cp16s(xd0 + sx * PXB + (uint32_t)(i * 8 * TXS * 2), X + off[i] + (size_t)c * TBK, (live >> i) & 1 ? 16 : 0);
        mb_cp_arrive(sb + POB + sx * 8);
        if (++sx == PNX) sx = 0, ph ^= 1;
    }
}

extern "C" __global__ void __launch_bounds__(PTH, 1) tf_exl3_prompt_gu_glm_pc(const half* __restrict__ Xg, const half* __restrict__ Xu,
        const uint64_t* __restrict__ Tg, const uint64_t* __restrict__ Tu, const int* __restrict__ items, const int* __restrict__ counts,
        const int* __restrict__ members, const half* __restrict__ svh_g, const half* __restrict__ svh_u, const half* __restrict__ suh_d,
        half* __restrict__ xd, int K, int N, int xdiv, float limit, int act_mode) {
    extern __shared__ __align__(128) unsigned char pc_smem[];
    (void)Xu;   // the shared rotation: gate and up read Xg's rows
    const int cb = blockIdx.x, item = blockIdx.y;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1], cnt = items[3 * item + 2], nb = (cnt + 7) >> 3;
    const uint32_t sb = (uint32_t)__cvta_generic_to_shared(pc_smem);
    int* rows_sh = reinterpret_cast<int*>(pc_smem + POR);
    if (threadIdx.x < PM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    if (threadIdx.x == PM) rows_sh[PM] = e, rows_sh[PM + 1] = cnt;
    if (threadIdx.x == 0)
        for (int s = 0; s < PNX; ++s) {
            mb_init(sb + POB + s * 8, PDW * 32 + 32);   // full: every decoder lane, every copier lane's landed rows
            mb_init(sb + POB + (PNX + s) * 8, nb > 8 ? PCW : PCW / 2);   // empty: the consumer warps that have members
        }
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, chunks = K / TBK;
    const PcEpi a{svh_g, svh_u, suh_d, xd, N, limit, act_mode};
    if (warp < PCW) {
        const int left = nb - (warp >> 3) * 8;
        switch (left < 0 ? 0 : left > 8 ? 8 : left) {
            case 0: pc_epilogue(a, pc_smem, [](float*, int) {}, warp, lane); break;
#define TF_EXL3_PROMPT_PC(n) \
    case n: pc_consume<n>(a, pc_smem, sb, chunks, warp, lane); break;
            TF_EXL3_PROMPT_PC(1) TF_EXL3_PROMPT_PC(2) TF_EXL3_PROMPT_PC(3) TF_EXL3_PROMPT_PC(4)
            TF_EXL3_PROMPT_PC(5) TF_EXL3_PROMPT_PC(6) TF_EXL3_PROMPT_PC(7) TF_EXL3_PROMPT_PC(8)
#undef TF_EXL3_PROMPT_PC
        }
        return;
    }
    if (warp < PCW + PDW) pc_decode(reinterpret_cast<const uint32_t*>(Tg[e]), reinterpret_cast<const uint32_t*>(Tu[e]), N >> 4, cb, sb, chunks, warp - PCW, lane);
    else pc_copy(Xg, rows_sh, K, xdiv, nb, sb, chunks, lane);
    pc_epilogue(a, pc_smem, [](float*, int) {}, warp, lane);
}

// The 64-member slot kernels on 128-member items: block y takes half y % 2 of item y / 2 (a 64-member plan's members).
template <typename Y>
__device__ __forceinline__ void prompt_down_glm_h(const half* __restrict__ Xd, const uint64_t* __restrict__ Td, const int* __restrict__ items,
        const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d, Y* __restrict__ y, int K, int D,
        int ys, int yrow, unsigned char* smem_raw, int* rows_sh) {
    const int cb = blockIdx.x, item = blockIdx.y >> 1, hf = blockIdx.y & 1;
    if (item >= counts[0]) return;
    const int e = items[3 * item], first = items[3 * item + 1] + hf * TPM, cnt = items[3 * item + 2] - hf * TPM;
    if (cnt <= 0) return;
    if (threadIdx.x < TPM) rows_sh[threadIdx.x] = (int)threadIdx.x < cnt ? members[first + threadIdx.x] : -1;
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    prompt_down<8, Y>(Xd, Td, svh_d, y, K, D, ys, yrow, smem_raw, rows_sh, cb, e, cnt, warp, lane);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down_glm_h(const half* __restrict__ Xd, const uint64_t* __restrict__ Td,
        const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members, const half* __restrict__ svh_d,
        float* __restrict__ y, int K, int D, int ys, int yrow) {
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes<8>()];
    __shared__ int rows_sh[TPM];
    prompt_down_glm_h<float>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, yrow, smem_raw, rows_sh);
}

extern "C" __global__ void __launch_bounds__(256, 2) tf_exl3_prompt_down_glm_h_b16(const half* __restrict__ Xd,
        const uint64_t* __restrict__ Td, const int* __restrict__ items, const int* __restrict__ counts, const int* __restrict__ members,
        const half* __restrict__ svh_d, __nv_bfloat16* __restrict__ y, int K, int D, int ys, int yrow) {
    __shared__ __align__(16) unsigned char smem_raw[tsmem_bytes<8>()];
    __shared__ int rows_sh[TPM];
    prompt_down_glm_h<__nv_bfloat16>(Xd, Td, items, counts, members, svh_d, y, K, D, ys, yrow, smem_raw, rows_sh);
}

}  // namespace tf_exl3_prompt_glm
