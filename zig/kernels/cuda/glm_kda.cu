// GLM keeps its existing image, entry points, bounds and compiler flags.
#include <stddef.h>
#include <stdint.h>
#include "common/kda.cuh"
#include "glm_pdl.cuh"

extern "C" __global__ void __launch_bounds__(1024) tf_glm_kda_chain(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ G, int g_stride, const __nv_bfloat16* __restrict__ cs,
        const float* __restrict__ cw, const float* __restrict__ state_in, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, const __nv_bfloat16* __restrict__ norm_w, float eps, float lower, int rows,
        __nv_bfloat16* __restrict__ out, float* __restrict__ state_out, float* __restrict__ k_save,
        __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save, float* __restrict__ b_save) {
    glm_pdl::enter();
    chain_kernel_body(
        H, P, p_stride, b_off, A, a_stride, G, g_stride, cs, cw, state_in, a_log, dt_bias, norm_w, eps, lower, rows,
        out, state_out, k_save, v_save, g_save, b_save);
}

extern "C" __global__ void __launch_bounds__(1024) tf_glm_kda_replay(
        int H, const float* __restrict__ state_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, float* __restrict__ state_out) {
    glm_pdl::enter();
    replay_kernel_body(H, state_in, k_save, v_save, g_save, b_save, rows, state_out);
}

extern "C" __global__ void __launch_bounds__(1024) tf_glm_kda_replay_layers(
        int H, const float* __restrict__ state_in, size_t state_stride, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        size_t kv_stride, size_t b_stride, int rows, float* __restrict__ state_out) {
    glm_pdl::enter();
    replay_layers_kernel_body(
        H, state_in, state_stride, k_save, v_save, g_save, b_save, kv_stride, b_stride, rows, state_out);
}

extern "C" __global__ void __launch_bounds__(512) tf_glm_kda_prep(
        int H, const __nv_bfloat16* __restrict__ P, int p_stride, int b_off, const __nv_bfloat16* __restrict__ A,
        int a_stride, const __nv_bfloat16* __restrict__ cs, const float* __restrict__ cw,
        const float* __restrict__ a_log, const float* __restrict__ dt_bias, float lower, float* __restrict__ q_out,
        float* __restrict__ k_save, __nv_bfloat16* __restrict__ v_save, float* __restrict__ g_save,
        float* __restrict__ b_save) {
    glm_pdl::enter();
    prep_kernel_body(
        H, P, p_stride, b_off, A, a_stride, cs, cw, a_log, dt_bias, lower, q_out, k_save, v_save, g_save, b_save);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_step(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    glm_pdl::enter();
    step_kernel_body<8, 16>(H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out);
}

// The same step with 4 and 2 warps a block: each warp keeps its 4 value rows and arithmetic, so the same bits.
extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_step4(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    glm_pdl::enter();
    step_kernel_body<4, 16>(H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out);
}

extern "C" __global__ void __launch_bounds__(64) tf_glm_kda_step2(
        int H, const float* __restrict__ state_in, const float* __restrict__ q_in, const float* __restrict__ k_save,
        const __nv_bfloat16* __restrict__ v_save, const float* __restrict__ g_save, const float* __restrict__ b_save,
        int rows, __nv_bfloat16* __restrict__ y_out, float* __restrict__ state_out) {
    glm_pdl::enter();
    step_kernel_body<2, 16>(H, state_in, q_in, k_save, v_save, g_save, b_save, rows, y_out, state_out);
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_out(
        int H, const __nv_bfloat16* __restrict__ y_in, const __nv_bfloat16* __restrict__ G, int g_stride,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out) {
    glm_pdl::enter();
    out_kernel_body(H, y_in, G, g_stride, norm_w, eps, out);
}

// A shared window's segments in one launch a kernel (z: the segment): each its own prep/step/out launch's blocks.
struct KdaSeg {
    const __nv_bfloat16* P;   // the segment's projection rows (its stream's copy)
    const __nv_bfloat16* A;   // its decay rows
    const __nv_bfloat16* G;   // its gate rows
    const __nv_bfloat16* cs;  // its stream's conv window
    const float* state_in;
    float* state_out;
    float* q;                 // the window's q scratch from the segment's first row
    float* k_save;
    __nv_bfloat16* v_save;
    float* g_save;
    float* b_save;
    __nv_bfloat16* y;         // the window's read-out scratch from the segment's first row
    __nv_bfloat16* out;       // the window's out rows from the segment's first row
    int rows;
};
constexpr int KDA_SEGS = 16;  // Engine.max_shared
struct KdaSegs {
    KdaSeg s[KDA_SEGS];
};
static_assert(sizeof(KdaSeg) == 112 && offsetof(KdaSeg, rows) == 104, "cuda/kda.zig's Seg");
static_assert(sizeof(KdaSegs) == KDA_SEGS * 112, "cuda/kda.zig's Segs");

extern "C" __global__ void __launch_bounds__(512) tf_glm_kda_prep_segs(
        int H, int p_stride, int b_off, int a_stride, const float* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float lower, const KdaSegs t) {
    glm_pdl::enter();
    const KdaSeg& g = t.s[blockIdx.z];
    if (static_cast<int>(blockIdx.x) >= g.rows) return;  // past the segment's own grid (its rows), the whole block
    prep_kernel_body(H, g.P, p_stride, b_off, g.A, a_stride, g.cs, cw, a_log, dt_bias, lower, g.q, g.k_save, g.v_save,
                     g.g_save, g.b_save);
}

extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_step_segs(int H, const KdaSegs t) {
    glm_pdl::enter();
    const KdaSeg& g = t.s[blockIdx.z];
    step_kernel_body<8, 16>(H, g.state_in, g.q, g.k_save, g.v_save, g.g_save, g.b_save, g.rows, g.y, g.state_out);
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_step4_segs(int H, const KdaSegs t) {
    glm_pdl::enter();
    const KdaSeg& g = t.s[blockIdx.z];
    step_kernel_body<4, 16>(H, g.state_in, g.q, g.k_save, g.v_save, g.g_save, g.b_save, g.rows, g.y, g.state_out);
}

extern "C" __global__ void __launch_bounds__(64) tf_glm_kda_step2_segs(int H, const KdaSegs t) {
    glm_pdl::enter();
    const KdaSeg& g = t.s[blockIdx.z];
    step_kernel_body<2, 16>(H, g.state_in, g.q, g.k_save, g.v_save, g.g_save, g.b_save, g.rows, g.y, g.state_out);
}

extern "C" __global__ void __launch_bounds__(128) tf_glm_kda_out_segs(
        int H, int g_stride, const __nv_bfloat16* __restrict__ norm_w, float eps, const KdaSegs t) {
    glm_pdl::enter();
    const KdaSeg& g = t.s[blockIdx.z];
    if (static_cast<int>(blockIdx.x) >= g.rows) return;
    out_kernel_body(H, g.y, g.G, g_stride, norm_w, eps, g.out);
}

// A shared window's projection rows into each segment's stream, 16-byte units, in one launch (y: the copy).
struct KdaCopy {
    const uint4* src;
    uint4* dst;
    int64_t n;  // 16-byte units
};
struct KdaCopies {
    KdaCopy c[KDA_SEGS];
};
static_assert(sizeof(KdaCopies) == KDA_SEGS * 24, "cuda/kda.zig's Copies");

extern "C" __global__ void __launch_bounds__(256) tf_glm_kda_copy_segs(const KdaCopies t) {
    glm_pdl::enter();
    const KdaCopy& c = t.c[blockIdx.y];
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < c.n; i += (int64_t)gridDim.x * blockDim.x)
        c.dst[i] = c.src[i];
}
