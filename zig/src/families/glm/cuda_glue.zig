//! GLM-5.3-Flash's glue on CUDA: the Python family's cuda/glue.py on glm_glue.cu, whose kernels keep Triton's bytes.

const std = @import("std");
const cuda = @import("cuda");
const mm_mod = @import("cuda_mm.zig");
const index_mod = @import("cuda_index.zig");

/// What the forward's kernels launch from: our glue's, glm_mm's when tiled, qlinear's lane for FP8 kinds, copy_rows.
pub const Set = struct {
    glue: *const Kernels,
    index: ?*const index_mod.Kernels = null,
    mm: ?*const mm_mod.Kernels = null,
    packs: ?*const mm_mod.Packed = null,
    prompt: bool = false,
    linear: cuda.qlinear.Linear = .{},
    copy_rows: ?cuda.Function = null,
    absorb_mma: bool = false,
    pdl_hc: bool = false, // TF_GLM_PDL_CHAIN's hc: a decode window's hc_post, hc_partial8 and hc_finish as dependents
    pdl_moe: bool = false, // and its moe: a decode window's router sums and top-k pick
};

/// A hidden-size norm's weight: bf16, or float32 (FP4-Local's) read as float32 by the _f32w kernels.
pub const Norm = union(enum) { bf16: u64, f32: u64 };

/// Our glm_glue.cu's functions.
pub const Kernels = struct {
    rmsnorm_4096: cuda.Function,
    rmsnorm_1536: cuda.Function,
    rmsnorm_512: cuda.Function,
    rmsnorm_4096_f32w: cuda.Function,
    residual_add: cuda.Function,
    stream_mean: cuda.Function,
    rank_sum: cuda.Function,
    hc_post: cuda.Function,
    hc_post_partial: cuda.Function, // TF_GLM_HC_FUSE: hc_post and the next site's hc_partial in one launch
    combine: cuda.Function,
    slice_sum: cuda.Function,
    top1: cuda.Function,
    top1_sums: cuda.Function,
    put_words: cuda.Function,
    embed_b16: cuda.Function,
    conv_shift: cuda.Function,
    swiglu_512: cuda.Function,
    swiglu_64: cuda.Function,
    group_sums: cuda.Function,
    topk: cuda.Function,
    router_part: cuda.Function,
    router_tile: cuda.Function,
    hc_partial: cuda.Function,
    hc_partial8: cuda.Function,
    hc_partial8_o2: cuda.Function, // its bytes from 13 blocks of two dots a K block (hc_part)
    hc_finish: cuda.Function,
    hc_finish_f32w: cuda.Function,
    hc_finish_split: cuda.Function, // the same bytes with warp 0's Sinkhorn beside the collapse (hc_split)
    hc_finish_f32w_split: cuda.Function,
    absorb_q: cuda.Function,
    expand_v: cuda.Function,
    lat_write: cuda.Function,
    lat_write_fp8: cuda.Function,
    index_write: cuda.Function,
    pool_keys: cuda.Function,
    select_rows: cuda.Function,
    select_rows_wide: cuda.Function,
    bmm_decode: cuda.Function,
    bmm_rows: cuda.Function,
    bmm_tile: cuda.Function,
    bmm_tile1: cuda.Function,
    bmm_d32s1: cuda.Function,
    bmm_d32s2: cuda.Function,
    bmm_d32s3: cuda.Function,
    bmm_d64s2: cuda.Function,
    chunks_16: cuda.Function,
    chunks_32: cuda.Function,
    chunks_16_fp8: cuda.Function,
    chunks_32_fp8: cuda.Function,
    chunks_16_fused: cuda.Function, // a sparse call past its partials: the chunks and their merge in one launch
    chunks_32_fused: cuda.Function,
    chunks_16_fp8_fused: cuda.Function,
    chunks_32_fp8_fused: cuda.Function,
    merge: cuda.Function,
    scores: cuda.Function,
    latent_mma_kn: cuda.Function,
    latent_mma_nk: cuda.Function,
    hc_split: bool = true, // TF_GLM_HC_SPLIT: a decode window's hc_finish on the split schedule (the same bits)
    hc_part: bool = true, // TF_GLM_HC_PART: a decode window to hc_part_rows rows takes tf_glm_hc_partial8_o2 (the same bits)

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        const info = @typeInfo(Kernels).@"struct";
        inline for (info.field_names, info.field_types) |name, T| if (T == cuda.Function) {
            @field(k, name) = try m.function("tf_glm_" ++ name);
        };
        const env = @import("sys").environment;
        k.hc_split = env.read(bool, "TF_GLM_HC_SPLIT", true) catch true;
        k.hc_part = env.read(bool, "TF_GLM_HC_PART", true) catch true;
        try k.bmm_tile.allowDynamicShared(bmm_tile_smem);
        try k.bmm_tile1.allowDynamicShared(bmm_tile1_smem[1]);
        try k.chunks_16.allowDynamicShared(chunksSmem(16));
        try k.chunks_32.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fused.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fused.allowDynamicShared(chunksSmem(32));
        try k.chunks_16_fp8_fused.allowDynamicShared(chunksSmem(16));
        try k.chunks_32_fp8_fused.allowDynamicShared(chunksSmem(32));
        try k.latent_mma_kn.allowDynamicShared(latent_mma_smem[0]);
        try k.latent_mma_nk.allowDynamicShared(latent_mma_smem[1]);
        return k;
    }
};

/// An element-wise kernel's launch: a block of 256 columns, `rows` x the blocks to cover `d`.
fn rowsLaunch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 256) }, .block = .{ .x = 256 } }, stream, a);
}

/// As above, 8 columns a thread, 1024 a block of 128 (Triton's BLOCK); d % 8 == 0, rows 16B-aligned; `pdl`: dependent.
fn rows8Launch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize, ptrs: anytype, pdl: bool) !void {
    if (d % 8 != 0) return error.WidthNotEights;
    inline for (ptrs) |p| if (p % 16 != 0) return error.RowsUnaligned;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 1024) }, .block = .{ .x = 128 }, .pdl = pdl }, stream, a);
}

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn pow2(d: usize) i64 {
    return @intCast(std.math.ceilPowerOfTwoAssert(usize, d));
}

/// glue.rmsnorm: bf16(w * bf16(x / sqrt(mean(x^2) + eps))) into `out`, with `xs` each 64 outputs' fp32 sum.
pub fn rmsnorm(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const f, const warps: u32 = switch (d) {
        4096 => .{ set.glue.rmsnorm_4096, 8 },
        1536 => .{ set.glue.rmsnorm_1536, 4 },
        512 => .{ set.glue.rmsnorm_512, 4 },
        else => return error.NoRmsnormWidth,
    };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(x_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(o_stride)));
    a.add(xs orelse @as(u64, 0));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(xs != null)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
}

/// rmsnorm by a Norm: a float32 weight (the hidden size) through rmsnorm_4096_f32w, rmsnorm_4096's arithmetic.
pub fn rmsnormBy(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, n: Norm, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const g = switch (n) {
        .bf16 => |w| return rmsnorm(set, stream, x, x_stride, w, out, o_stride, xs, eps, rows, d),
        .f32 => |g| g,
    };
    if (d != 4096) return error.NoRmsnormWidth;
    var a: cuda.Args = .{};
    inline for (.{ x, i64of(x_stride), g, out, i64of(o_stride), xs orelse @as(u64, 0), eps, @as(c_int, @intFromBool(xs != null)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.rmsnorm_4096_f32w, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &a);
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// swiglu's column block: 512 where it divides the width, else 64 (TP=3's 704 and 640, the latter 128 in glue.py).
fn swigluBlock(w: usize) !usize {
    if (w % 512 == 0) return 512;
    if (w % 64 == 0) return 64;
    return error.NoSwigluBlock;
}

/// glue.swiglu: [gate | up] -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)), 64-sums in `xs`.
pub fn swiglu(set: Set, stream: cuda.Stream, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
    const block = try swigluBlock(w);
    const f, const threads: u32 = if (block == 512) .{ set.glue.swiglu_512, 128 } else .{ set.glue.swiglu_64, 64 };
    var a: cuda.Args = .{};
    inline for (.{ gu, out, xs, limit, i64of(w) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(w / block) }, .block = .{ .x = threads } }, stream, &a);
}

/// glm_glue_bmm.cuh's Tile::smem(f32) of tf_glm_bmm_tile: 3 stages of 128 x and 64 w rows, 64 bf16 a row.
pub const bmm_tile_smem: u32 = 3 * (128 + 64) * 64 * 2;
/// tf_glm_bmm_tile1's [bf16, fp32 out]: 3 stages of 128 x and 128 w rows, 32 bf16 a row, or the fp32 tile if larger.
pub const bmm_tile1_smem = [2]u32{ 3 * (128 + 128) * 32 * 2, 128 * (128 + 4) * 4 };
/// glm_glue_num.cuh's three-stage [absorb, expand] dynamic shared bytes.
pub const latent_mma_smem = [2]u32{ 3 * (128 * 40 + 32 * 136) * 2, 3 * (128 * 40 + 128 * 40) * 2 };
/// glm_glue_attn.cuh's wide_smem<hb>(): q (hb x 512), two 32-key tiles, p (hb x 32) in bf16, then max/sum scratch.
pub fn chunksSmem(comptime hb: u32) u32 {
    return (hb * 512 + 2 * 32 * 512 + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// K slices of the router's dot (glue.ROUTER_KS): fixed by the shape, as the sliced and the single pass differ in bits.
pub const router_ks = 8;

/// glue.router: out [m][ne] fp32 = x . w rows; d a multiple of router_ks x 64: K slices in `part`, added in order.
pub fn router(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    if (d % (router_ks * 64) != 0) return error.RouterWidth; // the Python family's single-pass _router: no GLM width
    var a: cuda.Args = .{};
    inline for (.{ x, w, part, i64of(m), i64of(x_stride), i64of(d), i64of(ne), @as(c_int, router_ks) }) |v| a.add(v);
    if (x % 16 == 0 and x_stride % 8 == 0) // rows on 16 bytes: 64-row tiles staged by cp.async (decode's rows past m load as zeros)
        try cuda.launch.launch(set.glue.router_tile, .{ .grid = .{ .x = cdiv(m, 64), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a)
    else
        try cuda.launch.launch(set.glue.router_part, .{ .grid = .{ .x = cdiv(m, 16), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a);
    try sliceSum(set, stream, part, out, m * ne, router_ks, true);
}

/// glue.select: top_k by sigmoid + bias (lower id on ties), normed if `norm`, times `scale`, then the shared expert.
pub fn select(set: Set, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    var a: cuda.Args = .{};
    inline for (.{ logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
    // `pdl_moe`: a dependent launch; the kernel waits before it reads the logits or writes anything
    try cuda.launch.launch(set.glue.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 }, .pdl = set.pdl_moe }, stream, &a);
}

/// glue.combine: out [rows][d] fp32 = sum over slots in order of wts[r][k] y[r][k] (y fp32 in decode, bf16 in prefill).
pub fn combine(set: Set, stream: cuda.Stream, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ y, @as(c_int, @intFromBool(y_f32)), wts, out, i64of(d), cint(slots) }) |v| a.add(v);
    try rows8Launch(set.glue.combine, stream, &a, rows, d, .{ y, out }, false);
}

/// K blocks of the hyper-connection mixing dots (glue.HC_BLOCKS): fixed by the shape, like the router's slices.
pub const hc_blocks = 16;

/// The most rows tf_glm_hc_partial8_o2 takes; past them tf_glm_hc_partial8 is faster (source kit's measurement).
pub const hc_part_rows = 16;

/// The mixing kernels' scratch: rows x hc_blocks x 32 fp32 (24 dots and the sum of squares of each K block).
pub fn hcPartBytes(rows: usize) usize {
    return rows * hc_blocks * 32 * 4;
}

/// hc_pre's prompt rows: from this many rows a call, 4 rows a 128-thread hc_partial block (fewer: hc_partial8).
pub const hc_prompt_rows = 64;

/// glue.hc_pre: streams [rows][s d] -> the collapsed RMS-normed row and its 64-sums `xs`; `post`, `comb` (Sinkhorn).
pub fn hcPre(set: Set, stream: cuda.Stream, x: u64, fn_w: u64, base: u64, scale: u64, norm: Norm, out: u64, xs: u64, post: u64, comb: u64, part: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    var a: cuda.Args = .{};
    inline for (.{ x, fn_w, part, i64of(s * d), @as(c_int, hc_blocks) }) |v| a.add(v);
    // Decode windows (16 blocks a row) take each block's 8 steps at once (`pdl_hc`: dependents); prompt rows 4 a block.
    if (rows <= hc_part_rows and set.glue.hc_part) {
        try cuda.launch.launch(set.glue.hc_partial8_o2, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks, .z = 24 / 2 + 1 }, .block = .{ .x = 256 }, .pdl = set.pdl_hc }, stream, &a);
    } else if (rows < hc_prompt_rows) {
        try cuda.launch.launch(set.glue.hc_partial8, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks }, .block = .{ .x = 1024 }, .pdl = set.pdl_hc }, stream, &a);
    } else {
        a.add(i64of(rows));
        try cuda.launch.launch(set.glue.hc_partial, .{ .grid = .{ .x = cdiv(rows, 4), .y = hc_blocks }, .block = .{ .x = 128 } }, stream, &a);
    }
    try hcFinish(set, stream, x, base, scale, norm, out, xs, post, comb, part, eps, hc_eps, iters, rows);
}

/// hc_pre's second launch alone, on the K blocks' partials already in `part` (hcPre's first, or hcPostPartial).
pub fn hcFinish(set: Set, stream: cuda.Stream, x: u64, base: u64, scale: u64, norm: Norm, out: u64, xs: u64, post: u64, comb: u64, part: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize) !void {
    const split = set.glue.hc_split and rows < hc_prompt_rows; // a decode window's latency; prompt rows keep the one-warp schedule
    const f, const w = switch (norm) {
        .bf16 => |p| .{ if (split) set.glue.hc_finish_split else set.glue.hc_finish, p },
        .f32 => |p| .{ if (split) set.glue.hc_finish_f32w_split else set.glue.hc_finish_f32w, p },
    };
    var b: cuda.Args = .{};
    inline for (.{ x, part, base, scale, w, out, xs, post, comb, eps, hc_eps, @as(c_int, hc_blocks), cint(iters) }) |v| b.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 }, .pdl = set.pdl_hc and rows < hc_prompt_rows }, stream, &b);
}

/// TF_GLM_HC_FUSE: hcPost, then the next site's hc_partial (`fn_w`) into `part` for hcFinish; same bytes, x read once.
pub fn hcPostPartial(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, post: u64, comb: u64, fn_w: u64, part: u64, rows: usize, d: usize, s: usize, world: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    if (rows < hc_prompt_rows) return error.HcFuseRows; // hcPre's decode rows run hc_partial8
    inline for (.{ x, xout, g, fn_w }) |p| if (p % 16 != 0) return error.RowsUnaligned;
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, post, comb, fn_w, part, i64of(rows * d), cint(world), i64of(rows) }) |v| a.add(v);
    // a block: 4 rows by a quarter of each stream (4 of the K blocks), 512 threads
    try cuda.launch.launch(set.glue.hc_post_partial, .{ .grid = .{ .x = cdiv(rows, 4), .y = @intCast(d / 1024) }, .block = .{ .x = 512 } }, stream, &a);
}

/// glue.hc_post: branch = bf16(`g`, rank 0 first); xout[s] = bf16(post[s] branch + sum_t comb[t][s] x[t]); x may alias.
pub fn hcPost(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, post: u64, comb: u64, rows: usize, d: usize, s: usize, world: usize) !void {
    if (s != 4) return error.StreamsNotFour;
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, post, comb, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.hc_post, stream, &a, rows, d, .{ x, xout, g }, set.pdl_hc);
}

/// glue.rank_sum: out bf16 = partials summed rank 0 first, `me`'s from `own`, others' from `peers` in rank order.
pub fn rankSum(set: Set, stream: cuda.Stream, own: u64, peers: u64, peer_stride: usize, out: u64, me: usize, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ own, peers, out, i64of(peer_stride), i64of(d), cint(world), cint(me) }) |v| a.add(v);
    try rows8Launch(set.glue.rank_sum, stream, &a, rows, d, .{ own, peers, out }, false);
}

/// glue.residual_add (the MTP block's plain residual): xout = bf16(x + bf16(the partials `g` summed rank 0 first)).
pub fn residualAdd(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.residual_add, stream, &a, rows, d, .{ x, xout, g }, false);
}

/// glue.stream_mean: out [rows][d] = bf16 of the s streams' mean (the final collapse).
pub fn streamMean(set: Set, stream: cuda.Stream, x: u64, out: u64, rows: usize, d: usize, s: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, out, i64of(d), cint(s) }) |v| a.add(v);
    try rows8Launch(set.glue.stream_mean, stream, &a, rows, d, .{ x, out }, false);
}

/// glue.embed of a BF16 table: each row's token (ids i32) as `copies` streams of d values, out [rows][copies d].
pub fn embedB16(set: Set, stream: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ ids, table, out, i64of(d), cint(copies) }) |v| a.add(v);
    try rowsLaunch(set.glue.embed_b16, stream, &a, rows, d);
}

/// forward._shift_conv: each layer's 3-row conv window becomes rows keep .. keep + 2 of [window; projection rows].
pub fn convShift(set: Set, stream: cuda.Stream, conv: u64, proj: u64, keep: usize, proj_layer: usize, proj_row: usize, layers: usize, c: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ conv, proj, i64of(keep), i64of(3 * c), i64of(proj_layer), i64of(proj_row), i64of(c) }) |v| a.add(v);
    try rowsLaunch(set.glue.conv_shift, stream, &a, layers, c);
}

/// glue._router_sum and qmm._reduce: `total` values summed over `ks` slices in slice order, fp32 or bf16 out.
pub fn sliceSum(set: Set, stream: cuda.Stream, part: u64, out: u64, total: usize, ks: usize, wide: bool) !void {
    var a: cuda.Args = .{};
    inline for (.{ part, out, i64of(total), cint(ks), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    // `pdl_moe`: a dependent launch (tf_glm_router_tile before it lets it launch); the kernel waits before it reads
    try cuda.launch.launch(set.glue.slice_sum, .{ .grid = .{ .x = cdiv(total, 1024) }, .block = .{ .x = 128 }, .pdl = set.pdl_moe }, stream, &a);
}

test "the norm's block is the next power of two of the width" {
    try std.testing.expectEqual(@as(i64, 4096), pow2(4096));
    try std.testing.expectEqual(@as(i64, 8192), pow2(6144));
}

test "the wrappers' shape rules as glue.py's" {
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(6144)); // swiglu's block at the dense TP=2 width
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(1024)); // a routed expert's rows at TP=2
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(704)); // TP=3's first two ranks
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(640)); // TP=3's last rank
    try std.testing.expectError(error.NoSwigluBlock, swigluBlock(96));
    try std.testing.expectEqual(@as(i64, 512), pow2(288 + 1)); // _topk's BLOCK for 288 experts
    try std.testing.expectEqual(@as(i64, 16), pow2(8 + 1)); // its SLOTP for top-8 and the shared slot
}

test "hcPostPartial takes prompt calls of 4 streams of 4096 on 16-byte rows only, before any launch" {
    const set: Set = .{ .glue = undefined };
    const s: cuda.Stream = undefined;
    try std.testing.expectError(error.HcFuseRows, hcPostPartial(set, s, 0, 0, 0, 0, 0, 0, 0, hc_prompt_rows - 1, 4096, 4, 2));
    try std.testing.expectError(error.HcShape, hcPostPartial(set, s, 0, 0, 0, 0, 0, 0, 0, hc_prompt_rows, 2048, 4, 2));
    try std.testing.expectError(error.RowsUnaligned, hcPostPartial(set, s, 8, 8, 0, 0, 0, 0, 0, hc_prompt_rows, 4096, 4, 2));
}
