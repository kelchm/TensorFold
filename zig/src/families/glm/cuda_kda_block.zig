//! GLM-5.3-Flash's KDA block on CUDA, a rank's share of the Python family's kda_block: projection, chain, o_proj.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const prof = @import("cuda_prof.zig");
const l2pf = @import("cuda_l2pf.zig");
const projection = @import("cuda_proj.zig");

/// The input projection [q | k | v | f_a | g_a | b] [3 heads 128 + 256 + heads][dims]: one bf16 stack, or two parts.
pub const In = union(enum) {
    bf16: u64,
    split: struct { qkv: cuda.qmmf.Weight, rest: u64 }, // MXFP8 [q | k | v] for the lane (FP4-Local's), bf16 the rest
};

/// One KDA layer, `heads` heads: its input projection, f_b, g_b, o (bf16 or MXFP8), the chain's taps and norm.
pub const Layer = struct {
    proj: In,
    fb: u64,
    gb: u64,
    o: projection.Proj,
    conv_w: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    heads: usize,
    dims: usize,

    pub fn width(l: Layer) usize {
        return 3 * l.heads * kda.dk + 2 * kda.dk + l.heads;
    }

    /// The width the input stack's rows actually carry (TF_GLM_KDA_PAD): the COMMON width, so every rank agrees.
    pub fn matWidth(l: Layer) usize {
        return rowWidth(l.heads);
    }

    /// The input stack and its two views need glm_mm's eight-column output and row stride.
    pub fn tiledInput(l: Layer) bool {
        return l.matWidth() % 8 == 0;
    }
};

/// TF_GLM_KDA_PAD's one eight-aligned row width for every rank, set once at load (Engine.open); 0: no pad.
pub var pad_to: usize = 0;

/// The pad width for `kda_heads` split over `world` ranks: the widest rank's row, rounded up to eight columns.
pub fn padWidth(kda_heads: usize, world: usize) usize {
    const heads = (kda_heads + world - 1) / world;
    return std.mem.alignForward(usize, 3 * heads * kda.dk + 2 * kda.dk + heads, 8);
}

/// The fused input's row width on a rank of `own_heads` heads: pad_to when the pad is on.
pub fn rowWidth(own_heads: usize) usize {
    if (pad_to != 0) return pad_to;
    return 3 * own_heads * kda.dk + 2 * kda.dk + own_heads;
}

/// The block's scratch for `rows` rows: projection, decay and gate rows, chain saves, B16 partials, a split's tmp.
pub const Scratch = struct { p: u64, ka: u64, kg: u64, saves: kda.Saves, wide: kda.Wide, mm_part: u64, tmp: u64 = 0 };

/// forward.kda_block: out [rows][dims] fp32 partial; `wide`: three-kernel chain; `prompt`: commits the conv window.
pub fn block(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, out: u64) !void {
    try body(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt);
    try fill(set, stream, l, s, 0, rows, out);
}

/// forward.out_proj's fill for the KDA block: o_proj of rows lo..hi (s.saves.out) into out [hi - lo][dims] fp32.
pub fn fill(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, lo: usize, hi: usize, out: u64) !void {
    const n = l.heads * kda.dk;
    try projection.matmul(set, stream, l.o, s.saves.out + lo * n * 2, n, out, s.mm_part, hi - lo, l.dims, n, true);
}

/// The block up to its out_proj: the projections, the chain, and a prompt chunk's conv window commit.
pub fn body(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool) !void {
    return bodyProfile(set, ks, stream, l, s, x, x_stride, rows, conv_state, state_in, wide, prompt, null, null);
}

pub fn bodyProfile(set: glue.Set, ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, conv_state: u64, state_in: u64, wide: bool, prompt: bool, profile: ?*prof.Prof, o_site: ?l2pf.At) !void {
    var stamp = prof.span(profile, stream, .projections);
    try project(set, stream, l, s, x, x_stride, rows);
    stamp.end();
    if (o_site) |at| try at.fork(stream); // TF_GLM_L2PF: o read into L2 while the chain runs
    stamp = prof.span(profile, stream, .kda);
    defer stamp.end();
    try chainRows(ks, stream, l, s, rows, conv_state, state_in, wide);
    if (prompt) try glue.convShift(set, stream, conv_state, s.p, rows, rows * l.matWidth(), l.matWidth(), 1, 3 * l.heads * kda.dk);
}

/// The projections of `rows` rows into s.p, s.ka (decay) and s.kg (gate), row by row so segments share one call.
pub fn project(set: glue.Set, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize) !void {
    var selected = set;
    if (!l.tiledInput()) selected.mm = null;
    const n = l.heads * kda.dk;
    const w = l.matWidth();
    const fa = 3 * n;
    switch (l.proj) {
        .bf16 => |p| try b16.matmul(selected, stream, x, x_stride, p, s.p, s.mm_part, rows, w, l.dims, false),
        .split => |sp| { // each part on its own kernel into its columns of s.p: one row straight there, more through tmp
            // cc-defer: two row copies a window; an output row stride on the lane and glm_mm if a profile flags them
            const one = rows == 1;
            try projection.matmul(set, stream, .{ .mxfp8 = sp.qkv }, x, x_stride, if (one) s.p else s.tmp, s.mm_part, rows, fa, l.dims, false);
            if (!one) try columns(set, stream, s.tmp, fa, s.p, w, rows);
            // the bf16 rest (2 * dk + heads wide) tiles with the stack, or stays row-major with it when uneven
            try b16.matmul(selected, stream, x, x_stride, sp.rest, if (one) s.p + fa * 2 else s.tmp, s.mm_part, rows, w - fa, l.dims, false);
            if (!one) try columns(set, stream, s.tmp, w - fa, s.p + fa * 2, w, rows);
        },
    }
    try b16.matmul(selected, stream, s.p + fa * 2, w, l.fb, s.ka, s.mm_part, rows, n, kda.dk, false);
    try b16.matmul(selected, stream, s.p + (fa + kda.dk) * 2, w, l.gb, s.kg, s.mm_part, rows, n, kda.dk, false);
}

/// `rows` rows of `cols` bf16 packed at `src` into `dst`, rows `stride` apart (tf_glm_copy_rows).
fn columns(set: glue.Set, stream: cuda.Stream, src: u64, cols: usize, dst: u64, stride: usize, rows: usize) !void {
    const f = set.copy_rows orelse return error.FormatNotLoaded;
    var a: cuda.Args = .{};
    a.add(src);
    a.add(@as(c_longlong, @intCast(cols * 2)));
    a.add(dst);
    a.add(@as(c_longlong, @intCast(stride * 2)));
    a.add(@as(c_int, @intCast(rows)));
    a.add(@as(c_int, @intCast(cols * 2)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((rows * cols * 2 + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
}

/// The chain over `rows` projected rows (s.p, s.ka, s.kg) on one sequence's conv window and recurrent state.
pub fn chainRows(ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, s: Scratch, rows: usize, conv_state: u64, state_in: u64, wide: bool) !void {
    try kda.chain(ks, stream, chainOf(l, s, conv_state, state_in), rows, s.saves, if (wide or rows >= kda.wide_rows) s.wide else null);
}

/// chainRows' chain: the layer's fields (TF_GLM_KDA_PAD's row width), s's rows and the sequence's conv and state.
fn chainOf(l: Layer, s: Scratch, conv_state: u64, state_in: u64) kda.Chain {
    const n = l.heads * kda.dk;
    const fa = 3 * n;
    return .{ .p = s.p, .p_stride = l.matWidth(), .b_off = fa + 2 * kda.dk, .a = s.ka, .a_stride = n, .g = s.kg, .g_stride = n, .conv_state = conv_state, .conv_w = l.conv_w, .state_in = state_in, .a_log = l.a_log, .dt_bias = l.dt_bias, .norm_w = l.norm_w, .eps = l.eps, .lower = l.lower, .heads = l.heads };
}

/// A shared window's segment as chainRows (wide) would run it, for chainSegs: its q and y rows at `row0` of s.wide.
pub fn segOf(l: Layer, s: Scratch, rows: usize, conv_state: u64, state_in: u64, row0: usize) kda.Seg {
    return kda.seg(chainOf(l, s, conv_state, state_in), s.saves, s.wide, row0, rows);
}

/// A shared window's chains (decode, three-kernel): every segment of `t` in one launch a kernel (kda.chainSegs).
pub fn chainSegs(ks: *const kda.Kernels, stream: cuda.Stream, l: Layer, t: *const kda.Segs, n: usize) !void {
    try kda.chainSegs(ks, stream, chainOf(l, std.mem.zeroes(Scratch), 0, 0), t, n);
}

test {
    _ = @import("cuda_kda_tp3_test.zig");
}

test "GLM KDA segs: a segment's row and the shared fields are what chainRows passes, padded rows included" {
    const was = pad_to;
    defer pad_to = was;
    pad_to = padWidth(64, 3);
    const l: Layer = .{ .proj = .{ .bf16 = 0 }, .fb = 0, .gb = 0, .o = .{ .bf16 = 0 }, .conv_w = 11, .a_log = 12, .dt_bias = 13, .norm_w = 14, .eps = 1e-6, .lower = -5, .heads = 22, .dims = 6144 };
    const s: Scratch = .{ .p = 1, .ka = 2, .kg = 3, .saves = .{ .out = 4, .state_out = 5, .k = 6, .v = 7, .g = 8, .b = 9 }, .wide = .{ .q = 1 << 30, .y = 1 << 31 }, .mm_part = 0 };
    const c = chainOf(l, s, 21, 22);
    const g = segOf(l, s, 5, 21, 22, 3);
    try std.testing.expectEqualSlices(u64, &.{ c.p, c.a, c.g, c.conv_state, c.state_in, s.saves.state_out, s.saves.out }, &.{ g.p, g.a, g.g, g.cs, g.state_in, g.state_out, g.out });
    try std.testing.expectEqual(s.wide.q + 3 * 22 * kda.dk * 4, g.q);
    try std.testing.expectEqual(s.wide.y + 3 * 22 * kda.dk * 2, g.y);
    const shared = chainOf(l, std.mem.zeroes(Scratch), 0, 0); // chainSegs' layer fields
    try std.testing.expectEqual(pad_to, shared.p_stride);
    try std.testing.expect(shared.p_stride != l.width());
    inline for (.{ "p_stride", "b_off", "a_stride", "g_stride", "conv_w", "a_log", "dt_bias", "norm_w", "eps", "lower", "heads" }) |f|
        try std.testing.expectEqual(@field(c, f), @field(shared, f));
}
