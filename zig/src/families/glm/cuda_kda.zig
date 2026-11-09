//! GLM-5.3-Flash's KDA on CUDA (the Python family's kda): fused and three-kernel chains match bits; replay rebuilds.

const std = @import("std");
const cuda = @import("cuda");

pub const dk = 128;
pub const dv = 128;
/// Windows of this many rows or more (prefill chunks) take the three-kernel chain (kda.WIDE_ROWS).
pub const wide_rows = 64;

/// glm_kda.cu's entries.
pub const Kernels = struct {
    chain: cuda.Function,
    replay: cuda.Function,
    replay_layers: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    out: cuda.Function,
    prep_segs: cuda.Function, // a shared window's segments in one launch a kernel (chainSegs)
    step_segs: cuda.Function,
    out_segs: cuda.Function,
    segs: bool = true, // TF_GLM_KDA_SEGS=0: a shared window's segments take their own launches (the same bits)

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (.{ "chain", "replay", "replay_layers", "prep", "step", "out", "prep_segs", "step_segs", "out_segs" }) |name| @field(k, name) = try m.function("tf_glm_kda_" ++ name);
        k.segs = try @import("sys").environment.read(bool, "TF_GLM_KDA_SEGS", true); // once, local to the rank
        return k;
    }
};

/// The most segments one chainSegs launch takes (glm_kda.cu's KDA_SEGS: Engine.max_shared).
pub const max_segs = 16;

/// One segment of a shared window's KDA layer, as glm_kda.cu's KdaSeg: its rows, its stream's states and saves.
pub const Seg = extern struct { p: u64, a: u64, g: u64, cs: u64, state_in: u64, state_out: u64, q: u64, k: u64, v: u64, gs: u64, b: u64, y: u64, out: u64, rows: c_int, pad: c_int = 0 };
pub const Segs = extern struct { s: [max_segs]Seg };

/// Whether a window's `n` segments take chainSegs: decode rows on the three-kernel chain, two to max_segs of them.
pub fn useSegs(prefill: bool, wide: bool, on: bool, n: usize) bool {
    return !prefill and wide and on and n >= 2 and n <= max_segs;
}

/// A segment's table row: what its own `chain` (wide) launch takes, its q and y at `row0` of the window's scratch.
pub fn seg(c: Chain, s: Saves, w: Wide, row0: usize, rows: usize) Seg {
    const n = c.heads * dk;
    return .{ .p = c.p, .a = c.a, .g = c.g, .cs = c.conv_state, .state_in = c.state_in, .state_out = s.state_out, .q = w.q + row0 * n * 4, .k = s.k, .v = s.v, .gs = s.g, .b = s.b, .y = w.y + row0 * n * 2, .out = s.out, .rows = @intCast(rows) };
}

/// The three-kernel chain, all `t`'s segments in one launch each (z): each `chain`'s blocks; `c`: the layer's fields.
pub fn chainSegs(ks: *const Kernels, stream: cuda.Stream, c: Chain, t: *const Segs, n: usize) !void {
    if (n == 0 or n > max_segs) return error.TooManySegments;
    var most: usize = 0;
    for (t.s[0..n]) |g| most = @max(most, @as(usize, @intCast(g.rows)));
    if (most == 0) return;
    const h: c_int = @intCast(c.heads);
    const z: u32 = @intCast(n);
    var a: cuda.Args = .{};
    a.add(h);
    a.add(@as(c_int, @intCast(c.p_stride)));
    a.add(@as(c_int, @intCast(c.b_off)));
    a.add(@as(c_int, @intCast(c.a_stride)));
    for ([_]u64{ c.conv_w, c.a_log, c.dt_bias }) |v| a.add(v);
    a.add(c.lower);
    a.add(t.*);
    try cuda.launch.launch(ks.prep_segs, .{ .grid = .{ .x = @intCast(most), .y = @intCast(c.heads), .z = z }, .block = .{ .x = 512 } }, stream, &a);
    var b: cuda.Args = .{};
    b.add(h);
    b.add(t.*);
    try cuda.launch.launch(ks.step_segs, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / 8, .z = z }, .block = .{ .x = 8 * 32 } }, stream, &b);
    var o: cuda.Args = .{};
    o.add(h);
    o.add(@as(c_int, @intCast(c.g_stride)));
    o.add(c.norm_w);
    o.add(c.eps);
    o.add(t.*);
    try cuda.launch.launch(ks.out_segs, .{ .grid = .{ .x = @intCast(most), .y = @intCast(c.heads), .z = z }, .block = .{ .x = dv } }, stream, &o);
}

/// One layer's chain inputs, `heads` heads: p rows [q|k|v|f_a|g_a|...] (beta at `b_off`), decay and gate rows, states.
pub const Chain = struct {
    p: u64,
    p_stride: usize,
    b_off: usize,
    a: u64,
    a_stride: usize,
    g: u64,
    g_stride: usize,
    conv_state: u64,
    conv_w: u64,
    state_in: u64,
    a_log: u64,
    dt_bias: u64,
    norm_w: u64,
    eps: f32,
    lower: f32,
    heads: usize,
};

/// Where a chain writes: out bf16, state_out like state_in, and replay's rows: k fp32, v bf16, decay and beta fp32.
pub const Saves = struct { out: u64, state_out: u64, k: u64, v: u64, g: u64, b: u64 };

/// The three-kernel chain's scratch: q fp32 and the read-out bf16, [rows][heads][128] each.
pub const Wide = struct { q: u64, y: u64 };

/// kda.chain: `rows` rows through the fused kernel, or with `wide` through prep, step and out (WARPS 8, TR 16).
pub fn chain(ks: *const Kernels, stream: cuda.Stream, c: Chain, rows: usize, s: Saves, wide: ?Wide) !void {
    const h: c_int = @intCast(c.heads);
    if (wide) |w| {
        var a: cuda.Args = .{};
        a.add(h);
        a.add(c.p);
        a.add(@as(c_int, @intCast(c.p_stride)));
        a.add(@as(c_int, @intCast(c.b_off)));
        a.add(c.a);
        a.add(@as(c_int, @intCast(c.a_stride)));
        for ([_]u64{ c.conv_state, c.conv_w, c.a_log, c.dt_bias }) |v| a.add(v);
        a.add(c.lower);
        for ([_]u64{ w.q, s.k, s.v, s.g, s.b }) |v| a.add(v);
        const grid: cuda.launch.Dim3 = .{ .x = @intCast(rows), .y = @intCast(c.heads) };
        try cuda.launch.launch(ks.prep, .{ .grid = grid, .block = .{ .x = 512 } }, stream, &a);
        var b: cuda.Args = .{};
        b.add(h);
        for ([_]u64{ c.state_in, w.q, s.k, s.v, s.g, s.b }) |v| b.add(v);
        b.add(@as(c_int, @intCast(rows)));
        b.add(w.y);
        b.add(s.state_out);
        try cuda.launch.launch(ks.step, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / 8 }, .block = .{ .x = 8 * 32 } }, stream, &b);
        var o: cuda.Args = .{};
        o.add(h);
        o.add(w.y);
        o.add(c.g);
        o.add(@as(c_int, @intCast(c.g_stride)));
        o.add(c.norm_w);
        o.add(c.eps);
        o.add(s.out);
        return cuda.launch.launch(ks.out, .{ .grid = grid, .block = .{ .x = dv } }, stream, &o);
    }
    var a: cuda.Args = .{};
    a.add(h);
    a.add(c.p);
    a.add(@as(c_int, @intCast(c.p_stride)));
    a.add(@as(c_int, @intCast(c.b_off)));
    a.add(c.a);
    a.add(@as(c_int, @intCast(c.a_stride)));
    a.add(c.g);
    a.add(@as(c_int, @intCast(c.g_stride)));
    for ([_]u64{ c.conv_state, c.conv_w, c.state_in, c.a_log, c.dt_bias, c.norm_w }) |v| a.add(v);
    a.add(c.eps);
    a.add(c.lower);
    a.add(@as(c_int, @intCast(rows)));
    for ([_]u64{ s.out, s.state_out, s.k, s.v, s.g, s.b }) |v| a.add(v);
    try cuda.launch.launch(ks.chain, .{ .grid = .{ .x = @intCast(c.heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

/// kda.replay: the state after the first `rows` saved rows, from state_in.
pub fn replay(ks: *const Kernels, stream: cuda.Stream, state_in: u64, k: u64, v: u64, g: u64, b: u64, rows: usize, heads: usize, state_out: u64) !void {
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(heads)));
    for ([_]u64{ state_in, k, v, g, b }) |x| a.add(x);
    a.add(@as(c_int, @intCast(rows)));
    a.add(state_out);
    try cuda.launch.launch(ks.replay, .{ .grid = .{ .x = @intCast(heads) }, .block = .{ .x = 1024 } }, stream, &a);
}

test "the three-kernel chain's step grid is the Python family's" {
    try std.testing.expectEqual(@as(u32, 4), dv / 4 / 8); // DV / 4 / WARPS value rows a block, 8 warps of 4 rows
}

test "GLM KDA segs: the table is glm_kda.cu's KdaSegs and fits one launch's arguments" {
    try std.testing.expectEqual(@as(usize, 112), @sizeOf(Seg));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(Seg, "rows"));
    try std.testing.expectEqual(@as(usize, max_segs * 112), @sizeOf(Segs));
    try std.testing.expect(48 + @sizeOf(Segs) <= cuda.launch.Args.max_bytes); // prep's arguments, the most
}

test "GLM KDA segs: each segment keeps its own launch's pointers, its q and y rows apart in the window's scratch" {
    const heads = 32;
    const n = heads * dk;
    const rows = [_]usize{ 3, 8, 1, 5 };
    const w: Wide = .{ .q = 1 << 32, .y = 1 << 33 };
    var row0: usize = 0;
    var t: Segs = undefined;
    for (rows, 0..) |r, i| {
        const at: u64 = @as(u64, i + 1) << 40;
        const c: Chain = .{ .p = at, .p_stride = 0, .b_off = 0, .a = at + 1, .a_stride = n, .g = at + 2, .g_stride = n, .conv_state = at + 3, .conv_w = 0, .state_in = at + 4, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 0, .lower = 0, .heads = heads };
        const s: Saves = .{ .out = at + 5, .state_out = at + 6, .k = at + 7, .v = at + 8, .g = at + 9, .b = at + 10 };
        t.s[i] = seg(c, s, w, row0, r);
        const g = t.s[i];
        try std.testing.expectEqualSlices(u64, &.{ c.p, c.a, c.g, c.conv_state, c.state_in, s.state_out, s.k, s.v, s.g, s.b, s.out }, &.{ g.p, g.a, g.g, g.cs, g.state_in, g.state_out, g.k, g.v, g.gs, g.b, g.out });
        try std.testing.expectEqual(@as(c_int, @intCast(r)), g.rows);
        try std.testing.expectEqual(w.q + row0 * n * 4, g.q);
        try std.testing.expectEqual(w.y + row0 * n * 2, g.y);
        row0 += r;
    }
    // [q, q + rows n 4) and [y, y + rows n 2): one after another, together the window's rows (Buffers' wide q and y)
    for (t.s[1..rows.len], t.s[0 .. rows.len - 1], rows[0 .. rows.len - 1]) |g, was, r| {
        try std.testing.expectEqual(was.q + r * n * 4, g.q);
        try std.testing.expectEqual(was.y + r * n * 2, g.y);
    }
    try std.testing.expectEqual(w.q + row0 * n * 4, t.s[rows.len - 1].q + rows[rows.len - 1] * n * 4);
}

test "GLM KDA segs: only decode windows of two to sixteen segments on the three-kernel chain batch" {
    try std.testing.expect(useSegs(false, true, true, 2));
    try std.testing.expect(useSegs(false, true, true, max_segs));
    try std.testing.expect(!useSegs(false, true, true, 1)); // a lone segment keeps its own launches
    try std.testing.expect(!useSegs(false, true, true, max_segs + 1));
    try std.testing.expect(!useSegs(true, true, true, 4)); // prompts: whole prompts, the layer committed per segment
    try std.testing.expect(!useSegs(false, false, true, 4)); // TF_GLM_KDA_DECODE_WIDE=0: the fused chain per segment
    try std.testing.expect(!useSegs(false, true, false, 4)); // TF_GLM_KDA_SEGS=0
    const env = @import("sys").environment;
    try std.testing.expect(try env.value(bool, "TF_GLM_KDA_SEGS", null, true));
    try std.testing.expect(try env.value(bool, "TF_GLM_KDA_SEGS", "1", true));
    try std.testing.expect(!try env.value(bool, "TF_GLM_KDA_SEGS", "0", true));
    try std.testing.expectError(error.InvalidEnvironment, env.value(bool, "TF_GLM_KDA_SEGS", "off", true));
}
