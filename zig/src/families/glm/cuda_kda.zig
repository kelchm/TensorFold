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
    step4: cuda.Function, // the step with 4 and 2 warps a block: the same bits, more blocks for small windows
    step2: cuda.Function,
    out: cuda.Function,
    step_warps: u32 = 0, // TF_GLM_KDA_STEP_WARPS: 0 (auto) by rows (stepWarps); 8, 4 or 2 at every row count

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (.{ "chain", "replay", "replay_layers", "prep", "step", "step4", "step2", "out" }) |name| @field(k, name) = try m.function("tf_glm_kda_" ++ name);
        k.step_warps = stepWarpsSetting(@import("sys").environment.read(usize, "TF_GLM_KDA_STEP_WARPS", 0) catch 0);
        return k;
    }
};

/// Warps a step block takes by the window's rows: 2 for one row, 4 to 8 rows, 8 past that and for prompt chunks.
pub fn stepWarps(rows: usize) u32 {
    return if (rows == 1) 2 else if (rows <= 8) 4 else 8;
}

/// TF_GLM_KDA_STEP_WARPS' word (auto, 8, 4, 2) as warps; 0 is auto. No bit depends on it.
pub fn stepWarpsSetting(word: usize) u32 {
    return ([_]u32{ 0, 8, 4, 2 })[@min(word, 3)];
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

/// kda.chain: `rows` rows through the fused kernel, or with `wide` through prep, step (by stepWarps, TR 16) and out.
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
        const wp: u32 = if (ks.step_warps != 0) ks.step_warps else stepWarps(rows);
        const f = switch (wp) {
            2 => ks.step2,
            4 => ks.step4,
            else => ks.step,
        };
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / wp }, .block = .{ .x = wp * 32 } }, stream, &b);
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

test "a step's warps by rows: 2 for one row, 4 to 8 rows, 8 past that; every grid covers the value rows" {
    try std.testing.expectEqual(@as(u32, 2), stepWarps(1));
    for ([_]usize{ 2, 3, 4, 8 }) |r| try std.testing.expectEqual(@as(u32, 4), stepWarps(r));
    for ([_]usize{ 9, 16, 32, 64, 2048 }) |r| try std.testing.expectEqual(@as(u32, 8), stepWarps(r));
    for ([_]u32{ 2, 4, 8 }) |wp| try std.testing.expectEqual(@as(u32, dv), dv / 4 / wp * wp * 4);
    try std.testing.expectEqual([4]u32{ 0, 8, 4, 2 }, .{ stepWarpsSetting(0), stepWarpsSetting(1), stepWarpsSetting(2), stepWarpsSetting(3) });
    try std.testing.expectEqualSlices([]const u8, &.{ "auto", "8", "4", "2" }, @import("sys").environment.rule("TF_GLM_KDA_STEP_WARPS").words);
}
