//! Shared CUDA KDA: 128 key/value channels, four conv taps, BF16 rows and FP32 state and sums.

const std = @import("std");
const cuda = @import("root.zig");

/// Key and value channels a head (kda.cuh DK, DV); zig/kda_header_host.zig holds these and taps to the header.
pub const dk = 128;
pub const dv = 128;
/// The conv window's taps (kda.cuh TAPS): three kept rows and the current one.
pub const taps = 4;
/// Windows of this many rows or more (prefill chunks) take the three-kernel chain (kda.WIDE_ROWS).
pub const wide_rows = 64;

/// A model's KDA shape against the one the kernels are built for, refused before anything is allocated or launched.
pub fn geometry(key_dim: usize, value_dim: usize, conv_taps: usize) error{KdaGeometry}!void {
    if (key_dim != dk or value_dim != dv or conv_taps != taps) return error.KdaGeometry;
}

/// glm_kda.cu's entries.
pub const Kernels = struct {
    chain: cuda.Function,
    replay: cuda.Function,
    replay_layers: cuda.Function,
    prep: cuda.Function,
    step: cuda.Function,
    step4: ?cuda.Function = null, // 4- and 2-warp step variants: same bits, more blocks for small windows
    step2: ?cuda.Function = null,
    out: cuda.Function,
    pdl: bool = false, // dependent launches (each entry opens with glm_pdl::enter()); the loader's caller opts in
    step_warps: u32 = 0, // TF_GLM_KDA_STEP_WARPS: 0 (auto) by rows (stepWarps); 8, 4 or 2 at every row count

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = .{
            .chain = undefined,
            .replay = undefined,
            .replay_layers = undefined,
            .prep = undefined,
            .step = undefined,
            .out = undefined,
        };
        inline for (.{ "chain", "replay", "replay_layers", "prep", "step", "out" }) |name| @field(k, name) = try m.function("tf_glm_kda_" ++ name);
        k.step4 = try m.optionalFunction("tf_glm_kda_step4");
        k.step2 = try m.optionalFunction("tf_glm_kda_step2");
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

/// Selected step entry and warp count for `rows`, falling back to 8-warp `step` on older fatbins.
pub fn stepChoice(ks: *const Kernels, rows: usize) struct { f: cuda.Function, warps: u32 } {
    const wp: u32 = if (ks.step_warps != 0) ks.step_warps else stepWarps(rows);
    return switch (wp) {
        2 => if (ks.step2) |f| .{ .f = f, .warps = 2 } else .{ .f = ks.step, .warps = 8 },
        4 => if (ks.step4) |f| .{ .f = f, .warps = 4 } else .{ .f = ks.step, .warps = 8 },
        else => .{ .f = ks.step, .warps = 8 },
    };
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
    try validate(c, rows);
    const h: c_int = @intCast(c.heads);
    if (wide) |w| {
        const pdl = ks.pdl and rows < wide_rows; // a prompt chunk's chain launches plain: dependent, its first token comes later
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
        try cuda.launch.launch(ks.prep, .{ .grid = grid, .block = .{ .x = 512 }, .pdl = pdl }, stream, &a);
        var b: cuda.Args = .{};
        b.add(h);
        for ([_]u64{ c.state_in, w.q, s.k, s.v, s.g, s.b }) |v| b.add(v);
        b.add(@as(c_int, @intCast(rows)));
        b.add(w.y);
        b.add(s.state_out);
        const st = stepChoice(ks, rows);
        try cuda.launch.launch(st.f, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / st.warps }, .block = .{ .x = st.warps * 32 }, .pdl = pdl }, stream, &b);
        var o: cuda.Args = .{};
        o.add(h);
        o.add(w.y);
        o.add(c.g);
        o.add(@as(c_int, @intCast(c.g_stride)));
        o.add(c.norm_w);
        o.add(c.eps);
        o.add(s.out);
        return cuda.launch.launch(ks.out, .{ .grid = grid, .block = .{ .x = dv }, .pdl = pdl }, stream, &o);
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
    try cuda.launch.launch(ks.chain, .{ .grid = .{ .x = @intCast(c.heads) }, .block = .{ .x = 1024 }, .pdl = ks.pdl }, stream, &a);
}

/// kda.replay: the state after the first `rows` saved rows, from state_in.
pub fn replay(ks: *const Kernels, stream: cuda.Stream, state_in: u64, k: u64, v: u64, g: u64, b: u64, rows: usize, heads: usize, state_out: u64) !void {
    if (heads == 0 or heads > std.math.maxInt(c_int) or rows > std.math.maxInt(c_int)) return error.KdaShape;
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(heads)));
    for ([_]u64{ state_in, k, v, g, b }) |x| a.add(x);
    a.add(@as(c_int, @intCast(rows)));
    a.add(state_out);
    try cuda.launch.launch(ks.replay, .{ .grid = .{ .x = @intCast(heads) }, .block = .{ .x = 1024 }, .pdl = ks.pdl }, stream, &a);
}

/// The fixed kernel envelope; strides are elements, never bytes. Refuse before narrowing or launching.
fn validate(c: Chain, rows: usize) !void {
    const limit = std.math.maxInt(c_int);
    if (c.heads == 0 or c.heads > limit / (3 * dk) or rows == 0 or rows > limit) return error.KdaShape;
    const n = c.heads * dk;
    if (c.p_stride > limit or c.a_stride > limit or c.g_stride > limit or c.b_off > limit) return error.KdaShape;
    if (c.b_off < 3 * n or c.b_off + c.heads > c.p_stride or c.a_stride < n or c.g_stride < n) return error.KdaShape;
}

/// Batched replay strides are FP32/BF16 elements as declared by the original kernel, not byte offsets.
pub const Replay = struct {
    state_in: u64,
    state_stride: usize,
    k: u64,
    v: u64,
    g: u64,
    b: u64,
    kv_stride: usize,
    b_stride: usize,
    state_out: u64,
};

/// Restore each layer after a rejected suffix, preserving the saved-row order and the original launch ABI.
pub fn replayLayers(ks: *const Kernels, stream: cuda.Stream, s: Replay, rows: usize, heads: usize, layers: usize) !void {
    if (heads == 0 or heads > std.math.maxInt(c_int) or rows > std.math.maxInt(c_int) or
        layers == 0 or layers > std.math.maxInt(u32) / heads) return error.KdaShape;
    var a: cuda.Args = .{};
    a.add(@as(c_int, @intCast(heads)));
    a.add(s.state_in);
    a.add(s.state_stride);
    for ([_]u64{ s.k, s.v, s.g, s.b }) |x| a.add(x);
    a.add(s.kv_stride);
    a.add(s.b_stride);
    a.add(@as(c_int, @intCast(rows)));
    a.add(s.state_out);
    try cuda.launch.launch(ks.replay_layers, .{ .grid = .{ .x = @intCast(layers * heads) }, .block = .{ .x = 1024 }, .pdl = ks.pdl }, stream, &a);
}

test "shared KDA: step warps dispatch by rows, grids cover every value row, and fallback holds on older fatbins" {
    try std.testing.expectEqual(@as(u32, 2), stepWarps(1));
    for ([_]usize{ 2, 3, 4, 8 }) |r| try std.testing.expectEqual(@as(u32, 4), stepWarps(r));
    for ([_]usize{ 9, 16, 32, 64, 2048 }) |r| try std.testing.expectEqual(@as(u32, 8), stepWarps(r));
    for ([_]u32{ 2, 4, 8 }) |wp| try std.testing.expectEqual(@as(u32, dv), dv / 4 / wp * wp * 4);
    try std.testing.expectEqual([4]u32{ 0, 8, 4, 2 }, .{ stepWarpsSetting(0), stepWarpsSetting(1), stepWarpsSetting(2), stepWarpsSetting(3) });
    var ks: Kernels = .{
        .chain = .{ .d = undefined, .handle = @ptrFromInt(1) },
        .replay = .{ .d = undefined, .handle = @ptrFromInt(2) },
        .replay_layers = .{ .d = undefined, .handle = @ptrFromInt(3) },
        .prep = .{ .d = undefined, .handle = @ptrFromInt(4) },
        .step = .{ .d = undefined, .handle = @ptrFromInt(5) },
        .step4 = .{ .d = undefined, .handle = @ptrFromInt(7) },
        .step2 = .{ .d = undefined, .handle = @ptrFromInt(8) },
        .out = .{ .d = undefined, .handle = @ptrFromInt(6) },
    };
    try std.testing.expectEqual(@as(u32, 2), stepChoice(&ks, 1).warps);
    try std.testing.expectEqual(@as(usize, 8), @intFromPtr(stepChoice(&ks, 1).f.handle));
    try std.testing.expectEqual(@as(u32, 4), stepChoice(&ks, 4).warps);
    try std.testing.expectEqual(@as(usize, 7), @intFromPtr(stepChoice(&ks, 4).f.handle));
    try std.testing.expectEqual(@as(u32, 8), stepChoice(&ks, 9).warps);
    try std.testing.expectEqual(@as(usize, 5), @intFromPtr(stepChoice(&ks, 9).f.handle));
    ks.step_warps = 8;
    try std.testing.expectEqual(@as(u32, 8), stepChoice(&ks, 1).warps);
    ks.step_warps = 2;
    try std.testing.expectEqual(@as(u32, 2), stepChoice(&ks, 64).warps);
    ks.step2 = null;
    try std.testing.expectEqual(@as(u32, 8), stepChoice(&ks, 1).warps);
}

test "shared KDA: wide chain launches step2, step4 and step with matching grids, blocks and PDL" {
    const Rec = struct {
        var step_id: usize = 0;
        var grid_y: u32 = 0;
        var block_x: u32 = 0;
        var pdl_count: usize = 0;
        var total: usize = 0;
        fn plain(f: cuda.abi.Function, _: u32, gy: u32, _: u32, bx: u32, _: u32, _: u32, _: u32, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (total == 1) {
                step_id = @intFromPtr(f);
                grid_y = gy;
                block_x = bx;
            }
            total += 1;
            return 0;
        }
        fn ex(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (total == 1) {
                step_id = @intFromPtr(f);
                grid_y = c.grid_y;
                block_x = c.block_x;
            }
            for (c.attrs.?[0..c.num_attrs]) |a| if (a.id == .programmatic_stream_serialization and a.value.int == 1) {
                pdl_count += 1;
            };
            total += 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Rec.plain;
    d.api.cuLaunchKernelEx = Rec.ex;
    const stream: cuda.Stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    const ks: Kernels = .{
        .chain = .{ .d = &d, .handle = @ptrFromInt(1) },
        .replay = .{ .d = &d, .handle = @ptrFromInt(2) },
        .replay_layers = .{ .d = &d, .handle = @ptrFromInt(3) },
        .prep = .{ .d = &d, .handle = @ptrFromInt(4) },
        .step = .{ .d = &d, .handle = @ptrFromInt(5) },
        .step4 = .{ .d = &d, .handle = @ptrFromInt(7) },
        .step2 = .{ .d = &d, .handle = @ptrFromInt(8) },
        .out = .{ .d = &d, .handle = @ptrFromInt(6) },
        .pdl = true,
    };
    const heads: usize = 32;
    const n = heads * dk;
    const ch: Chain = .{ .heads = heads, .p = 0x1000, .p_stride = 3 * n + heads, .b_off = 3 * n, .a = 0x2000, .a_stride = n, .g = 0x3000, .g_stride = n, .conv_state = 0x4000, .conv_w = 0x5000, .state_in = 0x6000, .a_log = 0x7000, .dt_bias = 0x8000, .norm_w = 0x9000, .eps = 1e-6, .lower = -5 };
    const sv: Saves = .{ .out = 0xa000, .state_out = 0xb000, .k = 0xc000, .v = 0xd000, .g = 0xe000, .b = 0xf000 };
    const w: Wide = .{ .q = 0x10000, .y = 0x11000 };
    for ([_]struct { usize, usize, u32, u32, usize }{
        .{ 1, 8, 16, 64, 3 },
        .{ 4, 7, 8, 128, 3 },
        .{ 16, 5, 4, 256, 3 },
        .{ 64, 5, 4, 256, 0 },
    }) |case| {
        Rec.total = 0;
        Rec.pdl_count = 0;
        try chain(&ks, stream, ch, case[0], sv, w);
        try std.testing.expectEqual(@as(usize, 3), Rec.total);
        try std.testing.expectEqual(case[1], Rec.step_id);
        try std.testing.expectEqual(case[2], Rec.grid_y);
        try std.testing.expectEqual(case[3], Rec.block_x);
        try std.testing.expectEqual(case[4], Rec.pdl_count);
    }
}

test {
    _ = @import("kda_test.zig");
}
