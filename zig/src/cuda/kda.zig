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
    prep_segs: ?cuda.Function = null, // a shared window's segments in one launch a kernel (chainSegs)
    step_segs: ?cuda.Function = null,
    step4_segs: ?cuda.Function = null,
    step2_segs: ?cuda.Function = null,
    out_segs: ?cuda.Function = null,
    copy_segs: ?cuda.Function = null, // and the segments' projection rows into their streams (copySegs)
    pdl: bool = false, // dependent launches (each entry opens with glm_pdl::enter()); the loader's caller opts in
    step_warps: u32 = 0, // TF_GLM_KDA_STEP_WARPS: 0 (auto) by rows (stepWarps); 8, 4 or 2 at every row count
    segs: bool = false, // TF_GLM_KDA_SEGS: GLM opts in (default true); legacy callers stay unchanged

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
        inline for (.{ "step4", "step2", "prep_segs", "step_segs", "step4_segs", "step2_segs", "out_segs", "copy_segs" }) |name|
            @field(k, name) = try m.optionalFunction("tf_glm_kda_" ++ name);
        return k;
    }

    pub fn hasSegs(ks: *const Kernels) bool {
        return ks.prep_segs != null and ks.step_segs != null and ks.out_segs != null;
    }

    pub fn hasCopySegs(ks: *const Kernels) bool {
        return ks.copy_segs != null;
    }
};

/// The most segments one chainSegs or copySegs launch takes (glm_kda.cu's KDA_SEGS: Engine.max_shared).
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

/// One of copySegs' copies, as glm_kda.cu's KdaCopy: `n` 16-byte units from src to dst.
pub const Copy = extern struct { src: u64, dst: u64, n: i64 };
pub const Copies = extern struct { c: [max_segs]Copy };

/// Whether `bytes` from src to dst can go in copySegs: all three whole 16-byte units.
pub fn copyFits(src: u64, dst: u64, bytes: usize) bool {
    return (src | dst | @as(u64, bytes)) % 16 == 0;
}

/// The table's first `n` copies in one launch (tf_glm_kda_copy_segs, grid y the copy): cuMemcpyDtoDAsync's bytes.
pub fn copySegs(ks: *const Kernels, stream: cuda.Stream, t: *const Copies, n: usize) !void {
    if (n == 0 or n > max_segs) return error.TooManySegments;
    const f = ks.copy_segs orelse return error.KdaSegsUnavailable;
    var most: i64 = 0;
    for (t.c[0..n]) |c| most = @max(most, c.n);
    if (most <= 0) return;
    var a: cuda.Args = .{};
    a.add(t.*);
    const blocks: u32 = @intCast(@min(@as(i64, 128), @divFloor(most + 255, 256)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = blocks, .y = @intCast(n) }, .block = .{ .x = 256 }, .pdl = ks.pdl }, stream, &a);
}

/// Selected batched step entry and warp count for `total_rows`, falling back to 8-warp `step_segs`.
pub fn stepSegsChoice(ks: *const Kernels, total_rows: usize) !struct { f: cuda.Function, warps: u32 } {
    const base = ks.step_segs orelse return error.KdaSegsUnavailable;
    const wp: u32 = if (ks.step_warps != 0) ks.step_warps else stepWarps(total_rows);
    return switch (wp) {
        2 => if (ks.step2_segs) |f| .{ .f = f, .warps = 2 } else .{ .f = base, .warps = 8 },
        4 => if (ks.step4_segs) |f| .{ .f = f, .warps = 4 } else .{ .f = base, .warps = 8 },
        else => .{ .f = base, .warps = 8 },
    };
}

/// The three-kernel chain, all `t`'s segments in one launch each (z): each `chain`'s blocks; `c`: the layer's fields.
pub fn chainSegs(ks: *const Kernels, stream: cuda.Stream, c: Chain, t: *const Segs, n: usize) !void {
    if (n == 0 or n > max_segs) return error.TooManySegments;
    const prep_f = ks.prep_segs orelse return error.KdaSegsUnavailable;
    const out_f = ks.out_segs orelse return error.KdaSegsUnavailable;
    var most: usize = 0;
    var total_rows: usize = 0;
    for (t.s[0..n]) |g| {
        const r: usize = @intCast(g.rows);
        most = @max(most, r);
        total_rows += r;
    }
    if (most == 0) return;
    try validate(c, most);
    const st = try stepSegsChoice(ks, total_rows);
    const pdl = ks.pdl and most < wide_rows;
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
    try cuda.launch.launch(prep_f, .{ .grid = .{ .x = @intCast(most), .y = @intCast(c.heads), .z = z }, .block = .{ .x = 512 }, .pdl = pdl }, stream, &a);
    var b: cuda.Args = .{};
    b.add(h);
    b.add(t.*);
    try cuda.launch.launch(st.f, .{ .grid = .{ .x = @intCast(c.heads), .y = dv / 4 / st.warps, .z = z }, .block = .{ .x = st.warps * 32 }, .pdl = pdl }, stream, &b);
    var o: cuda.Args = .{};
    o.add(h);
    o.add(@as(c_int, @intCast(c.g_stride)));
    o.add(c.norm_w);
    o.add(c.eps);
    o.add(t.*);
    try cuda.launch.launch(out_f, .{ .grid = .{ .x = @intCast(most), .y = @intCast(c.heads), .z = z }, .block = .{ .x = dv }, .pdl = pdl }, stream, &o);
}

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

test "shared KDA: segment and copy tables match glm_kda.cu and fit launch argument storage" {
    try std.testing.expectEqual(@as(usize, 112), @sizeOf(Seg));
    try std.testing.expectEqual(@as(usize, 104), @offsetOf(Seg, "rows"));
    try std.testing.expectEqual(@as(usize, max_segs * 112), @sizeOf(Segs));
    try std.testing.expect(48 + @sizeOf(Segs) <= cuda.launch.Args.max_bytes);
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(Copy));
    try std.testing.expectEqual(@as(usize, max_segs * 24), @sizeOf(Copies));
    try std.testing.expect(copyFits(256, 4096, 48));
    try std.testing.expect(!copyFits(256, 4096 + 8, 48));
    try std.testing.expect(!copyFits(256 + 2, 4096, 48));
    try std.testing.expect(!copyFits(256, 4096, 40));
    try std.testing.expect(copyFits(12576 * 2 * 3, 0, 12576 * 2 * 5));
    try std.testing.expect(!copyFits(8726 * 2 * 3, 0, 8726 * 2 * 5));
    try std.testing.expect(copyFits(8728 * 2 * 3, 0, 8728 * 2 * 5));
}

test "shared KDA: segment rows stay disjoint in scratch and useSegs gates decode multi-segment wide windows" {
    const heads = 32;
    const n = heads * dk;
    const rows = [_]usize{ 3, 8, 1, 5 };
    const w: Wide = .{ .q = 1 << 32, .y = 1 << 33 };
    var row0: usize = 0;
    var t: Segs = undefined;
    for (rows, 0..) |r, i| {
        const at: u64 = @as(u64, i + 1) << 40;
        const c: Chain = .{ .p = at, .p_stride = 3 * n + heads, .b_off = 3 * n, .a = at + 1, .a_stride = n, .g = at + 2, .g_stride = n, .conv_state = at + 3, .conv_w = 0, .state_in = at + 4, .a_log = 0, .dt_bias = 0, .norm_w = 0, .eps = 0, .lower = 0, .heads = heads };
        const s: Saves = .{ .out = at + 5, .state_out = at + 6, .k = at + 7, .v = at + 8, .g = at + 9, .b = at + 10 };
        t.s[i] = seg(c, s, w, row0, r);
        const g = t.s[i];
        try std.testing.expectEqualSlices(u64, &.{ c.p, c.a, c.g, c.conv_state, c.state_in, s.state_out, s.k, s.v, s.g, s.b, s.out }, &.{ g.p, g.a, g.g, g.cs, g.state_in, g.state_out, g.k, g.v, g.gs, g.b, g.out });
        try std.testing.expectEqual(@as(c_int, @intCast(r)), g.rows);
        try std.testing.expectEqual(w.q + row0 * n * 4, g.q);
        try std.testing.expectEqual(w.y + row0 * n * 2, g.y);
        row0 += r;
    }
    for (t.s[1..rows.len], t.s[0 .. rows.len - 1], rows[0 .. rows.len - 1]) |g, was, r| {
        try std.testing.expectEqual(was.q + r * n * 4, g.q);
        try std.testing.expectEqual(was.y + r * n * 2, g.y);
    }
    try std.testing.expectEqual(w.q + row0 * n * 4, t.s[rows.len - 1].q + rows[rows.len - 1] * n * 4);
    try std.testing.expect(useSegs(false, true, true, 2));
    try std.testing.expect(useSegs(false, true, true, max_segs));
    try std.testing.expect(!useSegs(false, true, true, 1));
    try std.testing.expect(!useSegs(false, true, true, max_segs + 1));
    try std.testing.expect(!useSegs(true, true, true, 4));
    try std.testing.expect(!useSegs(false, false, true, 4));
    try std.testing.expect(!useSegs(false, true, false, 4));
    var ks: Kernels = .{
        .chain = .{ .d = undefined, .handle = @ptrFromInt(1) },
        .replay = .{ .d = undefined, .handle = @ptrFromInt(2) },
        .replay_layers = .{ .d = undefined, .handle = @ptrFromInt(3) },
        .prep = .{ .d = undefined, .handle = @ptrFromInt(4) },
        .step = .{ .d = undefined, .handle = @ptrFromInt(5) },
        .out = .{ .d = undefined, .handle = @ptrFromInt(6) },
        .prep_segs = .{ .d = undefined, .handle = @ptrFromInt(10) },
        .step_segs = .{ .d = undefined, .handle = @ptrFromInt(11) },
        .step4_segs = .{ .d = undefined, .handle = @ptrFromInt(12) },
        .step2_segs = .{ .d = undefined, .handle = @ptrFromInt(13) },
        .out_segs = .{ .d = undefined, .handle = @ptrFromInt(14) },
    };
    try std.testing.expectEqual(@as(u32, 4), (try stepSegsChoice(&ks, 4)).warps);
    try std.testing.expectEqual(@as(u32, 8), (try stepSegsChoice(&ks, 16)).warps);
    ks.step_warps = 2;
    try std.testing.expectEqual(@as(u32, 2), (try stepSegsChoice(&ks, 16)).warps);
    ks.step2_segs = null;
    try std.testing.expectEqual(@as(u32, 8), (try stepSegsChoice(&ks, 16)).warps);
}

test {
    _ = @import("kda_test.zig");
}
