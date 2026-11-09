//! The common EXL3 prompt path, after @BobClawblaw's #976 design and MiaAI-Lab's transposed kernels.
const std = @import("std");
const ex = @import("exl3_experts.zig");
const launch = @import("launch.zig");
const grouped = @import("grouped.zig");
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;

pub const tile = 64;
/// A launch's items: `tile` members on 256 threads, or TF_EXL3_PROMPT_TILE=128's 512 threads and dynamic shared bytes.
pub const Geometry = struct { tile: usize, threads: u32, shared: u32 = 0 };
pub const narrow: Geometry = .{ .tile = tile, .threads = 256 };
/// tsmem_bytes<128, 8>(): three stages of 256 rows of 40 halves and 16 tiles of two k steps' words (4 KiB).
pub const wide: Geometry = .{ .tile = 128, .threads = 512, .shared = 3 * (256 * 40 * 2 + 4096) };
/// With yrow set, pair p lands at (p / slots) * yrow + p % slots in the caller's output; b16: as bf16 (runGlm only).
pub const Scratch = struct { xg: u64, xu: u64, xd: u64, y: u64, rows: usize, slots: usize, yrow: usize = 0, b16: bool = false };
pub fn eligible(l: ex.Layer) bool {
    return l.codebook == .mcg and l.uniform_k2 == 8 and l.gu_stride == 0 and l.dims > 0 and l.width > 0 and l.dims % 256 == 0 and l.width % 128 == 0 and (l.act_mode == 0 or l.act_mode == 1);
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.Exl3Shape;
}
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.Exl3Shape;
}
fn aligned(n: usize) !usize {
    return (try add(n, 255)) / 256 * 256;
}
fn sizes(l: ex.Layer, rows: usize, slots: usize, own_y: bool) ![4]usize {
    if (!eligible(l) or rows == 0 or slots == 0 or slots > 32 or l.experts == 0 or l.experts > grouped.max_experts) return error.Exl3Shape;
    const pairs = try mul(rows, slots);
    if (pairs > std.math.maxInt(i32) or l.dims > std.math.maxInt(i32) or l.width > std.math.maxInt(i32)) return error.Exl3Shape;
    const rotated = try aligned(try mul(try mul(if (l.shared_input) rows else pairs, l.dims), 2));
    return .{ rotated, if (l.shared_input) 0 else rotated, try aligned(try mul(try mul(pairs, l.width), 2)), if (own_y) try aligned(try mul(try mul(pairs, l.dims), 4)) else 0 };
}
pub fn bytes(l: ex.Layer, rows: usize, slots: usize) !usize {
    return bytesFor(l, rows, slots, true);
}
pub fn bytesFor(l: ex.Layer, rows: usize, slots: usize, own_y: bool) !usize {
    const ns = try sizes(l, rows, slots, own_y);
    var sum: usize = 0;
    for (ns) |n| sum = try add(sum, n);
    return sum;
}
pub fn scratch(base: u64, l: ex.Layer, rows: usize, slots: usize) !Scratch {
    return scratchFor(base, l, rows, slots, true);
}
/// With own_y false, the caller must set y and yrow before launching.
pub fn scratchFor(base: u64, l: ex.Layer, rows: usize, slots: usize, own_y: bool) !Scratch {
    const ns = try sizes(l, rows, slots, own_y);
    const total = try bytesFor(l, rows, slots, own_y);
    _ = std.math.add(u64, base, total) catch return error.Exl3Shape;
    return .{ .xg = base, .xu = if (l.shared_input) base else base + ns[0], .xd = base + ns[0] + ns[1], .y = if (own_y) base + ns[0] + ns[1] + ns[2] else 0, .rows = rows, .slots = slots };
}
fn go(f: Function, s: Stream, grid: [3]usize, threads: u32, shared: u32, values: anytype) !void {
    var args: launch.Args = .{};
    inline for (values) |v| args.add(v);
    try launch.launch(f, .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads }, .shared = shared }, s, &args);
}
fn i(n: usize) i32 {
    return @intCast(n);
}
pub const Kernels = struct {
    rotate: Function,
    gateup: Function,
    down: Function,
    down_slots: ?Function = null,
    gateup_glm: ?Function = null,
    down_glm: ?Function = null,
    down_glm_b16: ?Function = null,
    gateup_glm128: ?Function = null,
    down_glm128: ?Function = null,
    down_glm128_b16: ?Function = null,
    tile: usize = tile, // TF_EXL3_PROMPT_TILE: runGlm's item members, 64 or 128 (the same bytes)
    pub fn resolve(mod: Module) !Kernels {
        const w = [3]Function{ try mod.function("tf_exl3_prompt_gu_glm128"), try mod.function("tf_exl3_prompt_down_glm128"), try mod.function("tf_exl3_prompt_down_glm128_b16") };
        for (w) |f| try f.allowDynamicShared(wide.shared);
        return .{ .rotate = try mod.function("tf_exl3_prompt_rotate"), .gateup = try mod.function("tf_exl3_prompt_gu"), .down = try mod.function("tf_exl3_prompt_down"), .down_slots = try mod.function("tf_exl3_prompt_down_slots"), .gateup_glm = try mod.function("tf_exl3_prompt_gu_glm"), .down_glm = try mod.function("tf_exl3_prompt_down_glm"), .down_glm_b16 = try mod.function("tf_exl3_prompt_down_glm_b16"), .gateup_glm128 = w[0], .down_glm128 = w[1], .down_glm128_b16 = w[2] };
    }
    /// GLM opts into patch 0078; other families retain the common prompt kernels.
    pub fn runGlm(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        if (sc.y % @as(u64, if (sc.b16) 8 else 16) != 0 or sc.xd % 8 != 0 or sc.yrow == 0) return error.Exl3Shape;
        var tuned = k;
        const g: Geometry = switch (k.tile) {
            tile => narrow,
            wide.tile => wide,
            else => return error.Exl3Shape,
        };
        tuned.gateup = (if (g.tile == wide.tile) k.gateup_glm128 else k.gateup_glm) orelse return error.Exl3Shape;
        const down = if (g.tile == wide.tile) (if (sc.b16) k.down_glm128_b16 else k.down_glm128) else if (sc.b16) k.down_glm_b16 else k.down_glm;
        tuned.down_slots = down orelse return error.Exl3Shape;
        var plain = sc;
        plain.b16 = false; // run refuses b16: only the slot kernel chosen here writes bf16
        try tuned.runAt(e, router, s, l, x, stride, picks, rows, plain, p, g);
    }
    pub fn run(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan) !void {
        return k.runAt(e, router, s, l, x, stride, picks, rows, sc, p, narrow);
    }
    /// `g`'s items: the plan, the grid and the gate/up and slot kernels' blocks (the compact down needs `narrow`).
    fn runAt(k: Kernels, e: *const ex.Experts, router: grouped.Router, s: Stream, l: ex.Layer, x: u64, stride: usize, picks: u64, rows: usize, sc: Scratch, p: grouped.Plan, g: Geometry) !void {
        if (!eligible(l) or rows == 0 or rows > sc.rows or stride < l.dims or stride > std.math.maxInt(i32)) return error.Exl3Shape;
        if (sc.y == 0 or sc.b16 or sc.yrow > std.math.maxInt(i32) or (sc.yrow != 0 and sc.yrow < sc.slots)) return error.Exl3Shape;
        _ = try bytes(l, rows, sc.slots);
        const pairs = rows * sc.slots;
        try router.route(s, picks, pairs, l.experts, g.tile, p);
        if (l.shared_input) {
            try go(k.rotate, s, .{ rows, l.dims / 128, 1 }, 32, 0, .{ x, i(stride), l.suh_g, sc.xg, i(l.dims) });
        } else {
            var args: launch.Args = .{};
            inline for (.{ x, i(stride), @as(i32, @backingInt(ex.DType.bf16)), picks, l.suh_g, l.suh_u, sc.xg, sc.xu, i(l.dims), i(sc.slots), i(l.experts) }) |v| args.add(v);
            try launch.launch(e.rot_in, .{ .grid = .{ .x = @intCast(pairs), .y = @intCast(l.dims / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &args);
        }
        const items = grouped.maxItems(pairs, l.experts, g.tile);
        try go(k.gateup, s, .{ l.width / 128, items, 1 }, g.threads, g.shared, .{ sc.xg, sc.xu, l.gate, l.up, p.items, p.counts, p.members, l.svh_g, l.svh_u, l.suh_d, sc.xd, i(l.dims), i(l.width), i(if (l.shared_input) sc.slots else 1), l.limit, l.act_mode });
        if (sc.yrow == 0) {
            if (g.tile != tile) return error.Exl3Shape;
            try go(k.down, s, .{ l.dims / 256, items, 1 }, 256, 0, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), i(sc.slots) });
            return;
        }
        try go(k.down_slots orelse return error.Exl3Shape, s, .{ l.dims / 256, items, 1 }, g.threads, g.shared, .{ sc.xd, l.down, p.items, p.counts, p.members, l.svh_d, sc.y, i(l.width), i(l.dims), i(sc.slots), i(sc.yrow) });
    }
};
