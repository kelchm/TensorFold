//! Opt-in bit-exact verification tile choices measured on paged SM121 TP2 (BF16 or standard FP8 KV).
const std = @import("std");
const index_mod = @import("cuda_index.zig");

pub const Options = struct {
    mla: bool = false,
    ranges: bool = false,

    pub fn enabled(o: Options) bool {
        return o.mla or o.ranges;
    }

    pub fn gate(o: Options, sm: usize, world: usize, paged: bool, kv_host: u8) Options {
        return if (sm == 121 and world == 2 and paged and kv_host == 0) o else .{};
    }

    /// One single-stream verify window. The caller later checks the cache and layer geometry.
    pub fn window(o: Options, pos: usize, rows: usize) Window {
        const wide = o.mla and rows >= 6 and rows <= 16 and
            ((o.ranges and pos >= 4096) or (rows >= 8 and pos >= 40960));
        const floor: usize = switch (rows) {
            2 => 16384,
            3...6 => 8192,
            8 => 4096,
            12, 16 => 2048,
            7, 9...11, 13...15 => 40960,
            else => return .{ .mla = wide },
        };
        return .{ .mla = wide, .scores = o.ranges and pos >= floor };
    }
};

pub const Window = struct {
    mla: bool = false,
    scores: bool = false,

    /// Keep upstream's measured tiling for every shape its decode policy already selected.
    pub fn scoreLoop(w: Window, upstream: ?usize, np: usize, nb: usize, heads: usize, pool: usize, rope: usize, pk: u64, pt: u64, ps: u64) ?usize {
        if (upstream) |run| return run;
        if (!w.scores or np < index_mod.topk or heads != 32 or pool != 4 or rope != 0 or pt == 0 or pk % 16 != 0 or ps % 16 != 0) return null;
        // Hold at least about 376 independent CTAs while expanding beyond upstream's decode gate.
        return @min(16, @max(1, nb / 376));
    }

    /// Exact 32-head sparse attention tile on paged BF16 (logical=false) or paged FP8 (logical=true).
    pub fn wide(w: Window, rows: usize, heads: usize, lw: usize, width: usize, scales: u64, pt: u64, logical: bool) bool {
        return w.mla and rows >= 6 and rows <= 16 and heads == 32 and lw == 512 and width == 2051 and pt != 0 and logical == (scales != 0);
    }
};

test "startup switches: verification tile ranges retain upstream scorer loop and cover paged BF16 and FP8 shapes" {
    const opts: Options = .{ .mla = true, .ranges = true };
    try std.testing.expect(!opts.gate(120, 2, true, 0).enabled());
    try std.testing.expect(!opts.gate(121, 4, true, 0).enabled());
    try std.testing.expect(!opts.gate(121, 2, false, 0).enabled());
    try std.testing.expect(!opts.gate(121, 2, true, 1).enabled());
    try std.testing.expect(opts.gate(121, 2, true, 0).enabled());
    for ([_][2]usize{ .{ 2, 16384 }, .{ 3, 8192 }, .{ 4, 8192 }, .{ 5, 8192 }, .{ 6, 8192 }, .{ 7, 40960 }, .{ 8, 4096 }, .{ 12, 2048 }, .{ 16, 2048 } }) |pair| {
        try std.testing.expect(!opts.window(pair[1] - 1, pair[0]).scores);
        try std.testing.expect(opts.window(pair[1], pair[0]).scores);
    }
    const window = opts.window(262144, 9);
    try std.testing.expectEqual(@as(?usize, 7), window.scoreLoop(7, 512, 1025, 16, 1, 64, 0, 0, 0));
    try std.testing.expectEqual(@as(?usize, 2), window.scoreLoop(null, 512, 1025, 32, 4, 0, 16, 1024, 4096));
    try std.testing.expectEqual(@as(?usize, null), window.scoreLoop(null, 511, 1025, 32, 4, 0, 16, 1024, 4096));
    try std.testing.expectEqual(@as(?usize, null), window.scoreLoop(null, 512, 1025, 32, 1, 64, 16, 1024, 4096));
    try std.testing.expectEqual(@as(?usize, null), window.scoreLoop(null, 512, 1025, 32, 4, 0, 16, 0, 4096));
    try std.testing.expect(!opts.window(4095, 7).mla);
    // Both paged FP8 (scales != 0, logical = true) and paged BF16 (scales == 0, logical = false) qualify.
    try std.testing.expect(opts.window(4096, 7).wide(7, 32, 512, 2051, 1024, 512, true));
    try std.testing.expect(opts.window(4096, 7).wide(7, 32, 512, 2051, 0, 512, false));
    try std.testing.expect(!opts.window(4096, 7).wide(7, 32, 576, 2048, 1024, 512, true));
    try std.testing.expect(!opts.window(4096, 7).wide(7, 32, 512, 2051, 1024, 512, false));
    try std.testing.expect(!opts.window(4096, 7).wide(7, 32, 512, 2051, 0, 512, true));
    try std.testing.expect(!opts.window(262144, 17).mla);
    try std.testing.expect(!(Options{ .mla = true }).window(4096, 8).mla);
    try std.testing.expect((Options{ .mla = true }).window(40960, 8).mla);
    try std.testing.expect(!(Options{ .ranges = true }).window(262144, 8).mla);
}
