//! CUDA body and tail time only; the budget is a target, excluding host dispatch and capsule capture.
const std = @import("std");
const max_layers = @import("config.zig").max_layers;

/// Exact row counts never borrow a short chunk's timing. Eight bands bound storage across arbitrary final lengths.
pub const LayerCosts = struct {
    gpa: std.mem.Allocator = undefined,
    layers: usize = 0,
    bands: [8]Band = undefined,
    count: usize = 0,
    replace: usize = 0,
    cut: [max_layers + 1]f64 = undefined,

    const Band = struct { rows: usize, ms: [max_layers + 1]f64 };
    // Finite saturation prevents invalid input or long-running drift from overflowing a sum of layer costs.
    const max_ms = 1e12;

    pub fn ready(c: *const LayerCosts) bool {
        return c.count != 0;
    }

    pub fn readyRows(c: *const LayerCosts, rows: usize) bool {
        for (c.bands[0..c.count]) |b| if (b.rows == rows) return true;
        return false;
    }

    pub fn deinit(c: *LayerCosts) void {
        c.* = .{};
    }

    /// Single-layer measurements for this exact row count, followed by its tail, establish its own band.
    pub fn calibrate(c: *LayerCosts, rows: usize, ms: []const f64) !void {
        if (rows == 0 or ms.len < 2 or ms.len > max_layers + 1) return error.BadCosts;
        for (ms) |v| if (v < 0 or !std.math.isFinite(v)) return error.BadCosts;
        if (c.ready() and ms.len != c.layers + 1) return error.BadCosts;
        const b = c.bandAt(rows) orelse blk: {
            const i = if (c.count < c.bands.len) c.count else c.replace;
            if (c.count < c.bands.len) c.count += 1 else c.replace = (c.replace + 1) % c.bands.len;
            break :blk &c.bands[i];
        };
        b.rows = rows;
        for (b.ms[0..ms.len], ms) |*out, v| out.* = @min(v, max_ms);
        c.layers = ms.len - 1;
    }

    fn bandAt(c: *LayerCosts, rows: usize) ?*Band {
        for (c.bands[0..c.count]) |*b| if (b.rows == rows) return b;
        return null;
    }

    pub fn observeTail(c: *LayerCosts, rows: usize, ms: f64) void {
        if (!(ms >= 0) or !std.math.isFinite(ms)) return;
        if (c.bandAt(rows)) |b| b.ms[c.layers] = 0.8 * b.ms[c.layers] + 0.2 * @min(ms, max_ms);
    }

    /// Update this band's measured range, without a global scale or a clamp that prevents convergence.
    pub fn observe(c: *LayerCosts, rows: usize, lo: usize, hi: usize, ms: f64) void {
        if (!(ms >= 0) or !std.math.isFinite(ms) or hi <= lo or hi > c.layers) return;
        const b = c.bandAt(rows) orelse return;
        var predicted: f64 = 0;
        for (b.ms[lo..hi]) |v| predicted += v;
        for (b.ms[lo..hi]) |*v| {
            const share = if (predicted > 0) v.* / predicted else 1 / @as(f64, @floatFromInt(hi - lo));
            v.* = 0.8 * v.* + 0.2 * @min(ms, max_ms) * share;
        }
    }

    /// Greedy's end from `start`: layers while they fit `cap`, always one (a layer slower than the budget alone).
    fn greedyEnd(c: *LayerCosts, start: usize, cap: f64) usize {
        var end = start + 1;
        var ms = c.cut[start];
        while (end < c.layers and ms + c.cut[end] <= cap) : (end += 1) ms += c.cut[end];
        return end;
    }

    fn piecesFrom(c: *LayerCosts, start: usize, cap: f64) usize {
        var n: usize = 0;
        var at = start;
        while (at < c.layers) {
            at = c.greedyEnd(at, cap);
            n += 1;
        }
        return n;
    }

    /// The next slice's end layer from `lo`: greedy gives the minimum slice count; among the first cuts that keep it,
    /// the one nearest the mean slice cost (ties: the deepest), so no tiny tail trails several near-budget stalls.
    pub fn stop(c: *LayerCosts, rows: usize, lo: usize, budget: f64) usize {
        if (lo >= c.layers or !(budget > 0)) return lo + 1;
        const band = c.bandAt(rows) orelse return lo + 1;
        @memcpy(c.cut[0 .. c.layers + 1], band.ms[0 .. c.layers + 1]);
        c.cut[c.layers - 1] += c.cut[c.layers]; // measured tail rides the last layer
        const cap = @max(0.001, budget);
        const end = c.greedyEnd(lo, cap);
        const count = c.piecesFrom(lo, cap);
        var left: f64 = 0; // the layers past lo, for the mean and each cut's distance from it
        for (c.cut[lo..c.layers]) |v| left += v;
        const target = left / @as(f64, @floatFromInt(count));
        var best = end;
        var best_gap = blk: {
            var sum: f64 = 0;
            for (c.cut[lo..end]) |v| sum += v;
            break :blk @abs(sum - target);
        };
        var hi = lo + 1;
        while (hi < end) : (hi += 1) {
            if (c.piecesFrom(hi, cap) > count - 1) continue;
            var sum: f64 = 0;
            for (c.cut[lo..hi]) |v| sum += v;
            const gap = @abs(sum - target);
            if (gap <= best_gap) { // equal gaps take the deeper cut, as the Python family's -hi
                best = hi;
                best_gap = gap;
            }
        }
        return best;
    }
};

test "GLM sliced fill: independent row bands and finite ongoing drift" {
    const t = std.testing;
    var c: LayerCosts = .{};
    try c.calibrate(64, &.{ 0.3, 0.3, 0.5 });
    try t.expectEqual(@as(usize, 1), c.stop(4096, 0, 20));
    try c.calibrate(4096, &.{ 13, 13, 0.5 });
    for (0..100) |_| c.observe(4096, 0, 1, 130);
    try t.expectApproxEqRel(@as(f64, 130), c.bands[1].ms[0], 1e-7);
    try t.expectEqual(@as(f64, 0.3), c.bands[0].ms[0]);
    for (0..100) |_| c.observe(4096, 0, 1, 1);
    try t.expectApproxEqRel(@as(f64, 1), c.bands[1].ms[0], 1e-7);
    c.observe(4096, 0, 1, std.math.inf(f64));
    c.observeTail(4096, std.math.nan(f64));
    try t.expect(std.math.isFinite(c.bands[1].ms[0]));
    for (1..100) |rows| try c.calibrate(rows, &.{ 1, 1, 0 });
    try t.expectEqual(@as(usize, 8), c.count);
}

test "GLM sliced fill: greedy budget, independent actual costs, slow layer alone" {
    const t = std.testing;
    var c: LayerCosts = .{};
    const actual = [_]f64{ 1, 1, 10, 1, 1, 1, 1, 1, 0.5 };
    try c.calibrate(4096, &actual);
    var at: usize = 0;
    var count: usize = 0;
    while (at < actual.len - 1) : (count += 1) {
        const end = c.stop(4096, at, 2.5);
        var elapsed: f64 = if (end == actual.len - 1) actual[actual.len - 1] else 0;
        for (actual[at..end]) |v| elapsed += v;
        try t.expect(elapsed <= 2.5 or end == at + 1);
        at = end;
    }
    try t.expectEqual(@as(usize, 5), count);
}
