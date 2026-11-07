//! CUDA body and tail time only; the budget is a target, excluding host dispatch and capsule capture.
const std = @import("std");
const max_layers = @import("config.zig").max_layers;

/// Fixed upper-row buckets retain the full-chunk measurement across arbitrary final lengths. Measurements
/// normalize upward to the bucket ceiling; predictions keep that ceiling cost, including fixed launch overhead.
/// Kernel geometry can be nonlinear: these measurements guide cuts, they do not establish a latency bound.
pub const LayerCosts = struct {
    gpa: std.mem.Allocator = undefined,
    layers: usize = 0,
    bands: [row_limits.len]Band = undefined,
    count: usize = 0,
    cut: [max_layers + 1]f64 = undefined,
    cold_tail: bool = false, // scratch for stopProfile: an unmeasured tail runs with its last layer alone

    pub const row_limits = [_]usize{ 32, 64, 128, 256, 512, 1024, 2048, 4095, 4096 };
    pub const Profile = enum(u2) { chunk, final, logprobs_chunk, logprobs_final };
    const Band = struct { rows: usize, ms: [max_layers + 1]f64, tails: [4]?f64 = @splat(null) };
    // Finite saturation prevents invalid input or long-running drift from overflowing a sum of layer costs.
    const max_ms = 1e12;

    fn ceiling(rows: usize) ?usize {
        if (rows == 0) return null;
        for (row_limits) |n| if (rows <= n) return n;
        return null;
    }

    pub fn ready(c: *const LayerCosts) bool {
        return c.count != 0;
    }

    pub fn readyRows(c: *const LayerCosts, rows: usize) bool {
        const n = ceiling(rows) orelse return false;
        for (c.bands[0..c.count]) |b| if (b.rows == n) return true;
        return false;
    }

    pub fn deinit(c: *LayerCosts) void {
        c.* = .{};
    }

    fn normalized(rows: usize, limit: usize, ms: f64) f64 {
        return @min(max_ms, @min(ms, max_ms) * (@as(f64, @floatFromInt(limit)) / @as(f64, @floatFromInt(rows))));
    }

    // Rising costs apply immediately; falling costs converge without a permanent peak or a drift clamp.
    fn update(old: f64, measured: f64) f64 {
        return if (measured >= old) measured else 0.8 * old + 0.2 * measured;
    }

    pub fn calibrate(c: *LayerCosts, rows: usize, ms: []const f64) !void {
        return c.calibrateProfile(rows, ms, .final);
    }

    /// Only a complete set of measured single layers establishes a new body bucket. Recalibration merges
    /// conservatively; a small final chunk cannot erase a slower observation in its bucket.
    pub fn calibrateProfile(c: *LayerCosts, rows: usize, ms: []const f64, profile: Profile) !void {
        const n = ceiling(rows) orelse return error.BadCosts;
        if (ms.len < 2 or ms.len > max_layers + 1) return error.BadCosts;
        for (ms) |v| if (v < 0 or !std.math.isFinite(v)) return error.BadCosts;
        if (c.ready() and ms.len != c.layers + 1) return error.BadCosts;
        const existing = c.bandAt(rows);
        const b = existing orelse blk: {
            const i = c.count;
            c.count += 1;
            c.bands[i] = .{ .rows = n, .ms = @splat(0) };
            break :blk &c.bands[i];
        };
        for (b.ms[0..ms.len], ms) |*out, v| {
            const measured = normalized(rows, n, v);
            out.* = if (existing != null) update(out.*, measured) else measured;
        }
        c.layers = ms.len - 1;
        c.observeTailProfile(rows, ms[c.layers], profile);
    }

    fn bandAt(c: *LayerCosts, rows: usize) ?*Band {
        const n = ceiling(rows) orelse return null;
        for (c.bands[0..c.count]) |*b| if (b.rows == n) return b;
        return null;
    }

    pub fn observeTail(c: *LayerCosts, rows: usize, ms: f64) void {
        c.observeTailProfile(rows, ms, .final);
    }

    /// Final draw/hold and prompt-logprob work have separate tail histories; an unseen profile never borrows
    /// a cheap ordinary tail. Logprob k and kernel shapes can still differ within a profile.
    pub fn observeTailProfile(c: *LayerCosts, rows: usize, ms: f64, profile: Profile) void {
        if (!(ms >= 0) or !std.math.isFinite(ms)) return;
        const b = c.bandAt(rows) orelse return;
        const measured = normalized(rows, b.rows, ms);
        const v = &b.tails[@backingInt(profile)];
        v.* = if (v.*) |old| update(old, measured) else measured;
        b.ms[c.layers] = v.*.?;
    }

    /// Update the measured body range at the bucket ceiling, without transferring costs across buckets.
    pub fn observe(c: *LayerCosts, rows: usize, lo: usize, hi: usize, ms: f64) void {
        if (!(ms >= 0) or !std.math.isFinite(ms) or hi <= lo or hi > c.layers) return;
        const b = c.bandAt(rows) orelse return;
        var predicted: f64 = 0;
        for (b.ms[lo..hi]) |v| predicted += v;
        const measured = normalized(rows, b.rows, ms);
        for (b.ms[lo..hi]) |*v| {
            const share = if (predicted > 0) v.* / predicted else 1 / @as(f64, @floatFromInt(hi - lo));
            v.* = update(v.*, measured * share);
        }
    }

    /// Greedy's end from `start`: layers while they fit `cap`, always one (a layer slower than the budget alone).
    fn greedyEnd(c: *LayerCosts, start: usize, cap: f64) usize {
        var end = start + 1;
        var ms = c.cut[start];
        while (end < c.layers and (!c.cold_tail or end < c.layers - 1) and ms + c.cut[end] <= cap) : (end += 1) ms += c.cut[end];
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
        return c.stopProfile(rows, lo, budget, .final);
    }

    pub fn stopProfile(c: *LayerCosts, rows: usize, lo: usize, budget: f64, profile: Profile) usize {
        if (lo >= c.layers or !std.math.isFinite(budget) or !(budget > 0)) return lo + 1;
        const band = c.bandAt(rows) orelse return lo + 1;
        @memcpy(c.cut[0 .. c.layers + 1], band.ms[0 .. c.layers + 1]);
        const tail = band.tails[@backingInt(profile)];
        c.cold_tail = tail == null;
        c.cut[c.layers - 1] += tail orelse 0; // measured matching tail rides the last layer
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
    try t.expectEqual(@as(usize, 4), c.count);
    try t.expect(c.readyRows(4096)); // arbitrary short lengths never evict this bucket
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
