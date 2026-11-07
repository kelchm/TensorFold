//! Bounded layer timing predictions for opt-in sliced prefill.
const std = @import("std");
/// A prompt chunk's per-layer costs by row band (the Python family's sliced-fill LayerCosts): the calibration chunk's
/// CUDA events fill the first band, later chunks drift an EWMA scale; `stop` cuts greedily to the budget -- a layer
/// slower than it runs alone (CUDA kernels cannot be preempted, so the setting is a target), the fewest slices that
/// greed takes, each cut nearest the mean slice cost.
pub const LayerCosts = struct {
    gpa: std.mem.Allocator = undefined,
    layers: usize = 0,
    scale: f64 = 1, // the calibration against the chunks that run
    overhead_ms: f64 = 0, // a slice's host cost around its layers (EWMA)
    bands: std.ArrayListUnmanaged(Band) = .empty, // by rows ascending; the first chunk's rows calibrates it
    cut: []f64 = &.{}, // `stop`'s scratch: a cost a layer, then the tail's (its last layer holds it)

    const Band = struct { rows: usize, ms: []f64 }; // one cost a layer, then the chunk tail's

    pub fn ready(c: *const LayerCosts) bool {
        return c.bands.items.len != 0;
    }

    pub fn deinit(c: *LayerCosts) void {
        for (c.bands.items) |b| c.gpa.free(b.ms);
        c.bands.deinit(c.gpa);
        if (c.cut.len > 0) c.gpa.free(c.cut);
        c.* = .{};
    }

    /// A chunk's per-layer timings at `rows` become that row band (ms[layers] its tail, 0 before one is measured);
    /// the first band sets the layer count, and a band's first reading stands (the scale tracks its drift).
    pub fn calibrate(c: *LayerCosts, rows: usize, ms: []const f64) !void {
        if (rows == 0 or ms.len < 2) return error.BadCosts;
        for (ms) |v| if (v < 0 or !std.math.isFinite(v)) return error.BadCosts;
        if (c.bandAt(rows) != null) return;
        if (c.ready() and ms.len != c.layers + 1) return error.BadCosts;
        // Publish scratch and the band only after every allocation succeeds.
        const cut = if (c.ready()) c.cut else try c.gpa.alloc(f64, ms.len);
        errdefer if (!c.ready()) c.gpa.free(cut);
        const mine = try c.gpa.dupe(f64, ms);
        errdefer c.gpa.free(mine);
        const at = for (c.bands.items, 0..) |b, i| {
            if (b.rows > rows) break i;
        } else c.bands.items.len;
        try c.bands.insert(c.gpa, at, .{ .rows = rows, .ms = mine });
        c.layers = ms.len - 1;
        c.cut = cut;
    }

    fn bandAt(c: *LayerCosts, rows: usize) ?*Band {
        for (c.bands.items) |*b| if (b.rows == rows) return b;
        return null;
    }

    /// The bands' interpolation at `rows` into `out` (clamped past either end, as the Python family's costs()).
    fn costsAt(c: *LayerCosts, rows: usize, out: []f64) void {
        const bs = c.bands.items;
        var lo: usize = 0;
        var hi: usize = bs.len - 1;
        for (bs, 0..) |b, i| {
            if (b.rows <= rows) lo = i;
            if (b.rows >= rows) {
                hi = i;
                break;
            }
        }
        if (bs[lo].rows == bs[hi].rows) {
            @memcpy(out, bs[lo].ms);
            return;
        }
        const t = @as(f64, @floatFromInt(rows - bs[lo].rows)) / @as(f64, @floatFromInt(bs[hi].rows - bs[lo].rows));
        for (out, bs[lo].ms, bs[hi].ms) |*o, a, b| o.* = a + t * (b - a);
    }

    /// A `rows` chunk's tail took `ms` (its layers past runRange: the head passes, the draws, the commit).
    pub fn observeTail(c: *LayerCosts, rows: usize, ms: f64) void {
        if (!(ms >= 0) or !std.math.isFinite(ms)) return;
        if (c.bandAt(rows)) |b| {
            const old = b.ms[c.layers];
            b.ms[c.layers] = if (old > 0) 0.8 * old + 0.2 * ms else ms;
        }
    }

    /// A slice's host cost around its layers (EWMA): stop() spends the budget net of it.
    pub fn observeOverhead(c: *LayerCosts, ms: f64) void {
        if (!(ms >= 0) or !std.math.isFinite(ms)) return;
        c.overhead_ms = 0.8 * c.overhead_ms + 0.2 * ms;
    }

    /// The slice [lo, hi) took `ms` of GPU time: the EWMA scale tracks the calibration's drift (clamped 0.5..8, and
    /// only the layers' time -- the tail and the host's dispatch must not inflate every layer's prediction).
    pub fn observe(c: *LayerCosts, rows: usize, lo: usize, hi: usize, ms: f64) void {
        if (!c.ready() or !(ms > 0) or !std.math.isFinite(ms) or hi <= lo or lo >= c.layers) return;
        c.costsAt(rows, c.cut);
        var predicted: f64 = 0;
        for (c.cut[lo..@min(hi, c.layers)]) |v| predicted += v;
        if (predicted > 0) c.scale = std.math.clamp(0.8 * c.scale + 0.2 * ms / predicted, 0.5, 8);
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
        c.costsAt(rows, c.cut);
        for (c.cut[0..c.layers]) |*v| v.* *= c.scale;
        c.cut[c.layers - 1] += c.cut[c.layers] * c.scale; // the tail rides the chunk's last layer
        const cap = @max(0.001, budget - c.overhead_ms);
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

test "GLM sliced fill: LayerCosts cuts inside the budget, greedy's fewest slices, a slow layer alone" {
    const a = std.testing.allocator;
    var c: LayerCosts = .{ .gpa = a };
    defer c.deinit();
    try c.calibrate(4096, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 0 });
    try std.testing.expect(c.ready());
    // 2.5 ms takes two of the 1 ms layers: four equal slices over the eight
    var hi: usize = 0;
    var slices: usize = 0;
    var worst: f64 = 0;
    while (hi < 8) : (slices += 1) {
        const end = c.stop(4096, hi, 2.5);
        try std.testing.expect(end > hi and end <= 8);
        var ms: f64 = 0;
        for (c.cut[hi..end]) |v| ms += v;
        worst = @max(worst, ms);
        hi = end;
    }
    try std.testing.expectEqual(@as(usize, 4), slices); // greedy's minimum, 8 layers / 2
    try std.testing.expect(worst <= 2.5); // and every slice inside the budget
    // a layer slower than the budget runs alone: the 10 ms one cannot divide
    var s: LayerCosts = .{ .gpa = a };
    defer s.deinit();
    try s.calibrate(4096, &.{ 1, 1, 10, 1, 1, 1, 1, 1, 0 });
    try std.testing.expectEqual(@as(usize, 2), s.stop(4096, 0, 2.5));
    try std.testing.expectEqual(@as(usize, 3), s.stop(4096, 2, 2.5)); // the slow layer's slice alone
    var n: usize = 0;
    hi = 0;
    while (hi < 8) : (n += 1) hi = s.stop(4096, hi, 2.5);
    try std.testing.expectEqual(@as(usize, 5), n); // 2 + 1 + 2 + 2 + 1: still greedy's count
    // a slice that overran its prediction scales the rest; the host's overhead narrows the next cut
    s.observe(4096, 0, 2, 4); // 2 ms predicted, 4 taken: 1.2
    try std.testing.expectApproxEqRel(@as(f64, 1.2), s.scale, 1e-9);
    s.observeOverhead(0.5);
    try std.testing.expectApproxEqRel(@as(f64, 0.1), s.overhead_ms, 1e-9);
    try std.testing.expectEqual(@as(usize, 1), s.stop(4096, 0, 1.6)); // 1.6 - 0.1 of host cost: no two scaled layers fit
    // the tail rides the last layer once measured, and a second row band interpolates between the two
    s.observeTail(4096, 3);
    _ = s.stop(4096, 6, 100); // cut[7] now holds the scaled layer and tail: (1 + 3) * 1.2
    try std.testing.expectApproxEqRel(@as(f64, 4.8), s.cut[7], 1e-9);
    var t: LayerCosts = .{ .gpa = a };
    defer t.deinit();
    try t.calibrate(4096, &.{ 1, 1, 1, 1, 1, 1, 1, 1, 0 });
    try t.calibrate(8192, &.{ 2, 2, 2, 2, 2, 2, 2, 2, 6 });
    try std.testing.expectEqual(@as(usize, 2), t.stop(6000, 0, 3.05)); // between the bands: 1.46 a layer, two fit
}
