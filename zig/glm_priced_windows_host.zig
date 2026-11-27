//! GLM's actual measurement/publishing policies under a fake window timer; no CUDA driver calls.
const std = @import("std");
const glm = @import("glm");
const lanes = @import("lanes");
const env = @import("sys").env;
const t = std.testing;
const costs = glm.lane_costs;
const Timer = struct {
    windows: usize = 0,
    drafts: usize = 0,
    forms: usize = 0,
    slope: f64 = 0.05,
    err: ?anyerror = null,
    window_override: ?f64 = null,
    draft_override: ?f64 = null,
    pub fn form(s: *Timer, _: u64) !void {
        s.forms += 1;
    }
    pub fn window(s: *Timer, i: usize) !f64 {
        if (s.err) |e| return e;
        s.windows += 1;
        if (s.window_override) |v| return v;
        return 10 + s.slope * @as(f64, @floatFromInt(costs.widths[i] - 1));
    }
    pub fn draft(s: *Timer, depth: usize) !f64 {
        s.drafts += 1;
        try t.expectEqual(@as(usize, 7), depth);
        if (s.draft_override) |v| return v;
        return 7;
    }
};

test "GLM priced windows: fake target race publishes all widths and a fixed block price" {
    var timer: Timer = .{};
    const kept = (try costs.attempt(&timer, true, null)).?;
    const table = costs.tableForMode(kept, true, true).?;
    try t.expect(timer.windows > costs.widths.len and timer.drafts > 0);
    try t.expectApproxEqAbs(@as(f64, 1), table.step_ms, 1e-9);
    for (costs.widths, kept.ms) |width, ms| try t.expectApproxEqAbs(ms, table.window[width - 1].ms, 1e-9);
    try t.expect(costs.tableForMode(kept, true, false) == null);
}

test "GLM priced windows: pinned blocks time windows only under the opt-in" {
    var off: Timer = .{};
    const seed = try costs.measurements.measure(&off, true, true);
    try t.expectEqual(@as(usize, 0), off.windows);
    try t.expectEqual(@as(usize, 0), off.drafts);
    try t.expect(costs.tableFor(seed, true) == null);
    var on: Timer = .{};
    const measured = (try costs.attempt(&on, true, true)).?;
    try t.expect(on.windows > 0 and on.drafts > 0);
    try t.expect(costs.tableForMode(measured, true, true) != null);
    try t.expect(!costs.measurements.reusable(seed, false));
}

test "GLM priced windows: unusable samples use seeds while device, transport and OOM errors propagate" {
    for ([_]f64{ 0, std.math.nan(f64), std.math.inf(f64) }) |bad_win| {
        var timer: Timer = .{ .window_override = bad_win };
        try t.expect(try costs.attempt(&timer, true, null) == null);
    }
    for ([_]f64{ -7, std.math.nan(f64), std.math.inf(f64) }) |bad_draft| {
        var timer: Timer = .{ .draft_override = bad_draft };
        try t.expect(try costs.attempt(&timer, true, null) == null);
    }
    for ([_]anyerror{ error.OutOfMemory, error.OutOfDeviceMemory, error.CudaContextLost, error.ProxyFailed }) |fatal| {
        var failing: Timer = .{ .err = fatal };
        try t.expectError(fatal, costs.attempt(&failing, true, null));
    }
    try t.expect(costs.wantsTiming(false, true, false, 1024));
    try t.expect(costs.wantsTiming(false, true, true, 1024));
    try t.expect(!costs.wantsTiming(false, true, true, 32));
    try t.expect(!costs.wantsTiming(false, false, true, 1024));
}

fn facts(on: bool, table: ?costs.Table, copies: bool) !void {
    var e: glm.engine.Engine = undefined;
    var drafter: glm.dflash.Drafter = undefined;
    e.mtp = false;
    e.drafter = &drafter;
    e.m.priced_windows = on;
    var backend: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined }, .alone = true, .costs = table };
    const m = backend.factsWithCopies(copies);
    if (on) {
        try t.expect(m.priced_windows and m.mtp and m.plain_guard and m.lone_cost_tile == 16);
        if (table) |v| {
            try t.expectApproxEqAbs(v.window[0].ms, m.window_costs[0].ms, 1e-9);
            try t.expect(m.window_costs_lone);
        } else try t.expectEqual(@as(f64, 52), m.window_costs[0].ms);
    } else {
        try t.expect(!m.priced_windows and m.mtp);
        try t.expectEqual(@as(f64, 52), m.window_costs[0].ms);
    }
    e.mtp = true;
    const serving = backend.factsWithCopies(copies);
    try t.expect(serving.mtp);
    if (on) if (table) |v| {
        try t.expectApproxEqAbs(v.window[0].ms, serving.window_costs[0].ms, 1e-9);
        try t.expect(serving.window_costs_lone);
    };
    e.drafter = null;
    try t.expect(!backend.facts().priced_windows);
}
test "GLM priced windows: actual facts publish measured table even without MTP and retain copy policy" {
    var timer: Timer = .{};
    const table = costs.tableForMode((try costs.attempt(&timer, true, null)).?, true, true);
    for ([_]bool{ false, true }) |copies| {
        try facts(true, table, copies);
        try facts(true, null, copies);
        try facts(false, table, copies);
    }
}

test "GLM priced windows: dflash_only composes measured pricing with TF_GLM_COPY_8_16 without MTP" {
    var timer: Timer = .{};
    const table = costs.tableForMode((try costs.attempt(&timer, true, null)).?, true, true).?;
    var e: glm.engine.Engine = undefined;
    var drafter: glm.dflash.Drafter = undefined;
    e.mtp = false;
    e.drafter = &drafter;
    e.m.priced_windows = true;
    var backend: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined }, .alone = false, .long_copy_match = true, .costs = table };
    const m = backend.factsWithCopies(true);
    try t.expect(m.priced_windows and m.mtp and m.speculate and m.fixed_draft_block and m.plain_guard);
    try t.expect(m.window_costs_lone and m.root_copies and m.context_copies);
    try t.expectEqual(@as(u32, 16), m.lone_cost_tile);
    try t.expectEqual(@as(u32, 15), m.fixed_copy_drafts);
    try t.expectEqual(@as(u32, 8), m.copy_match);
    try t.expectEqual(@as(u32, 16), m.copy_reply_match);
    try t.expectApproxEqAbs(table.window[0].ms, m.window_costs[0].ms, 1e-9);
    try t.expect(backend.copy_confidence);
}

test "GLM priced windows: off retains the 0.7 cut while on leaves every already-drafted chance" {
    _ = env.unsetenv("TF_GLM_DRAFT_CONFIDENCE");
    defer _ = env.unsetenv("TF_GLM_DRAFT_CONFIDENCE");
    const Lane = struct { held_n: usize = 0, chances: [7]f64 = @splat(0.9) };
    const requests = [_]struct { depth: usize }{.{ .depth = 7 }};
    var off: Lane = .{};
    var on: Lane = .{};
    try glm.copy_policy.blockChances(&requests, &[_]*Lane{&off}, false);
    try glm.copy_policy.blockChances(&requests, &[_]*Lane{&on}, true);
    try t.expectEqual(@as(usize, 7), off.held_n);
    for (off.chances[0..3]) |p| try t.expectEqual(@as(f64, 0.9), p);
    for (off.chances[3..]) |p| try t.expectEqual(@as(f64, 0), p);
    for (on.chances) |p| try t.expectEqual(@as(f64, 0.9), p);
}

test "GLM priced windows: actual native publication body and startup cleanup compile without execution" {
    std.mem.doNotOptimizeAway(&glm.native.priceWindows);
}

test "GLM priced windows: failed-race fallback releases scratch, resets form and idles, cleanup failure propagates" {
    const Owner = struct {
        order: usize = 0,
        form: u64 = 0,
        fail: bool = false,
        pub fn release(o: *@This(), slot: u32) !void {
            try t.expectEqual(@as(u32, 0), slot);
            if (o.fail) return error.MockCleanup;
            try t.expectEqual(@as(usize, 0), o.order);
            o.order += 1;
        }
        pub fn expertForm(o: *@This(), form: u64) !void {
            try t.expectEqual(@as(usize, 1), o.order);
            o.order += 1;
            o.form = form;
        }
        pub fn idle(o: *@This()) !void {
            try t.expectEqual(@as(usize, 2), o.order);
            o.order += 1;
        }
    };
    var classic: Owner = .{};
    try t.expect(try costs.fallback(&classic, null) == null);
    try t.expectEqual(@as(usize, 3), classic.order);
    try t.expectEqual(@as(u64, 0), classic.form);
    var factored: Owner = .{};
    try t.expect(try costs.fallback(&factored, true) == null);
    try t.expectEqual(@as(u64, std.math.maxInt(u64)), factored.form);
    var failed: Owner = .{ .fail = true };
    try t.expectError(error.MockCleanup, costs.fallback(&failed, null));
    try t.expectEqual(@as(usize, 0), failed.order);
}
