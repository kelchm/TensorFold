//! GLM's opt-in copy facts; acceptance estimates and cost allocation stay in the shared lane core.
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");

/// With DFlash2: a window's ms by rows on two GB10s (EXL3, TF_GLM_PROF: 86 to 107 ms at 5 to 8), shared as glm_cuda's.
pub const window_costs = costTable(@import("cuda_forward.zig").window_rows, 52.0, 6.8);
pub const shared_costs = costTable(engine.shared_rows, 76.0, 4.1);

fn costTable(comptime n: usize, comptime one: f64, comptime row: f64) [n]lanes.config.Cost {
    var c: [n]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = one + row * @as(f64, @floatFromInt(i)) };
    return c;
}

/// The serving drafts switch wins; unset keeps the family's facts. `measured`: timed prices, so no seed tables.
pub fn apply(base: lanes.Model, enabled: bool, measured: bool) lanes.Model {
    if (!enabled or !base.mtp) return base;
    var m = base;
    m.root_copies = true;
    m.context_copies = true;
    m.copy_match = 4;
    m.own_copy_rates = true;
    m.plain_guard = true;
    if (!measured) m.window_costs = &window_costs;
    if (!measured) m.shared_costs = &shared_costs;
    m.window_costs_lone = if (base.priced_windows) base.window_costs_lone else false; // every copy pays for its rows, in shared rounds too
    m.shared_drafts = 0;
    // Keep head confidence in the backend's probabilities so every copy still pays its target-window cost.
    m.draft_confidence = null;
    return m;
}

/// Optional TF_GLM_COPY_8_16 policy for DFlash2: 8 prompt tokens, 16 reply tokens, up to 15 drafts.
pub fn applyLong(base: lanes.Model, enabled: bool, measured: bool, long_match: bool) lanes.Model {
    var m = apply(base, enabled, measured);
    if (long_match and enabled and base.mtp) {
        m.fixed_copy_drafts = 15;
        m.copy_match = 8;
        m.copy_reply_match = 16;
    }
    return m;
}

/// The same shared confidence prefix, encoded as zero tail probabilities for the ordinary cost allocator.
pub fn confidence(floor: ?f64, chances: []f64) void {
    if (floor == null) return;
    const kept = (lanes.mtp_policy.Controls{ .floor = floor }).prefix(chances);
    @memset(chances[kept..], 0);
}

/// Publish the explicit head prefix as the held count, so even a cheaper wider tile cannot cross the cap.
pub fn held(ptr: *anyopaque, s: *lanes.Stream, _: @import("std").mem.Allocator) !?lanes.stream.Held {
    const self: *@import("cuda_lanes.zig").Lanes = @ptrCast(@alignCast(ptr));
    if (self.mtp_confidence == null or !self.copy_confidence or self.x.e.drafter != null) return null;
    const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
    if (l.held_n > engine.max_drafts) return error.NoHeldDrafts;
    var chances: [engine.max_drafts]f64 = undefined;
    const out = chances[0..l.held_n];
    const b = self.backend();
    if (!try b.vtable.probabilities.?(ptr, s, out)) return error.NoDraftProbabilities;
    return .{ .count = @intCast((lanes.mtp_policy.Controls{ .floor = self.mtp_confidence }).prefix(out)) };
}

/// A priced parallel block leaves raw chances for the shared calibrator; off preserves the product cut.
pub fn blockChances(requests: anytype, records: anytype, priced: bool) !void {
    const dflash = @import("dflash.zig");
    const cut = if (priced) @as(f64, 0) else try dflash.confidenceFor(requests.len > 1);
    for (requests, records) |r, l| {
        l.held_n = @min(r.depth, dflash.BLOCK - 1);
        if (priced) continue;
        var chain: f64 = 1;
        for (l.chances[0..l.held_n], 0..) |*c, j| {
            chain *= c.*;
            if (j > 0 and chain < cut) c.* = 0;
            if (c.* == 0) chain = 0;
        }
    }
}

/// DFlash2: validate requests before wire traffic, fold normal keeps into Kind.block, then calibrate chances.
pub fn draftBlock(self: *@import("cuda_lanes.zig").Lanes, requests: []const lanes.backend.DraftRequest) !void {
    const std = @import("std");
    const dflash = @import("dflash.zig");
    const proposals = @import("dflash_proposals.zig");
    const Lane = @import("cuda_lanes.zig").Lane;
    if (requests.len == 0 or requests.len > dflash.MAX_S) return error.TooManyStreams;
    if (self.x.e.drafter == null) return error.NoDrafter;
    var reqs: [dflash.MAX_S]dflash.Req = undefined;
    var records: [dflash.MAX_S]*Lane = undefined;
    var commit_mask: u16 = 0;
    // Pass 1: pure validation with zero mutations or wire traffic.
    for (requests, 0..) |r, i| {
        const l = self.lanes.getPtr(r.stream) orelse return error.UnknownStream;
        if (r.lanes != null) return error.TreesNotBuilt;
        if (l.slot >= self.x.e.seqs.items.len) return error.UnknownSlot;
        const seq = self.x.e.seqs.items[l.slot] orelse return error.UnknownSlot;
        for (records[0..i]) |prev| if (prev.slot == l.slot) return error.SlotTwice;
        var n: usize = 0;
        const anchor = if (r.rows) |rows| a: {
            if (rows.len == 0 or rows.len > r.follow.len) return error.BadCommit;
            for (rows, 0..) |row, j| if (row != j) return error.TreesNotBuilt;
            n = rows.len;
            break :a r.follow[rows.len - 1];
        } else self.value(r.first orelse return error.NoFirstToken);
        if (std.math.cast(u32, n) == null) return error.BadStep;
        if (r.depth > 0 and anchor >= self.x.e.c.vocab) return error.BadTokenId;
        const keep_n: usize = if (l.pending_rows) |pending| (if (r.rows != null) n else pending) else 0;
        if (l.pending_rows != null) {
            try proposals.checkCommit(self.x.e, l.slot, keep_n);
            if (r.rows != null) commit_mask |= @as(u16, 1) << @intCast(i);
        }
        if (r.position != seq.st.pos + keep_n + 1) return error.PositionMismatch;
        reqs[i] = .{ .slot = l.slot, .pos = 0, .n = n, .anchor = anchor, .propose = r.depth > 0, .out = &l.held, .probs = &l.chances };
        records[i] = l;
    }
    // Pass 2: rare rowless keeps (r.rows == null, pending_rows != null) flush upfront via bind.
    for (requests, records[0..requests.len]) |r, l| if (r.rows == null and l.pending_rows != null) {
        _ = try self.bind(r.stream, null);
    };
    // Pass 3: single Kind.block step carrying all folded keeps; keep pending_rows synced on partial errors.
    errdefer for (records[0..requests.len], 0..) |l, i| {
        if ((commit_mask & (@as(u16, 1) << @intCast(i))) != 0) {
            if (self.x.e.seqs.items[l.slot]) |seq| if (seq.st.pending == 0) {
                l.pending_rows = null;
            };
        }
    };
    try self.x.blockKeeping(reqs[0..requests.len], commit_mask);
    for (records[0..requests.len], 0..) |l, i| {
        if ((commit_mask & (@as(u16, 1) << @intCast(i))) != 0) l.pending_rows = null;
    }
    try blockChances(requests, records[0..requests.len], self.x.e.m.priced_windows);
}

test "blockChances: TF_GLM_SHARED_DRAFT_CONFIDENCE overrides multi-stream rounds while preserving lone cuts" {
    const std = @import("std");
    const sys = @import("sys");
    const t = std.testing;
    const Lane = struct { held_n: usize = 0, chances: [7]f64 = @splat(0.85) };
    const Req = struct { depth: usize };
    _ = sys.env.unsetenv("TF_GLM_DRAFT_CONFIDENCE");
    _ = sys.env.unsetenv("TF_GLM_SHARED_DRAFT_CONFIDENCE");
    defer {
        _ = sys.env.unsetenv("TF_GLM_DRAFT_CONFIDENCE");
        _ = sys.env.unsetenv("TF_GLM_SHARED_DRAFT_CONFIDENCE");
    }
    const one_req = [_]Req{.{ .depth = 7 }};
    const two_req = [_]Req{ .{ .depth = 7 }, .{ .depth = 7 } };
    // (a) Unset shared cut inherits TF_GLM_DRAFT_CONFIDENCE (0.70 default -> 0.85^2=0.7225 kept, 0.85^3=0.614 cut).
    var l1: Lane = .{};
    var s0: Lane = .{};
    var s1: Lane = .{};
    try blockChances(&one_req, &[_]*Lane{&l1}, false);
    try blockChances(&two_req, &[_]*Lane{ &s0, &s1 }, false);
    for ([_]Lane{ l1, s0, s1 }) |ln| {
        try t.expectEqual(@as(usize, 7), ln.held_n);
        for (ln.chances[0..2]) |p| try t.expectEqual(@as(f64, 0.85), p);
        for (ln.chances[2..]) |p| try t.expectEqual(@as(f64, 0), p);
    }
    // (b) Explicit TF_GLM_SHARED_DRAFT_CONFIDENCE=0.35 keeps 0.85^6=0.377 on shared rounds while lone stays at 0.70.
    _ = sys.env.setenv("TF_GLM_DRAFT_CONFIDENCE", "0.70", 1);
    _ = sys.env.setenv("TF_GLM_SHARED_DRAFT_CONFIDENCE", "0.35", 1);
    l1 = .{};
    s0 = .{};
    s1 = .{};
    try blockChances(&one_req, &[_]*Lane{&l1}, false);
    try blockChances(&two_req, &[_]*Lane{ &s0, &s1 }, false);
    for (l1.chances[0..2]) |p| try t.expectEqual(@as(f64, 0.85), p);
    for (l1.chances[2..]) |p| try t.expectEqual(@as(f64, 0), p);
    for ([_]Lane{ s0, s1 }) |ln| {
        for (ln.chances[0..6]) |p| try t.expectEqual(@as(f64, 0.85), p);
        try t.expectEqual(@as(f64, 0), ln.chances[6]);
    }
    // (c) Malformed shared cut is rejected by environment.parse and blockChances on multi-stream rounds.
    for ([_][]const u8{ "bad", "-0.1", "1.05", "nan", "inf" }) |raw| {
        try t.expectError(error.InvalidEnvironment, sys.environment.parse("TF_GLM_SHARED_DRAFT_CONFIDENCE", raw));
    }
    _ = sys.env.setenv("TF_GLM_SHARED_DRAFT_CONFIDENCE", "1.2", 1);
    s0 = .{};
    s1 = .{};
    try t.expectError(error.InvalidEnvironment, blockChances(&two_req, &[_]*Lane{ &s0, &s1 }, false));
}
