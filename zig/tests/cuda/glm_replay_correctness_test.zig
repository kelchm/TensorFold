//! Driver-free planted controls for adversarial replay and checked block outputs.
const std = @import("std");
const replay = @import("glm_cuda").cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const exec = @import("glm_replay_execute.zig");
const evidence = @import("glm_replay_evidence.zig");

fn sample() fixture.Fixture {
    var f = std.mem.zeroes(fixture.Fixture);
    f.provenance.tokenizer.vocab_size = 8;
    f.config.pool.configured_capacity = 1088;
    f.config.active_streams = &.{0};
    for (&f.streams, 0..) |*s, i| {
        s.stream_id = @intCast(i);
        s.owner = @intCast(i / 2);
        s.prompt_tokens = &.{0};
        s.tape_tokens = &.{ 3, 7, 1, 5, 2 };
    }
    f.forwards = &.{
        .{ .forward_id = 0, .round_id = 0, .parts = &.{.{ .stream_id = 0, .position = 1, .tape_offset = 0, .offered = 4, .kept = 1 }} },
        .{ .forward_id = 1, .round_id = 1, .parts = &.{.{ .stream_id = 0, .position = 2, .tape_offset = 1, .offered = 3, .kept = 3 }} },
    };
    return f;
}

fn reference(f: fixture.Fixture) [2]evidence.Observed {
    var observations = std.mem.zeroes([2]evidence.Observed);
    for (&observations, f.forwards) |*o, r| for (r.parts, 0..) |p, i| {
        @memcpy(o.outputs[i][0..p.kept], f.streams[p.stream_id].tape_tokens[p.tape_offset..][0..p.kept]);
        for (&o.shots[i]) |*shot| {
            shot.position = p.position + p.kept;
            shot.device_position = @intCast(shot.position);
            shot.tap_rows = @intCast(p.kept);
        }
    };
    return observations;
}

fn leakingCache(compiled: []exec.Compiled) void {
    var rows: [16]?u32 = @splat(null);
    for (compiled) |*c| for (c.parts[0..c.n]) |p| for (p.ids, p.out, 0..) |id, *out, row| {
        const at = p.position + row;
        if (rows[at] == null) rows[at] = id;
        out.* = rows[at].?;
    };
}

test "replay poison changes every discarded ID reproducibly without changing tape kept rows or anchors" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = sample();
    const normal = try exec.compile(a, f, f.forwards);
    const left = try exec.compileValidation(a, f, f.forwards);
    const right = try exec.compileValidation(a, f, f.forwards);
    for (normal, left.rounds, right.rounds, f.forwards) |plain, x, y, r| for (r.parts, 0..) |p, i| {
        const ids = x.parts[i].ids;
        try std.testing.expectEqualSlices(u32, ids, y.parts[i].ids);
        try std.testing.expectEqualSlices(u32, plain.parts[i].ids[0..p.kept], ids[0..p.kept]);
        try std.testing.expectEqual(plain.parts[i].anchor, x.parts[i].anchor);
        try std.testing.expectEqual(plain.parts[i].position, x.parts[i].position);
        for (p.kept..p.offered) |row| {
            try std.testing.expect(ids[row] < 8);
            try std.testing.expect(ids[row] != plain.parts[i].ids[row]);
        }
        try std.testing.expectEqualSlices(u32, f.streams[p.stream_id].tape_tokens[p.tape_offset..][0..p.offered], plain.parts[i].ids);
    };
    try exec.checkMeasurementInputs(normal);
    try std.testing.expectError(error.PoisonedMeasurement, exec.checkMeasurementInputs(left.rounds));
    var wrap: [1]u32 = undefined;
    try exec.poison(&wrap, &.{7}, 0, 8);
    try std.testing.expectEqual(@as(u32, 0), wrap[0]);
}

test "replay planted stale lookahead cache passes identical tape but fails poisoned reoffer control" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const f = sample();
    const normal = try exec.compile(a, f, f.forwards);
    const poisoned = try exec.compileValidation(a, f, f.forwards);
    const want = reference(f);
    leakingCache(normal);
    try evidence.checkMeasured(normal, f.forwards, &want);
    leakingCache(poisoned.rounds);
    try std.testing.expectError(error.MeasurementSelection, evidence.checkOutputs(poisoned.rounds, f.forwards, &want));
}

test "replay post-block comparison rejects earlier kept row corruption and final rank state corruption" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const f = sample();
    const compiled = try exec.compile(arena.allocator(), f, f.forwards);
    const want = reference(f);
    for (compiled) |*c| for (c.parts[0..c.n]) |p| @memcpy(p.out, p.ids);
    try evidence.checkMeasured(compiled, f.forwards, &want);
    compiled[0].parts[0].out[0] ^= 1;
    try std.testing.expectError(error.MeasurementSelection, evidence.checkMeasured(compiled, f.forwards, &want));
    compiled[0].parts[0].out[0] ^= 1;
    compiled[1].parts[0].out[2] ^= 1;
    try std.testing.expectError(error.MeasurementSelection, evidence.checkMeasured(compiled, f.forwards, &want));
    var state = want[1].shots[0][1];
    state.recurrent[19] ^= 1;
    try std.testing.expectError(error.ReferenceLogicalState, evidence.compare(want[1].shots[0][1], state, 3));
    state = want[1].shots[0][1];
    state.logits[2][19] ^= 1;
    try std.testing.expectError(error.ReferenceLogits, evidence.compare(want[1].shots[0][1], state, 3));
    var short = want;
    short[0].shots[0][1].tap_rows = 0;
    try std.testing.expectError(error.ReferenceExtent, evidence.checkMeasured(compiled, f.forwards, &short));
}

test "replay continuation clips page-rounded reservation to 64-aligned non512 capacity with verify headroom" {
    const n = try exec.reservationContinuation(3, 1000, 1088);
    try std.testing.expectEqual(@as(usize, 72), n);
    try std.testing.expectEqual(@as(usize, 1088), 1000 + n + 16);
    try std.testing.expectEqual(@as(usize, 1008), try exec.reservationContinuation(4, 1024, 2048));
    try std.testing.expectError(error.Capacity, exec.reservationContinuation(1, 505, 1088));
    try std.testing.expectError(error.Overflow, exec.reservationContinuation(std.math.maxInt(usize), 1, 1088));
}

test "replay typed logical preload refuses schedule or identity drift before device startup" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = sample();
    const digest = "0000000000000000000000000000000000000000000000000000000000000000";
    f.fixture_sha256 = digest;
    f.config_sha256 = digest;
    const want = reference(f);
    var v: evidence.Logical = .{ .schema = "glm-replay-native-logical-v1", .fixture_sha256 = digest, .config_sha256 = digest, .binary_sha256 = digest, .runtime_sha256 = digest, .kv_shard = false, .forwards = f.forwards, .owners = .{ 0, 0, 1, 1 }, .observations = &want };
    var raw = try std.json.Stringify.valueAlloc(a, v, .{});
    _ = try evidence.load(a, raw, f, digest, false);
    try std.testing.expectError(error.ReferenceIdentity, evidence.load(a, raw, f, digest, true));
    v.forwards = f.forwards[0..1];
    raw = try std.json.Stringify.valueAlloc(a, v, .{});
    try std.testing.expectError(error.ReferenceSchedule, evidence.load(a, raw, f, digest, false));
}

test {
    _ = replay;
}

const BlockControl = struct {
    complete_calls: usize = 0,
    forwards: usize = 0,
    active: bool = false,
    corrupt_output: bool = false,
    corrupt_final: bool = false,
    first: ?[]u32 = null,
    rows: u64 = 0,
    kept: u64 = 0,
    records: [4]replay.Forward = undefined,
    shots: [2]replay.Snapshot,

    pub fn prepare(_: *BlockControl, _: []const replay.Stream) !void {}
    pub fn beginBlock(s: *BlockControl) !void {
        if (s.active) return error.UnfinishedReplay;
        s.active = true;
        s.forwards = 0;
        s.rows = 0;
        s.kept = 0;
        s.first = null;
    }
    pub fn round(s: *BlockControl, id: u64, parts: []const replay.Part) !void {
        if (!s.active or id != s.forwards) return error.ReplayForwardId;
        var offered: u64 = 0;
        var kept: u64 = 0;
        for (parts) |p| {
            @memcpy(p.out, p.ids);
            offered += p.ids.len;
            kept += p.keep;
            if (s.first == null) s.first = p.out;
        }
        s.records[s.forwards] = .{ .id = id, .rows = offered, .kept = kept, .counts = .{} };
        s.forwards += 1;
        s.rows += offered;
        s.kept += kept;
    }
    pub fn complete(s: *BlockControl, expected: u64) !replay.Completion {
        if (!s.active or s.forwards != expected) return error.IncompleteReplay;
        if (s.complete_calls == 1 and expected != 2) return error.PerForwardSyncControl;
        s.complete_calls += 1;
        s.active = false;
        @memcpy(s.records[s.forwards..][0..s.forwards], s.records[0..s.forwards]);
        if (s.corrupt_output and s.complete_calls == 2) s.first.?[0] ^= 1;
        const r: replay.Receipt = .{ .forwards = expected, .groups = expected, .rows = s.kept, .calls = 0, .lost = 0, .owner_layers = 0, .owner_rows = 0, .owner_bytes = 0, .carries = 0, .twins = 0 };
        return .{ .ranks = .{ r, r }, .elapsed_ns = 1, .records = s.records[0 .. 2 * s.forwards], .timings = &.{} };
    }
    pub fn capture(s: *BlockControl, _: u32, _: usize) ![2]replay.Snapshot {
        if (s.active or s.complete_calls != 2 or s.forwards != 2) return error.PerForwardSyncControl;
        var shots = s.shots;
        if (s.corrupt_final) shots[1].ring[31] ^= 1;
        return shots;
    }
};

test "replay whole-schedule gate has one completion and catches block-only outputs and final-state faults" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = sample();
    f.preparation = f.forwards[0..1];
    const compiled = try exec.compile(a, f, f.forwards);
    const prep = try exec.compile(a, f, f.preparation);
    const want = reference(f);
    const budget: @import("glm_replay_control.zig").Budget = .{ .half_pages = 256, .logical_pages = 512, .page_argument = 512, .reservation = .{ 1, 1, 1, 1 }, .owner_pages = .{ 2, 2 }, .twin_headroom = 1 };
    const geometry: @import("glm_replay_control.zig").Geometry = .{ .layers = 11, .heads = 32, .latent = 512 };
    var control: BlockControl = .{ .shots = want[1].shots[0] };
    _ = try @import("glm_replay_validate.zig").checkWhole(&control, f, budget, prep, compiled, &want, false, geometry, false);
    try std.testing.expectEqual(@as(usize, 2), control.complete_calls);
    control = .{ .shots = want[1].shots[0], .corrupt_output = true };
    try std.testing.expectError(error.MeasurementSelection, @import("glm_replay_validate.zig").checkWhole(&control, f, budget, prep, compiled, &want, false, geometry, false));
    control = .{ .shots = want[1].shots[0], .corrupt_final = true };
    try std.testing.expectError(error.ReferenceLogicalState, @import("glm_replay_validate.zig").checkWhole(&control, f, budget, prep, compiled, &want, false, geometry, false));
}

test "replay all four uneven parts poison every discarded row and check every kept output" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = sample();
    const tape = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0 };
    for (&f.streams) |*s| s.tape_tokens = &tape;
    var parts: [4]@import("glm_replay_control.zig").Part = undefined;
    const keeps = [_]usize{ 1, 3, 8, 7 };
    for (&parts, keeps, 0..) |*p, kept, i| p.* = .{ .stream_id = @intCast(i), .position = 1, .tape_offset = 0, .offered = 8, .kept = kept };
    const round: fixture.Round = .{ .round_id = 0, .forward_id = 0, .parts = &parts };
    f.forwards = &.{round};
    const compiled = try exec.compile(a, f, f.forwards);
    const poison = try exec.compileValidation(a, f, f.forwards);
    var want = std.mem.zeroes([1]evidence.Observed);
    for (compiled[0].parts[0..4], poison.rounds[0].parts[0..4], parts, 0..) |plain, other, p, i| {
        for (0..p.offered) |row| {
            if (row < p.kept) try std.testing.expectEqual(plain.ids[row], other.ids[row]) else try std.testing.expect(plain.ids[row] != other.ids[row] and other.ids[row] < 8);
        }
        @memcpy(plain.out, plain.ids);
        @memcpy(want[0].outputs[i][0..p.kept], plain.ids[0..p.kept]);
        for (&want[0].shots[i]) |*shot| {
            shot.position = 1 + p.kept;
            shot.device_position = @intCast(shot.position);
            shot.tap_rows = @intCast(p.kept);
        }
    }
    try evidence.checkMeasured(compiled, f.forwards, &want);
    for (compiled[0].parts[0..4], parts) |plain, p| {
        plain.out[p.kept - 1] ^= 1;
        try std.testing.expectError(error.MeasurementSelection, evidence.checkMeasured(compiled, f.forwards, &want));
        plain.out[p.kept - 1] ^= 1;
    }
    parts[0].offered = 16;
    parts[1].offered = 16;
    const wide: fixture.Round = .{ .round_id = 0, .forward_id = 0, .parts = parts[0..2] };
    const wide_poison = try exec.compileValidation(a, f, &.{wide});
    for (wide_poison.rounds[0].parts[0..2], parts[0..2]) |p, meta| for (meta.kept..16) |row| {
        try std.testing.expect(p.ids[row] != tape[row] and p.ids[row] < 8);
    };
}

test "replay block receipt refuses lifetime owner totals even when all forward deltas are correct" {
    const f = sample();
    const g: @import("glm_replay_control.zig").Geometry = .{ .layers = 11, .heads = 32, .latent = 512 };
    const records = [_]replay.Forward{
        .{ .id = 0, .rows = 4, .kept = 1, .counts = .{} },
        .{ .id = 1, .rows = 3, .kept = 3, .counts = .{} },
        .{ .id = 0, .rows = 4, .kept = 1, .counts = .{} },
        .{ .id = 1, .rows = 3, .kept = 3, .counts = .{} },
    };
    const rank: replay.Receipt = .{ .forwards = 2, .groups = 2, .rows = 4, .calls = 0, .lost = 0, .owner_layers = 0, .owner_rows = 0, .owner_bytes = 0, .checks = 3, .carries = 0, .twins = 0 };
    var c: replay.Completion = .{ .ranks = .{ rank, rank }, .records = &records, .timings = &.{}, .elapsed_ns = 1 };
    try exec.checkCompletion(c, f.forwards, false, g);
    c.ranks[1].owner_layers = 99;
    try std.testing.expectError(error.OwnerCounters, exec.checkCompletion(c, f.forwards, false, g));
}
