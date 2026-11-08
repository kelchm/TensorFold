//! Planted host controls for checkpoint receipts, local clocks, exclusive output and oracle preparation.
const std = @import("std");
const glm = @import("glm_cuda");
const replay = glm.cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const meta = @import("glm_replay_metadata.zig");
const provenance = @import("glm_replay_provenance.zig");
const evidence = @import("glm_replay_evidence.zig");
const result = @import("glm_replay_result.zig");
const ctl = @import("glm_replay_control.zig");
const zero = "0000000000000000000000000000000000000000000000000000000000000000";
const one = "1111111111111111111111111111111111111111111111111111111111111111";
const checkpoints: provenance.Checkpoints = .{ .scheme = "tensorfold-full-identity-v1", .target = .{ zero, zero }, .drafter = .{ one, one } };

test "replay old or missing checkpoint receipts fail and runtime checks bind actual weights on either rank" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var logical = std.mem.zeroes(evidence.Logical);
    logical.runtime_sha256 = zero;
    logical.checkpoints = checkpoints;
    const binding = @import("glm_replay_binding.zig");
    try binding.runtime(logical, zero, checkpoints);
    inline for (.{ "target", "drafter" }) |field| for (0..2) |rank| {
        var changed = checkpoints;
        @field(changed, field)[rank] = "2222222222222222222222222222222222222222222222222222222222222222";
        try std.testing.expectError(error.CheckpointMismatch, binding.runtime(logical, zero, changed));
        try std.testing.expectError(error.ReferenceRuntime, binding.runtime(logical, one, changed));
    };
    var gate = std.mem.zeroes(meta.Gate);
    gate.checkpoints = checkpoints;
    const gate_raw = try std.json.Stringify.valueAlloc(a, gate, .{});
    var value = try std.json.parseFromSliceLeaky(std.json.Value, a, gate_raw, .{});
    _ = value.object.swapRemove("checkpoints");
    const old = try std.json.Stringify.valueAlloc(a, value, .{});
    try std.testing.expectError(error.MissingField, std.json.parseFromSliceLeaky(meta.Gate, a, old, .{}));
    const cp = try std.json.parseFromSliceLeaky(std.json.Value, a, try std.json.Stringify.valueAlloc(a, checkpoints, .{}), .{});
    var missing_scheme = cp;
    _ = missing_scheme.object.swapRemove("scheme");
    try std.testing.expectError(error.MissingField, std.json.parseFromSliceLeaky(provenance.Checkpoints, a, try std.json.Stringify.valueAlloc(a, missing_scheme, .{}), .{}));
    logical.checkpoints = checkpoints;
    const raw = try std.json.Stringify.valueAlloc(a, logical, .{});
    value = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
    _ = value.object.swapRemove("checkpoints");
    try std.testing.expectError(error.MissingField, evidence.load(a, try std.json.Stringify.valueAlloc(a, value, .{}), std.mem.zeroes(fixture.Fixture), zero, false));
}

test "replay authenticated runtime gates refuse mislabeled actual checkpoints even without expected pins" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var m = std.mem.zeroes(meta.Metadata);
    m.fixture_sha256 = zero;
    m.config_sha256 = zero;
    m.source.binary_sha256 = zero;
    m.logical_evidence = .{ .path = "logical", .sha256 = zero };
    var gate: meta.Gate = .{ .schema = "glm-replay-native-gate-v1", .gate = "sharding_equality", .status = "pass", .fixture_sha256 = zero, .config_sha256 = zero, .binary_sha256 = zero, .completed_ranks = .{ 0, 1 }, .comparisons = 1, .kv_shard = false, .runtime_sha256 = zero, .checkpoints = checkpoints, .coverage = meta.coverage, .evidence_sha256 = zero };
    inline for (.{ "sharding_equality", "correctness", "timing_neutrality" }) |name| {
        gate.gate = name;
        const raw = try std.json.Stringify.valueAlloc(a, gate, .{});
        try tmp.dir.writeFile(io, .{ .sub_path = name, .data = raw });
        const digest = try a.create([64]u8);
        digest.* = @import("glm_replay_hash.zig").bytes(raw);
        @field(m, name) = .{ .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name }), .sha256 = digest };
    }
    try meta.checkRuntime(a, io, m, false, zero, checkpoints);
    inline for (.{ "target", "drafter" }) |field| for (0..2) |rank| {
        var actual = checkpoints;
        @field(actual, field)[rank] = "2222222222222222222222222222222222222222222222222222222222222222";
        try std.testing.expectError(error.CheckpointMismatch, meta.checkRuntime(a, io, m, false, zero, actual));
    };
}

test "replay native writer emits rank-local elapsed and headline layout deltas with authenticated session fields" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/receipt.jsonl", .{tmp.sub_path});
    var f = std.mem.zeroes(fixture.Fixture);
    f.fixture_sha256 = zero;
    f.schedule_sha256 = zero;
    f.tokens_sha256 = zero;
    f.config_sha256 = zero;
    const parts = [_]ctl.Part{.{ .stream_id = 0, .position = 1, .tape_offset = 0, .offered = 1, .kept = 1 }};
    f.forwards = &.{.{ .forward_id = 0, .round_id = 0, .parts = &parts }};
    var m = std.mem.zeroes(meta.Metadata);
    m.logical_evidence = .{ .path = "reference", .sha256 = zero };
    const record: replay.Forward = .{ .id = 0, .rows = 1, .kept = 1, .counts = .{} };
    const records = [_]replay.Forward{ record, record };
    const receipt: replay.Receipt = .{ .forwards = 1, .groups = 1, .rows = 1, .calls = 0, .lost = 0, .owner_layers = 0, .owner_rows = 0, .owner_bytes = 0, .checks = 2, .carries = 0, .twins = 0, .elapsed_ns = 100, .realtime_start_ns = 1000, .realtime_end_ns = 1101 };
    var headline: replay.Completion = .{ .ranks = .{ receipt, receipt }, .elapsed_ns = 100, .records = &records, .timings = &.{} };
    headline.ranks[1].elapsed_ns = 173;
    headline.ranks[1].checks = 5;
    headline.ranks[1].realtime_start_ns = 2000;
    headline.ranks[1].realtime_end_ns = 2174;
    const timings = [_]replay.timing.Record{ .{ .id = 0, .ms = @splat(1) }, .{ .id = 0, .ms = @splat(2) } };
    var profile = headline;
    profile.timings = &timings;
    profile.ranks[0].checks = 20;
    profile.ranks[1].checks = 50;
    var e: glm.engine.Engine = undefined;
    e.capacity = 1088;
    e.w.config = .{};
    e.w.layers = &.{};
    const identity: provenance.Provenance = .{ .session_nonce = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", .campaign_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", .checkpoints = checkpoints };
    const b: ctl.Budget = .{ .half_pages = 256, .logical_pages = 512, .page_argument = 512, .reservation = .{ 1, 1, 1, 1 }, .owner_pages = .{ 2, 2 }, .twin_headroom = 1 };
    const g: ctl.Geometry = .{ .layers = 11, .heads = 32, .latent = 512 };
    const anchor_hash = try result.anchor(a, io, path, f, m, profile, headline, false, zero, identity);
    try result.write(a, io, path, f, b, m, headline, profile, &anchor_hash, 42, false, &e, g, zero, identity, one);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    var lines = std.mem.tokenizeScalar(u8, raw, '\n');
    for (0..2) |rank| {
        const v = try std.json.parseFromSliceLeaky(std.json.Value, a, lines.next().?, .{});
        const completed_ranks = v.object.get("completed_ranks").?.array.items;
        try std.testing.expectEqual(@as(usize, 2), completed_ranks.len);
        try std.testing.expectEqual(@as(i64, 0), completed_ranks[0].integer);
        try std.testing.expectEqual(@as(i64, 1), completed_ranks[1].integer);
        try std.testing.expectEqual(@as(i64, @intCast(headline.ranks[rank].elapsed_ns)), v.object.get("wall_ns").?.integer);
        try std.testing.expectEqual(@as(i64, @intCast(headline.ranks[rank].checks)), v.object.get("layout_checks").?.integer);
        try std.testing.expectEqualStrings(identity.session_nonce, v.object.get("session_nonce").?.string);
        try std.testing.expectEqualStrings(identity.campaign_sha256, v.object.get("campaign_sha256").?.string);
        try std.testing.expectEqualStrings(one, v.object.get("telemetry_sha256").?.string);
        try std.testing.expectEqualStrings(one, v.object.get("checkpoints").?.object.get("drafter").?.array.items[1].string);
    }
    try std.testing.expect(lines.next() == null);
    const anchor_path = try std.fmt.allocPrint(a, "{s}.anchor.json", .{path});
    const anchor_raw = try std.Io.Dir.cwd().readFileAlloc(io, anchor_path, a, .limited(1 << 20));
    const anchor = try std.json.parseFromSliceLeaky(std.json.Value, a, anchor_raw, .{});
    try std.testing.expectEqualStrings(identity.session_nonce, anchor.object.get("session_nonce").?.string);
    const completed_ranks = anchor.object.get("completed_ranks").?.array.items;
    try std.testing.expectEqual(@as(i64, 0), completed_ranks[0].integer);
    try std.testing.expectEqual(@as(i64, 1), completed_ranks[1].integer);
    const ranks = anchor.object.get("headline_completion").?.object.get("ranks").?.array.items;
    try std.testing.expectEqual(@as(i64, 173), ranks[1].object.get("elapsed_ns").?.integer);
    try std.testing.expectEqual(@as(i64, 5), ranks[1].object.get("checks").?.integer);
    try std.testing.expectEqual(@as(i64, 2000), ranks[1].object.get("realtime_start_ns").?.integer);
    try std.testing.expectEqual(@as(i64, 2174), ranks[1].object.get("realtime_end_ns").?.integer);
    try std.testing.expectEqual(@as(i64, 1000), ranks[0].object.get("realtime_start_ns").?.integer);
    try std.testing.expectError(error.PathAlreadyExists, result.writeNew(io, path, "copied stale start"));
    const preserved = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    try std.testing.expectEqualStrings(raw, preserved);
    headline.ranks[1].elapsed_ns = 0;
    try std.testing.expectError(error.IncompleteReplay, result.write(a, io, "unused", f, b, m, headline, profile, &anchor_hash, 42, false, &e, g, zero, identity, one));
}

const Preparation = struct {
    positions: [4]usize = @splat(1),
    tokens: [4]usize = @splat(0),
    sequence: [8]u32 = undefined,
    calls: usize = 0,
    active: bool = false,
    ingested: bool = true,
    fresh: bool = false,
    omit_ingest: bool = false,
    batch_fault: bool = false,

    pub fn prepare(s: *Preparation, streams: []const replay.Stream) !void {
        if (streams.len != 4) return error.NotFresh;
        for (streams, 0..) |stream, i| {
            if (stream.slot != i) return error.NotFresh;
            s.positions[i] = stream.prompt.len;
        }
        s.fresh = true;
    }

    pub fn beginBlock(s: *Preparation) !void {
        if (!s.fresh or s.active or !s.ingested) return error.PreparationOrder;
        s.active = true;
    }

    pub fn round(s: *Preparation, id: u64, parts: []const replay.Part) !void {
        if (!s.active or id != 0) return error.PreparationOrder;
        for (parts) |p| {
            if (p.position != s.positions[p.slot] or p.ids.len != p.out.len) return error.PreparationPosition;
            if (p.keep != 1 or p.ids.len != 1 or parts.len != 1) s.batch_fault = true;
            if (p.ids[0] != 3 + s.tokens[p.slot] or p.anchor != 3 + s.tokens[p.slot] + p.keep) return error.PreparationTape;
            @memcpy(p.out, p.ids);
            s.positions[p.slot] += p.keep;
            s.tokens[p.slot] += p.keep;
            s.sequence[s.calls] = p.slot;
            s.calls += 1;
        }
        s.ingested = !s.omit_ingest;
    }

    pub fn complete(s: *Preparation, expected: u64) !replay.Completion {
        if (!s.active or expected != 1 or !s.ingested) return error.PreparationOrder;
        s.active = false;
        return undefined;
    }
};

test "replay oracle prepares all four slots with ordered solo one-row commits and ingests" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = std.mem.zeroes(fixture.Fixture);
    f.config.pool.configured_capacity = 1088;
    for (&f.streams, 0..) |*stream, i| {
        stream.stream_id = @intCast(i);
        stream.prompt_tokens = &.{0};
        stream.tape_tokens = &.{ 3, 4, 5, 6, 7 };
    }
    f.preparation = &.{.{ .forward_id = 0, .round_id = 0, .parts = &.{
        .{ .stream_id = 2, .position = 1, .tape_offset = 0, .offered = 4, .kept = 2 },
        .{ .stream_id = 0, .position = 1, .tape_offset = 0, .offered = 2, .kept = 1 },
        .{ .stream_id = 3, .position = 1, .tape_offset = 0, .offered = 3, .kept = 3 },
        .{ .stream_id = 1, .position = 1, .tape_offset = 0, .offered = 1, .kept = 1 },
    } }};
    const b: ctl.Budget = .{ .half_pages = 256, .logical_pages = 512, .page_argument = 512, .reservation = .{ 1, 1, 1, 1 }, .owner_pages = .{ 2, 2 }, .twin_headroom = 1 };
    var s: Preparation = .{};
    try @import("glm_replay_validate.zig").prepareOracle(&s, f, b);
    try std.testing.expect(s.fresh and !s.active and s.ingested and !s.batch_fault);
    try std.testing.expectEqualSlices(u32, &.{ 2, 2, 0, 3, 3, 3, 1 }, s.sequence[0..s.calls]);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 3, 4 }, &s.positions);
    s = .{ .omit_ingest = true };
    try std.testing.expectError(error.PreparationOrder, @import("glm_replay_validate.zig").prepareOracle(&s, f, b));
    s = .{};
    const compiled = try @import("glm_replay_execute.zig").compile(arena.allocator(), f, f.preparation);
    try @import("glm_replay_execute.zig").prepare(&s, f, b, compiled);
    try std.testing.expect(s.batch_fault);
}
