//! Invented host controls exercise the actual post-block telemetry parser and path guards.
const std = @import("std");
const t = @import("glm_replay_telemetry.zig");
const sha = "1111111111111111111111111111111111111111111111111111111111111111";
const devices: [2][]const u8 = .{ "GPU-host-control-0", "GPU-host-control-1" };

fn window() t.Window {
    var bindings: t.Bindings = undefined;
    inline for (@typeInfo(t.Bindings).@"struct".field_names) |key| @field(bindings, key) = sha;
    bindings.arm = "A";
    return .{ .bindings = bindings, .ids = .{ .run_id = "run", .pair_id = "pair", .start_id = "start", .start_order = 0, .cell_id = "cell", .block_id = "block" }, .ranks = .{ .{ .rank = 0, .realtime_start_ns = 100, .realtime_end_ns = 200 }, .{ .rank = 1, .realtime_start_ns = 110, .realtime_end_ns = 190 } }, .session_nonce = sha, .campaign_sha256 = sha, .checkpoints = .{ .scheme = "tensorfold-full-identity-v1", .target = .{ sha, sha }, .drafter = .{ sha, sha } } };
}

fn telemetry(w: t.Window, hash: []const u8, data: []const t.Sample) t.Telemetry {
    return .{ .schema = "glm-replay-telemetry-v1", .status = "measured", .window_sha256 = hash, .bindings = w.bindings, .ids = w.ids, .completed_ranks = .{ 0, 1 }, .ranks = .{ .{ .rank = 0, .device_uuid = devices[0], .realtime_start_ns = 100, .realtime_end_ns = 200, .max_sample_gap_ns = 1000, .samples = data }, .{ .rank = 1, .device_uuid = devices[1], .realtime_start_ns = 110, .realtime_end_ns = 190, .max_sample_gap_ns = 1000, .samples = data } }, .session_nonce = w.session_nonce, .campaign_sha256 = w.campaign_sha256, .checkpoints = w.checkpoints };
}

fn samples() [3]t.Sample {
    var result: [3]t.Sample = undefined;
    for (&result, [_]u64{ 90, 120, 210 }) |*s, time|
        s.* = .{ .realtime_ns = time, .sm_clock_mhz = 1000, .memory_clock_mhz = 1000, .temperature_millicelsius = 40000, .power_milliwatts = 1000, .used_memory_bytes = 1000 };
    return result;
}

test "native telemetry refuses stubs, changed windows, devices, samples, time gaps and stale sessions" {
    const a = std.testing.allocator;
    const w = window();
    const hash = try t.digest(a, w);
    try std.testing.expectEqualStrings("be32189e1e06a002781dcf11a9ccf5922cdbba7b223316a1a5447b0e8176349d", &hash);
    var data = samples();
    const original = telemetry(w, &hash, &data);
    var actual = original;
    try t.check(a, actual, w, devices);
    actual.ranks[1].samples = &.{};
    try std.testing.expectError(error.TelemetrySamples, t.check(a, actual, w, devices));
    actual = original;
    actual.status = "unrun";
    try std.testing.expectError(error.TelemetryUnmeasured, t.check(a, actual, w, devices));
    actual = original;
    actual.window_sha256 = sha;
    try std.testing.expectError(error.HashMismatch, t.check(a, actual, w, devices));
    actual = original;
    actual.ids.block_id = "other-block";
    try std.testing.expectError(error.TelemetryBinding, t.check(a, actual, w, devices));
    actual = original;
    actual.session_nonce = "2222222222222222222222222222222222222222222222222222222222222222";
    try std.testing.expectError(error.TelemetryBinding, t.check(a, actual, w, devices));
    actual = original;
    actual.ranks[1].device_uuid = devices[0];
    try std.testing.expectError(error.TelemetryRank, t.check(a, actual, w, devices));
    actual = original;
    actual.ranks[0].realtime_start_ns = 101;
    try std.testing.expectError(error.TelemetryRank, t.check(a, actual, w, devices));
    actual = original;
    actual.ranks[0].max_sample_gap_ns = 10;
    try std.testing.expectError(error.TelemetryGap, t.check(a, actual, w, devices));
    data[1].realtime_ns = 90;
    try std.testing.expectError(error.TelemetryGap, t.check(a, original, w, devices));
    data = samples();
    data[1].realtime_ns = 95;
    try std.testing.expectError(error.TelemetryCoverage, t.check(a, original, w, devices));
}

test "native future telemetry rejects prehashed specs unsafe paths existing files and symlink traversal" {
    for ([_][]const u8{ "", "/absolute", "../bad", "dir/../bad", "dir//bad", "./bad", "dir\\bad" }) |path|
        try std.testing.expectError(error.TelemetryPath, t.checkPath(path));
    try t.checkPath("blocks/cell.json");
    try std.testing.expectError(error.UnknownField, std.json.parseFromSlice(t.Ref, std.testing.allocator, "{\"path\":\"block.json\",\"sha256\":\"old-prehashed\"}", .{}));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/block.json", .{tmp.sub_path});
    const ref: t.Ref = .{ .path = path };
    try t.preflight(a, io, ref);
    try tmp.dir.writeFile(io, .{ .sub_path = "block.json", .data = "preserve" });
    try std.testing.expectError(error.TelemetryOutputExists, t.preflight(a, io, ref));
    try tmp.dir.deleteFile(io, "block.json");
    try tmp.dir.symLink(io, "missing-target", "block.json", .{});
    try std.testing.expectError(error.TelemetrySymlink, t.preflight(a, io, ref));
}

test "native post-block delivery writes window once and hashes the bytes actually parsed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/block.json", .{tmp.sub_path});
    const ref: t.Ref = .{ .path = path };
    try t.preflight(a, io, ref);
    const w = window();
    const hash = try t.digest(a, w);
    const data = samples();
    const raw = try std.json.Stringify.valueAlloc(a, telemetry(w, &hash, &data), .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "block.json", .data = raw });
    const got = try t.receive(a, io, ref, w, devices, 0);
    try std.testing.expectEqual(@import("glm_replay_hash.zig").bytes(raw), got);
    const request = try tmp.dir.readFileAlloc(io, "block.json.window.json", a, .limited(1 << 20));
    const parsed = try std.json.parseFromSliceLeaky(t.Window, a, request, .{});
    try t.checkWindow(parsed);
    try std.testing.expectError(error.PathAlreadyExists, t.receive(a, io, ref, w, devices, 0));
    const absent: t.Ref = .{ .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/missing.json", .{tmp.sub_path}) };
    try std.testing.expectError(error.TelemetryUnavailable, t.receive(a, io, absent, w, devices, 0));
}

test "native telemetry and source manifests parse only the exact authenticated file bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const source = .{ .harness_commit = "1111111111111111111111111111111111111111" };
    const value = .{ .schema = "glm-replay-native-source-v1", .base_commit = @import("glm_replay_fixture.zig").base_commit, .harness_commit = source.harness_commit, .files = .{ .{ .path = "zig/src/families/glm/cuda_replay.zig", .bytes = 1, .sha256 = sha }, .{ .path = "zig/tests/cuda/glm_replay.zig", .bytes = 1, .sha256 = sha } } };
    const raw = try std.json.Stringify.valueAlloc(a, value, .{});
    try tmp.dir.writeFile(io, .{ .sub_path = "source.json", .data = raw });
    const expected = @import("glm_replay_hash.zig").bytes(raw);
    const ref = .{ .path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/source.json", .{tmp.sub_path}), .sha256 = &expected };
    try t.sourceManifest(a, io, ref, source);
    try tmp.dir.writeFile(io, .{ .sub_path = "source.json", .data = try std.fmt.allocPrint(a, "{s}\n", .{raw}) });
    try std.testing.expectError(error.HashMismatch, t.sourceManifest(a, io, ref, source));
}

test "native telemetry request uses actual rank windows source export and measured identity" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const expected = window();
    var f = std.mem.zeroes(@import("glm_replay_fixture.zig").Fixture);
    inline for (.{ "fixture_sha256", "config_sha256", "schedule_sha256", "tokens_sha256" }) |key| @field(f, key) = sha;
    const meta = @import("glm_replay_metadata.zig");
    var m = std.mem.zeroes(meta.Metadata);
    inline for (@typeInfo(t.Ids).@"struct".field_names) |key| @field(m, key) = @field(expected.ids, key);
    inline for (@typeInfo(meta.Source).@"struct".field_names) |key| @field(m.source, key) = sha;
    m.source_manifest = .{ .path = "source.json", .sha256 = sha };
    const identity: @import("glm_replay_provenance.zig").Provenance = .{ .session_nonce = expected.session_nonce, .campaign_sha256 = expected.campaign_sha256, .checkpoints = expected.checkpoints };
    var completion = .{ .ranks = expected.ranks };
    const got = try @import("glm_replay_delivery.zig").window(a, f, m, completion, true, sha, identity);
    try std.testing.expectEqualDeep(expected.ranks, got.ranks);
    try std.testing.expectEqualStrings("B", got.bindings.arm);
    try std.testing.expectEqualStrings(sha, got.bindings.source_tree_sha256);
    try std.testing.expectEqualStrings(&try t.digest(a, m.source), got.bindings.source_sha256);
    try std.testing.expectEqualStrings(&try t.digest(a, identity.checkpoints), got.bindings.checkpoints_sha256);
    completion.ranks[1].realtime_start_ns = 0;
    try std.testing.expectError(error.TelemetryWindow, @import("glm_replay_delivery.zig").window(a, f, m, completion, false, sha, identity));
    m.source_manifest = null;
    try std.testing.expectError(error.SourceManifestUnrun, @import("glm_replay_delivery.zig").window(a, f, m, completion, false, sha, identity));
}
