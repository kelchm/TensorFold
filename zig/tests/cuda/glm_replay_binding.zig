//! Authenticate logical references before startup and retain the bytes used to qualify measurement.
const std = @import("std");
const fixture = @import("glm_replay_fixture.zig");
const meta = @import("glm_replay_metadata.zig");
const evidence = @import("glm_replay_evidence.zig");
const hash = @import("glm_replay_hash.zig");
const result = @import("glm_replay_result.zig");

pub fn load(a: std.mem.Allocator, io: std.Io, m: meta.Metadata, f: fixture.Fixture, sharded: bool) !evidence.Logical {
    const artifact = m.logical_evidence orelse return error.LogicalEvidenceUnrun;
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, artifact.path, a, .limited(512 << 20));
    try hash.matches(artifact.sha256, hash.bytes(raw));
    return evidence.load(a, raw, f, m.source.binary_sha256, sharded);
}

pub fn runtime(reference: evidence.Logical, actual: []const u8) !void {
    if (!std.mem.eql(u8, reference.runtime_sha256, actual)) return error.ReferenceRuntime;
}

pub fn retain(a: std.mem.Allocator, io: std.Io, prefix: []const u8, f: fixture.Fixture, m: meta.Metadata, observations: []const evidence.Observed, comparisons: u64, sharded: bool, runtime_hash: []const u8) !void {
    const logical: evidence.Logical = .{
        .schema = "glm-replay-native-logical-v1",
        .coverage = meta.coverage,
        .fixture_sha256 = f.fixture_sha256,
        .config_sha256 = f.config_sha256,
        .binary_sha256 = m.source.binary_sha256,
        .runtime_sha256 = runtime_hash,
        .kv_shard = sharded,
        .forwards = f.forwards,
        .owners = .{ f.streams[0].owner, f.streams[1].owner, f.streams[2].owner, f.streams[3].owner },
        .observations = observations,
    };
    const raw = try std.json.Stringify.valueAlloc(a, logical, .{});
    const path = try std.fmt.allocPrint(a, "{s}.logical.json", .{prefix});
    try result.writeNew(io, path, raw);
    const digest = hash.bytes(raw);
    inline for (.{ "correctness", "timing_neutrality" }, .{ "correctness", "timing" }) |gate, suffix| {
        const output = try std.fmt.allocPrint(a, "{s}.{s}.json", .{ prefix, suffix });
        try result.gate(a, io, output, f, m, gate, comparisons, sharded, runtime_hash, &digest);
    }
}

pub fn agreement(a: std.mem.Allocator, m: meta.Metadata, settings_hash: []const u8, sharded: bool) ![32]u8 {
    const raw = try std.json.Stringify.valueAlloc(a, .{
        .fixture = m.fixture_sha256,
        .config = m.config_sha256,
        .source = m.source,
        .ids = .{ m.run_id, m.pair_id, m.start_id, m.start_order, m.block_id, m.cell_id },
        .logical = if (m.logical_evidence) |v| v.sha256 else null,
        .correctness = if (m.correctness) |v| v.sha256 else null,
        .sharding_equality = if (m.sharding_equality) |v| v.sha256 else null,
        .timing_neutrality = if (m.timing_neutrality) |v| v.sha256 else null,
        .default_server_neutrality = if (m.default_server_neutrality) |v| v.sha256 else null,
        .telemetry = m.telemetry.sha256,
    }, .{});
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    sha.update("glm-replay-bound-session-v1\x00");
    sha.update(raw);
    sha.update(settings_hash);
    sha.update(&.{@intFromBool(sharded)});
    return sha.finalResult();
}
