//! Local provenance and gate inputs validated before rank-session allocation.
const std = @import("std");
const fixture = @import("glm_replay_fixture.zig");
const hash = @import("glm_replay_hash.zig");
pub const Source = struct {
    engine_commit: []const u8,
    harness_commit: []const u8,
    binary_sha256: []const u8,
    fatbin_sha256: []const u8,
    compiler_sha256: []const u8,
    kit_commit: []const u8,
    image_digest: []const u8,
    device_image_digest: []const u8,
};
pub const Artifact = struct { path: []const u8, sha256: []const u8 };
pub const Metadata = struct {
    schema: []const u8,
    run_id: []const u8,
    pair_id: []const u8,
    start_id: []const u8,
    start_order: u8,
    block_id: []const u8,
    cell_id: []const u8,
    fixture_sha256: []const u8,
    config_sha256: []const u8,
    source: Source,
    fatbin: Artifact,
    compiler: Artifact,
    telemetry: Artifact,
    correctness: ?Artifact,
    sharding_equality: ?Artifact,
    timing_neutrality: ?Artifact,
    default_server_neutrality: ?Artifact,
};
pub const Gate = struct {
    schema: []const u8,
    gate: []const u8,
    status: []const u8,
    fixture_sha256: []const u8,
    config_sha256: []const u8,
    binary_sha256: []const u8,
    completed_ranks: [2]u8,
    comparisons: u64,
    kv_shard: bool,
    runtime_sha256: []const u8,
};

pub fn read(a: std.mem.Allocator, io: std.Io, path: []const u8, f: fixture.Fixture, measure: bool, sharded: bool) !Metadata {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20));
    const m = try std.json.parseFromSliceLeaky(Metadata, a, raw, .{ .allocate = .alloc_always });
    if (!eq(m.schema, "glm-replay-native-metadata-v1") or m.start_order > 1 or m.run_id.len == 0 or m.pair_id.len == 0 or m.start_id.len == 0 or m.block_id.len == 0 or m.cell_id.len == 0) return error.Metadata;
    if (!eq(m.fixture_sha256, f.fixture_sha256) or !eq(m.config_sha256, f.config_sha256)) return error.MetadataFixture;
    if (!eq(m.source.engine_commit, fixture.base_commit)) return error.HistoricalBase;
    if (!eq(m.source.kit_commit, "13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7") or
        !eq(m.source.image_digest, "sha256:0956a18406597abc252564b6f23e7f205dc35c5eac5b475ea6c0c229d10d4d9c") or
        !eq(m.source.device_image_digest, "sha256:ab992b177ec576277720a8acf71d36ed094f56b724d9acb8aebc0d83afad2a34")) return error.SourceProvenance;
    inline for (.{ "engine_commit", "harness_commit", "kit_commit" }) |field| try fixture.hex(@field(m.source, field), 40);
    inline for (.{ "binary_sha256", "fatbin_sha256", "compiler_sha256" }) |field| try fixture.hex(@field(m.source, field), 64);
    inline for (.{ "image_digest", "device_image_digest" }) |field| {
        const s = @field(m.source, field);
        if (!std.mem.startsWith(u8, s, "sha256:")) return error.ImageDigest;
        try fixture.hex(s[7..], 64);
    }
    const executable = try std.process.executablePathAlloc(io, a);
    try hash.matches(m.source.binary_sha256, try fileHash(a, io, executable));
    try artifact(a, io, m.fatbin);
    try artifact(a, io, m.compiler);
    if (!eq(m.fatbin.sha256, m.source.fatbin_sha256) or !eq(m.compiler.sha256, m.source.compiler_sha256)) return error.ArtifactIdentity;
    if (measure) {
        try gate(a, io, m.sharding_equality orelse return error.ShardingEqualityUnrun, m, "sharding_equality", sharded, null);
        try gate(a, io, m.correctness orelse return error.CorrectnessUnrun, m, "correctness", sharded, null);
        try gate(a, io, m.timing_neutrality orelse return error.TimingNeutralityUnrun, m, "timing_neutrality", sharded, null);
        try gate(a, io, m.default_server_neutrality orelse return error.DefaultServerNeutralityUnrun, m, "default_server_neutrality", sharded, null);
    }
    return m;
}

pub fn checkTokenizer(a: std.mem.Allocator, io: std.Io, dir: []const u8, f: fixture.Fixture) !void {
    for (f.provenance.tokenizer.files) |file| {
        const path = try std.fs.path.join(a, &.{ dir, file.path });
        try hash.matches(file.sha256, try fileHash(a, io, path));
    }
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn fileHash(a: std.mem.Allocator, io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const scratch = try a.alloc(u8, 1 << 20);
    defer a.free(scratch);
    var reader = file.reader(io, scratch);
    var sha: std.crypto.hash.sha2.Sha256 = .init(.{});
    while (true) {
        const chunk = reader.interface.peekGreedy(1) catch |err| switch (err) {
            error.EndOfStream => break,
            else => return err,
        };
        sha.update(chunk);
        reader.interface.toss(chunk.len);
    }
    var out: [32]u8 = undefined;
    sha.final(&out);
    return std.fmt.bytesToHex(out, .lower);
}

fn artifact(a: std.mem.Allocator, io: std.Io, v: Artifact) !void {
    try hash.matches(v.sha256, try fileHash(a, io, v.path));
}

pub fn checkRuntime(a: std.mem.Allocator, io: std.Io, m: Metadata, sharded: bool, runtime_hash: []const u8) !void {
    try gate(a, io, m.sharding_equality.?, m, "sharding_equality", sharded, runtime_hash);
    try gate(a, io, m.correctness.?, m, "correctness", sharded, runtime_hash);
    try gate(a, io, m.timing_neutrality.?, m, "timing_neutrality", sharded, runtime_hash);
    try gate(a, io, m.default_server_neutrality.?, m, "default_server_neutrality", sharded, runtime_hash);
}

pub fn checkTelemetry(a: std.mem.Allocator, io: std.Io, m: Metadata) !void {
    try artifact(a, io, m.telemetry);
}

fn gate(a: std.mem.Allocator, io: std.Io, v: Artifact, m: Metadata, name: []const u8, sharded: bool, runtime_hash: ?[]const u8) !void {
    try artifact(a, io, v);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, v.path, a, .limited(1 << 20));
    const g = try std.json.parseFromSliceLeaky(Gate, a, raw, .{});
    try fixture.hex(g.runtime_sha256, 64);
    try checkGate(g, m, name, sharded, runtime_hash);
}

pub fn checkGate(g: Gate, m: Metadata, name: []const u8, sharded: bool, runtime_hash: ?[]const u8) !void {
    if (!eq(g.schema, "glm-replay-native-gate-v1") or !eq(g.gate, name) or !eq(g.status, "pass") or g.comparisons == 0 or g.kv_shard != sharded or (runtime_hash != null and !eq(g.runtime_sha256, runtime_hash.?)) or !std.mem.eql(u8, &g.completed_ranks, &.{ 0, 1 })) return error.GateUnproven;
    if (!eq(g.fixture_sha256, m.fixture_sha256) or !eq(g.config_sha256, m.config_sha256) or !eq(g.binary_sha256, m.source.binary_sha256)) return error.GateIdentity;
}

test "replay gates reject unrun, wrong arm and changed runtime evidence" {
    var m = std.mem.zeroes(Metadata);
    m.fixture_sha256 = "fixture";
    m.config_sha256 = "config";
    m.source.binary_sha256 = "binary";
    var g: Gate = .{ .schema = "glm-replay-native-gate-v1", .gate = "correctness", .status = "pass", .fixture_sha256 = "fixture", .config_sha256 = "config", .binary_sha256 = "binary", .completed_ranks = .{ 0, 1 }, .comparisons = 1, .kv_shard = false, .runtime_sha256 = "runtime" };
    try checkGate(g, m, "correctness", false, "runtime");
    try std.testing.expectError(error.GateUnproven, checkGate(g, m, "correctness", true, "runtime"));
    try std.testing.expectError(error.GateUnproven, checkGate(g, m, "correctness", false, "changed"));
    g.status = "unrun";
    try std.testing.expectError(error.GateUnproven, checkGate(g, m, "correctness", false, "runtime"));
    g.status = "pass";
    g.completed_ranks = .{ 0, 0 };
    try std.testing.expectError(error.GateUnproven, checkGate(g, m, "correctness", false, "runtime"));
    g.completed_ranks = .{ 0, 1 };
    g.binary_sha256 = "other binary";
    try std.testing.expectError(error.GateIdentity, checkGate(g, m, "correctness", false, "runtime"));
}

test "replay native metadata emits string block identities for the common JSONL schema" {
    var m = std.mem.zeroes(Metadata);
    m.block_id = "block-7";
    const raw = try std.json.Stringify.valueAlloc(std.testing.allocator, m, .{});
    defer std.testing.allocator.free(raw);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, raw, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("block-7", parsed.value.object.get("block_id").?.string);
}
