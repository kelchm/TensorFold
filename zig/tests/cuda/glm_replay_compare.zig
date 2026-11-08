//! Driver-free comparison of native off/on logical validation evidence.
const std = @import("std");
const fixture = @import("glm_replay_fixture.zig");
const meta = @import("glm_replay_metadata.zig");
const result = @import("glm_replay_result.zig");
const Logical = @import("glm_replay_evidence.zig").Logical;

pub fn run(a: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len != 2 and args.len != 3) return error.ReplayArguments;
    const off = try read(a, io, args[0]);
    const on = try read(a, io, args[1]);
    try off.checkpoints.require(on.checkpoints);
    if (off.kv_shard or !on.kv_shard or off.forwards.len != on.forwards.len or off.observations.len != off.forwards.len or on.observations.len != on.forwards.len or off.forwards.len == 0) return error.ComparisonArms;
    if (!std.mem.eql(u8, off.fixture_sha256, on.fixture_sha256) or !std.mem.eql(u8, off.config_sha256, on.config_sha256) or !std.mem.eql(u8, off.binary_sha256, on.binary_sha256) or !std.mem.eql(u8, off.runtime_sha256, on.runtime_sha256) or !std.meta.eql(off.owners, on.owners)) return error.ComparisonIdentity;
    for (off.forwards, on.forwards, off.observations, on.observations) |af, bf, aobs, bobs| {
        if (af.forward_id != bf.forward_id or af.round_id != bf.round_id or af.parts.len != bf.parts.len or af.parts.len == 0 or af.parts.len > 4) return error.ComparisonSchedule;
        for (af.parts, bf.parts, 0..) |ap, bp, i| {
            if (!std.meta.eql(ap, bp) or ap.stream_id >= 4 or ap.kept == 0 or ap.kept > 16) return error.ComparisonSchedule;
            if (!std.mem.eql(u32, aobs.outputs[i][0..ap.kept], bobs.outputs[i][0..ap.kept])) return error.ShardingSelection;
            for (0..2) |rank| {
                const astate = aobs.shots[i][rank];
                const bstate = bobs.shots[i][rank];
                const owner = on.owners[ap.stream_id] == rank;
                if (astate.has_latent != 1 or bstate.has_latent != @intFromBool(owner)) return error.ShardingOwner;
                inline for (.{ "position", "device_position", "ring_from", "index_from", "conv", "recurrent", "index_keys", "index_gates", "pooled_keys", "ring" }) |field| {
                    if (!std.meta.eql(@field(astate, field), @field(bstate, field))) return error.ShardingLogicalState;
                }
                if (owner and (!std.meta.eql(astate.latent_codes, bstate.latent_codes) or !std.meta.eql(astate.latent_scales, bstate.latent_scales))) return error.ShardingLatentState;
                for (0..ap.kept) |r| {
                    if (!std.meta.eql(astate.taps[r], bstate.taps[r]) or !std.meta.eql(astate.logits[r], bstate.logits[r])) return error.ShardingRows;
                }
            }
        }
    }
    if (args.len == 3) for ([_]bool{ false, true }) |sharded| {
        const path = try std.fmt.allocPrint(a, "{s}.sharding-{s}.json", .{ args[2], if (sharded) "on" else "off" });
        const evidence_hash = try meta.fileHash(a, io, args[@intFromBool(sharded)]);
        const gate: meta.Gate = .{ .schema = "glm-replay-native-gate-v1", .gate = "sharding_equality", .status = "pass", .fixture_sha256 = off.fixture_sha256, .config_sha256 = off.config_sha256, .binary_sha256 = off.binary_sha256, .completed_ranks = .{ 0, 1 }, .comparisons = 2 * off.forwards.len, .kv_shard = sharded, .runtime_sha256 = off.runtime_sha256, .checkpoints = off.checkpoints, .coverage = meta.coverage, .evidence_sha256 = &evidence_hash };
        const raw = try std.json.Stringify.valueAlloc(a, gate, .{ .whitespace = .indent_2 });
        try result.writeNew(io, path, raw);
    };
    std.debug.print("PASS glm-replay-compare: {d} forwards, both ranks' logical target/taps/live-ring equal off/on\n", .{off.forwards.len});
}

fn read(a: std.mem.Allocator, io: std.Io, path: []const u8) !Logical {
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(512 << 20));
    const v = try std.json.parseFromSliceLeaky(Logical, a, raw, .{});
    try v.checkpoints.check();
    if (!std.mem.eql(u8, v.schema, "glm-replay-native-logical-v1")) return error.Schema;
    if (!std.mem.eql(u8, v.coverage, meta.coverage)) return error.ReferenceCoverage;
    try @import("glm_replay_evidence.zig").checkObservations(v.forwards, v.observations);
    try fixture.hex(v.runtime_sha256, 64);
    try fixture.hex(v.fixture_sha256, 64);
    try fixture.hex(v.config_sha256, 64);
    try fixture.hex(v.binary_sha256, 64);
    return v;
}
