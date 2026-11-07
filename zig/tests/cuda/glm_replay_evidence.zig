//! Parsed logical references and post-completion correctness checks, never inside the timed block.
const std = @import("std");
const replay = @import("glm_cuda").cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const exec = @import("glm_replay_execute.zig");

pub const Observed = struct {
    shots: [4][2]replay.Snapshot = undefined,
    outputs: [4][16]u32 = undefined,
    records: [2]replay.Forward = undefined,
};

pub const Logical = struct {
    schema: []const u8,
    coverage: []const u8,
    fixture_sha256: []const u8,
    config_sha256: []const u8,
    binary_sha256: []const u8,
    runtime_sha256: []const u8,
    kv_shard: bool,
    forwards: []const fixture.Round,
    owners: [4]u8,
    observations: []const Observed,
};

pub fn load(a: std.mem.Allocator, raw: []const u8, f: fixture.Fixture, binary: []const u8, sharded: bool) !Logical {
    const v = try std.json.parseFromSliceLeaky(Logical, a, raw, .{});
    if (!std.mem.eql(u8, v.schema, "glm-replay-native-logical-v1")) return error.Schema;
    if (!std.mem.eql(u8, v.coverage, @import("glm_replay_metadata.zig").coverage)) return error.ReferenceCoverage;
    try fixture.hex(v.runtime_sha256, 64);
    if (!std.mem.eql(u8, v.fixture_sha256, f.fixture_sha256) or !std.mem.eql(u8, v.config_sha256, f.config_sha256) or
        !std.mem.eql(u8, v.binary_sha256, binary) or v.kv_shard != sharded) return error.ReferenceIdentity;
    for (f.streams, v.owners) |s, owner| if (s.owner != owner) return error.ReferenceIdentity;
    if (v.forwards.len != f.forwards.len) return error.ReferenceSchedule;
    for (v.forwards, f.forwards) |got, want| {
        if (got.forward_id != want.forward_id or got.round_id != want.round_id or
            got.parts.len != want.parts.len) return error.ReferenceSchedule;
        for (got.parts, want.parts) |gp, wp| if (!std.meta.eql(gp, wp)) return error.ReferenceSchedule;
    }
    try checkObservations(f.forwards, v.observations);
    for (f.forwards, v.observations) |r, o| for (r.parts, 0..) |p, i| {
        for (o.outputs[i][0..p.kept]) |id| if (id >= f.provenance.tokenizer.vocab_size) return error.ReferenceVocabulary;
        for (o.shots[i], 0..) |shot, rank| {
            const owns = !sharded or v.owners[p.stream_id] == rank;
            if (shot.has_latent != @intFromBool(owns) or shot.ring_from > shot.position or
                shot.index_from > shot.position) return error.ReferenceExtent;
        }
    };
    return v;
}

pub fn checkObservations(rounds: []const fixture.Round, observations: []const Observed) !void {
    if (rounds.len == 0 or rounds.len != observations.len) return error.ReferenceSchedule;
    for (rounds, observations) |r, o| {
        if (r.parts.len == 0 or r.parts.len > 4) return error.ReferenceSchedule;
        for (r.parts, 0..) |p, i| {
            if (p.stream_id >= 4 or p.kept == 0 or p.kept > 16 or p.kept > p.offered) return error.ReferenceSchedule;
            for (o.shots[i]) |shot| if (shot.tap_rows != p.kept or shot.position != p.position + p.kept or
                shot.device_position != p.position + p.kept) return error.ReferenceExtent;
        }
    }
}

pub fn checkMeasured(compiled: []const exec.Compiled, rounds: []const fixture.Round, observations: []const Observed) !void {
    try exec.checkMeasurementInputs(compiled);
    try checkOutputs(compiled, rounds, observations);
}

pub fn checkOutputs(compiled: anytype, rounds: []const fixture.Round, observations: []const Observed) !void {
    try checkObservations(rounds, observations);
    if (compiled.len != rounds.len) return error.ReferenceSchedule;
    for (compiled, rounds, observations) |c, r, o| {
        if (c.n != r.parts.len) return error.ReferenceSchedule;
        for (c.parts[0..c.n], r.parts, 0..) |part, p, i| {
            if (part.slot != p.stream_id or part.position != p.position or part.ids.len != p.offered or
                part.keep != p.kept or part.out.len != p.offered) return error.ReferenceSchedule;
            if (!std.mem.eql(u32, part.out[0..p.kept], o.outputs[i][0..p.kept])) return error.MeasurementSelection;
        }
    }
}

pub fn logical(a: replay.Snapshot, b: replay.Snapshot) !void {
    inline for (.{ "position", "device_position", "ring_from", "index_from", "has_latent", "conv", "recurrent", "latent_codes", "latent_scales", "index_keys", "index_gates", "pooled_keys", "ring" }) |field| {
        if (!std.meta.eql(@field(a, field), @field(b, field))) return error.ReferenceLogicalState;
    }
}

pub fn rows(a: replay.Snapshot, ai: usize, b: replay.Snapshot, bi: usize) !void {
    if (ai >= a.tap_rows or bi >= b.tap_rows) return error.TapExtent;
    if (!std.mem.eql(u8, &a.taps[ai], &b.taps[bi])) return error.ReferenceTaps;
    if (!std.mem.eql(u8, &a.logits[ai], &b.logits[bi])) return error.ReferenceLogits;
}

pub fn compare(a: replay.Snapshot, b: replay.Snapshot, kept: usize) !void {
    try logical(a, b);
    if (a.tap_rows != kept or b.tap_rows != kept) return error.TapExtent;
    for (0..kept) |i| try rows(a, i, b, i);
}
