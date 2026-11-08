//! Build a post-completion request from actual gathered clocks and the exact measured block bindings.
const std = @import("std");
const telemetry = @import("glm_replay_telemetry.zig");
const fixture = @import("glm_replay_fixture.zig");
const meta = @import("glm_replay_metadata.zig");
const Provenance = @import("glm_replay_provenance.zig").Provenance;

pub fn window(a: std.mem.Allocator, f: fixture.Fixture, m: meta.Metadata, completion: anytype, sharded: bool, runtime_hash: []const u8, identity: Provenance) !telemetry.Window {
    const source = m.source_manifest orelse return error.SourceManifestUnrun;
    const result: telemetry.Window = .{
        .bindings = .{
            .fixture_sha256 = f.fixture_sha256,
            .config_sha256 = f.config_sha256,
            .schedule_sha256 = f.schedule_sha256,
            .tokens_sha256 = f.tokens_sha256,
            .arm = if (sharded) "B" else "A",
            .source_sha256 = try a.dupe(u8, &try telemetry.digest(a, m.source)),
            .source_tree_sha256 = source.sha256,
            .runtime_sha256 = runtime_hash,
            .checkpoints_sha256 = try a.dupe(u8, &try telemetry.digest(a, identity.checkpoints)),
        },
        .ids = .{ .run_id = m.run_id, .pair_id = m.pair_id, .start_id = m.start_id, .start_order = m.start_order, .cell_id = m.cell_id, .block_id = m.block_id },
        .ranks = .{
            .{ .rank = 0, .realtime_start_ns = completion.ranks[0].realtime_start_ns, .realtime_end_ns = completion.ranks[0].realtime_end_ns },
            .{ .rank = 1, .realtime_start_ns = completion.ranks[1].realtime_start_ns, .realtime_end_ns = completion.ranks[1].realtime_end_ns },
        },
        .session_nonce = identity.session_nonce,
        .campaign_sha256 = identity.campaign_sha256,
        .checkpoints = identity.checkpoints,
    };
    try telemetry.checkWindow(result);
    return result;
}
