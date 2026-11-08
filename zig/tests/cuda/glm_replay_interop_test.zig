//! Actual native formatter output crosses the Python validator using explicitly synthetic CPU state only.
const std = @import("std");
const glm = @import("glm_cuda");
const replay = glm.cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");
const meta = @import("glm_replay_metadata.zig");
const result = @import("glm_replay_result.zig");
const Provenance = @import("glm_replay_provenance.zig").Provenance;
const zero = "0000000000000000000000000000000000000000000000000000000000000000";
const one = "1111111111111111111111111111111111111111111111111111111111111111";

fn emit(a: std.mem.Allocator, io: std.Io, output: []const u8, f: fixture.Fixture, budget: ctl.Budget, sharded: bool) !void {
    var m = std.mem.zeroes(meta.Metadata);
    m.run_id = "synthetic-native-formatter-control";
    m.pair_id = "pair-0";
    m.start_id = if (sharded) "B-start-0" else "A-start-0";
    m.start_order = @intFromBool(sharded);
    m.block_id = "block-0";
    m.cell_id = "synthetic-cell";
    m.source = .{ .engine_commit = fixture.base_commit, .harness_commit = "1111111111111111111111111111111111111111", .binary_sha256 = zero, .fatbin_sha256 = zero, .compiler_sha256 = zero, .kit_commit = "13f2464ee84ab2b5dabf0bb264d1144b02cf5fb7", .image_digest = "sha256:" ++ zero, .device_image_digest = "sha256:" ++ zero };
    m.logical_evidence = .{ .path = "synthetic-unused", .sha256 = zero };
    const identity: Provenance = .{ .session_nonce = if (sharded) one else zero, .campaign_sha256 = one, .checkpoints = .{ .scheme = "tensorfold-full-identity-v1", .target = .{ zero, zero }, .drafter = .{ one, one } } };
    const records = try a.alloc(replay.Forward, 2 * f.forwards.len);
    const timings = try a.alloc(replay.timing.Record, records.len);
    var rows: u64 = 0;
    var groups: u64 = 0;
    for (f.forwards, 0..) |forward, i| {
        var offered: u64 = 0;
        var kept: u64 = 0;
        for (forward.parts) |part| {
            offered += part.offered;
            kept += part.kept;
        }
        rows += offered;
        groups += forward.parts.len;
        for (0..2) |rank| {
            const index = rank * f.forwards.len + i;
            records[index] = .{ .id = forward.forward_id, .rows = offered, .kept = kept, .counts = .{} };
            if (sharded) {
                records[index].counts.owner_layers = 11;
                records[index].counts.owner_rows = 11 * offered;
                records[index].counts.owner_bytes = 11 * offered * 32768;
            }
            timings[index] = .{ .id = forward.forward_id, .ms = @splat(@as(f32, 0.1) + @as(f32, @floatFromInt(rank))) };
        }
    }
    const rank: replay.Receipt = .{ .forwards = f.forwards.len, .groups = groups, .rows = rows, .calls = 0, .lost = 0, .owner_layers = if (sharded) 11 * f.forwards.len else 0, .owner_rows = if (sharded) 11 * rows else 0, .owner_bytes = if (sharded) 11 * rows * 32768 else 0, .checks = @intFromBool(sharded), .carries = 0, .twins = 0, .elapsed_ns = 1000000, .realtime_start_ns = 100, .realtime_end_ns = 200 };
    var headline: replay.Completion = .{ .ranks = .{ rank, rank }, .elapsed_ns = rank.elapsed_ns, .records = records, .timings = &.{} };
    headline.ranks[1].elapsed_ns += 1;
    var profile = headline;
    profile.timings = timings;
    var layers: [11]glm.weights.Layer = undefined;
    for (&layers) |*layer| layer.mixer = .{ .mla = undefined };
    var engine: glm.engine.Engine = undefined;
    engine.capacity = f.config.pool.configured_capacity;
    engine.w.config = .{};
    engine.w.layers = &layers;
    const geometry: ctl.Geometry = .{ .layers = 11, .heads = 32, .latent = 512 };
    const anchor = try result.anchor(a, io, output, f, m, profile, headline, sharded, zero, identity);
    try result.write(a, io, output, f, budget, m, headline, profile, &anchor, 42, sharded, &engine, geometry, zero, identity, zero);
}

test "replay actual native JSONL and anchors pass common Python validation and planted corruption controls" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const fixture_path = "tools/perf/glm_replay/fixtures/host-prose.json";
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, fixture_path, a, .limited(1 << 20));
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    const outputs = [_][]const u8{ try std.fs.path.join(a, &.{ dir, "off.jsonl" }), try std.fs.path.join(a, &.{ dir, "on.jsonl" }) };
    for (outputs, 0..) |output, i| {
        var loaded = try fixture.load(a, raw, true, i == 1);
        defer loaded.deinit();
        try emit(a, io, output, loaded.fixture, loaded.budget, i == 1);
    }
    const checked = try std.process.run(a, io, .{ .argv = &.{ "python3", "-m", "tools.perf.glm_replay.tests.native_interop", fixture_path, outputs[0], outputs[1] } });
    if (!std.meta.eql(checked.term, .{ .exited = 0 })) {
        std.debug.print("native/common interop failed: {s}\n{s}\n", .{ checked.stdout, checked.stderr });
        return error.NativeCommonInterop;
    }
}
