//! JSONL receipts are assembled and written only after both-rank completion.
const std = @import("std");
const glm = @import("glm_cuda");
const replay = glm.cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const ctl = @import("glm_replay_control.zig");
const meta = @import("glm_replay_metadata.zig");
const hash = @import("glm_replay_hash.zig");
const Provenance = @import("glm_replay_provenance.zig").Provenance;
const Phase = struct { staging: f64, target: f64, tap_bookkeeping: f64, selection: f64, target_commit: f64, context_commit: f64 };
const Owner = struct { layer_visits: u64, rows: u64, query_send_bytes: u64, output_send_bytes: u64 };
const Forward = struct {
    forward_id: u64,
    round_id: u64,
    parts: []const ctl.Part,
    phases_ns: Phase,
    owner: Owner,
    graph_replays: u64 = 0,
    eager_fallbacks: u64 = 0,
    target_commits: u64,
    context_commits: u64,
    ingest_calls: u64 = 1,
    proposer_calls: u64 = 0,
    lost: u64 = 0,
};
pub const Effective = struct {
    runtime_sha256: []const u8,
    engine: []const u8 = "native",
    speed: []const u8 = "standard",
    tp: u32 = 2,
    slots: u32 = 4,
    dense_bits: u32 = 8,
    head_bits: u32 = 8,
    kv: []const u8 = "fp8",
    kv_shard: bool,
    multi_lone: u32 = 0,
    pool_h: usize,
    page_argument: usize,
    page_tokens: usize = 512,
    configured_capacity: usize,
    dsa_layers: u64,
    local_heads: u64,
    latent_width: u64,
    row_dtype_bytes: u64 = 2,
    query_bytes_per_row: u64,
    output_bytes_per_row: u64,
    graph_rows: []const u32 = &.{},
};
const Checks = struct {
    correctness: bool = true,
    timing_neutral: bool = true,
    default_server_neutral: bool = true,
    fresh_prefill: bool = true,
    proposer_calls: u64 = 0,
    lost: u64 = 0,
    carries: u64 = 0,
    twins: u64 = 0,
};
const Memory = struct {
    logical_pages: usize,
    key_page_bytes: u64,
    latent_page_bytes: u64,
    row_page_bytes: u64,
    tables_bytes: ?u64 = null,
    scratch_bytes: ?u64 = null,
    other_bytes: ?u64 = null,
    peak_physical_bytes: ?u64 = null,
};
pub const TimingSource = struct {
    wall: []const u8 = "observer_off",
    phases: []const u8 = "separate_observer_on_anchor",
    anchor_sha256: []const u8,
};
const Result = struct {
    schema: []const u8 = "glm-replay-result-v1",
    fixture_sha256: []const u8,
    schedule_sha256: []const u8,
    tokens_sha256: []const u8,
    config_sha256: []const u8,
    run_id: []const u8,
    pair_id: []const u8,
    start_id: []const u8,
    session_nonce: []const u8,
    campaign_sha256: []const u8,
    checkpoints: meta.Checkpoints,
    start_order: u8,
    block_id: []const u8,
    cell_id: []const u8,
    arm: []const u8,
    rank: usize,
    status: []const u8 = "complete",
    exclusion_reason: ?[]const u8 = null,
    source: meta.Source,
    effective_config: Effective,
    effective_config_sha256: []const u8,
    preparation_ns: u64,
    wall_ns: u64,
    layout_checks: u64,
    timing_source: TimingSource,
    completed_ranks: [2]u32 = .{ 0, 1 },
    forwards: []const Forward,
    checks: Checks = .{},
    memory: Memory,
    telemetry_sha256: []const u8,
};

pub fn write(a: std.mem.Allocator, io: std.Io, output: []const u8, f: fixture.Fixture, b: ctl.Budget, m: meta.Metadata, completion: replay.Completion, profile: replay.Completion, anchor_hash: []const u8, prep_ns: u64, sharded: bool, e: *glm.engine.Engine, g: ctl.Geometry, runtime_hash: []const u8, provenance: Provenance, telemetry_hash: []const u8) !void {
    try @import("glm_replay_measure.zig").checkObserver(completion, f.forwards.len, false);
    try @import("glm_replay_measure.zig").checkObserver(profile, f.forwards.len, true);
    const rb = g.heads * g.latent * 2;
    const effective: Effective = .{ .runtime_sha256 = runtime_hash, .kv_shard = sharded, .pool_h = b.half_pages, .page_argument = b.page_argument, .configured_capacity = e.capacity, .dsa_layers = g.layers, .local_heads = g.heads, .latent_width = g.latent, .query_bytes_per_row = rb, .output_bytes_per_row = rb };
    const encoded = try std.json.Stringify.valueAlloc(a, effective, .{});
    const value = try std.json.parseFromSliceLeaky(hash.Value, a, encoded, .{});
    const effective_hash = try hash.digest(a, value);
    const geometry = glm.pages.geometry(e);
    const memory: Memory = .{ .logical_pages = b.logical_pages, .key_page_bytes = geometry.keyPage(), .latent_page_bytes = geometry.latPage(), .row_page_bytes = if (sharded) geometry.bytes(b.half_pages) else b.logical_pages * geometry.wholePage(), .tables_bytes = 4 * glm.pages.tableLen(e.capacity) * 4 * @as(u64, if (sharded) 2 else 1) };
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    const records = try a.alloc(Forward, f.forwards.len);
    for (0..2) |rank| {
        for (records, f.forwards, completion.records[rank * records.len ..][0..records.len], profile.timings[rank * records.len ..][0..records.len]) |*record, r, observed, timed| {
            if (timed.id != r.forward_id) return error.TimingForwardId;
            var owner_rows: u64 = 0;
            var peer_rows: u64 = 0;
            for (r.parts) |p| {
                if (f.streams[p.stream_id].owner == rank) owner_rows += p.offered else peer_rows += p.offered;
            }
            var phase: Phase = undefined;
            inline for (@typeInfo(Phase).@"struct".field_names, 0..) |field, i| {
                const ns = @as(f64, timed.ms[i]) * 1e6;
                if (!std.math.isFinite(ns) or ns < 0) return error.InvalidTiming;
                @field(phase, field) = ns;
            }
            record.* = .{ .forward_id = r.forward_id, .round_id = r.round_id, .parts = r.parts, .phases_ns = phase, .owner = .{ .layer_visits = observed.counts.owner_layers, .rows = observed.counts.owner_rows, .query_send_bytes = if (sharded) g.layers * peer_rows * rb else 0, .output_send_bytes = if (sharded) g.layers * owner_rows * rb else 0 }, .target_commits = observed.kept, .context_commits = observed.kept };
            if (record.owner.query_send_bytes + record.owner.output_send_bytes != observed.counts.owner_bytes) return error.OwnerBytes;
        }
        const rank_receipt = completion.ranks[rank];
        if (rank_receipt.elapsed_ns == 0) return error.IncompleteReplay;
        const result: Result = .{ .fixture_sha256 = f.fixture_sha256, .schedule_sha256 = f.schedule_sha256, .tokens_sha256 = f.tokens_sha256, .config_sha256 = f.config_sha256, .run_id = m.run_id, .pair_id = m.pair_id, .start_id = m.start_id, .session_nonce = provenance.session_nonce, .campaign_sha256 = provenance.campaign_sha256, .checkpoints = provenance.checkpoints, .start_order = m.start_order, .block_id = m.block_id, .cell_id = m.cell_id, .arm = if (sharded) "B" else "A", .rank = rank, .source = m.source, .effective_config = effective, .effective_config_sha256 = &effective_hash, .preparation_ns = prep_ns, .wall_ns = rank_receipt.elapsed_ns, .layout_checks = rank_receipt.checks, .timing_source = .{ .anchor_sha256 = anchor_hash }, .forwards = records, .memory = memory, .telemetry_sha256 = telemetry_hash };
        try std.json.Stringify.value(result, .{}, &out.writer);
        try out.writer.writeByte('\n');
    }
    try writeNew(io, output, out.written());
}

pub fn anchor(a: std.mem.Allocator, io: std.Io, output: []const u8, f: fixture.Fixture, m: meta.Metadata, profile: replay.Completion, headline: replay.Completion, sharded: bool, runtime_hash: []const u8, provenance: Provenance) ![64]u8 {
    try @import("glm_replay_measure.zig").checkObserver(profile, f.forwards.len, true);
    try @import("glm_replay_measure.zig").checkObserver(headline, f.forwards.len, false);
    const raw = try std.json.Stringify.valueAlloc(a, .{
        .schema = "glm-replay-native-phase-anchor-v1",
        .session_nonce = provenance.session_nonce,
        .campaign_sha256 = provenance.campaign_sha256,
        .checkpoints = provenance.checkpoints,
        .fixture_sha256 = f.fixture_sha256,
        .schedule_sha256 = f.schedule_sha256,
        .runtime_sha256 = runtime_hash,
        .logical_evidence_sha256 = m.logical_evidence.?.sha256,
        .source = m.source,
        .ids = .{ .run_id = m.run_id, .pair_id = m.pair_id, .start_id = m.start_id, .start_order = m.start_order, .block_id = m.block_id, .cell_id = m.cell_id },
        .kv_shard = sharded,
        .timing_source = "separate_observer_on_anchor",
        .completed_ranks = [2]u32{ 0, 1 },
        .completion = profile,
        .headline_completion = headline,
    }, .{});
    const path = try std.fmt.allocPrint(a, "{s}.anchor.json", .{output});
    try writeNew(io, path, raw);
    return hash.bytes(raw);
}

pub fn gate(a: std.mem.Allocator, io: std.Io, output: []const u8, f: fixture.Fixture, m: meta.Metadata, name: []const u8, comparisons: u64, sharded: bool, runtime_hash: []const u8, evidence_hash: []const u8, checkpoints: meta.Checkpoints) !void {
    const record: meta.Gate = .{ .schema = "glm-replay-native-gate-v1", .gate = name, .status = "pass", .fixture_sha256 = f.fixture_sha256, .config_sha256 = f.config_sha256, .binary_sha256 = m.source.binary_sha256, .completed_ranks = .{ 0, 1 }, .comparisons = comparisons, .kv_shard = sharded, .runtime_sha256 = runtime_hash, .checkpoints = checkpoints, .coverage = meta.coverage, .evidence_sha256 = evidence_hash };
    const raw = try std.json.Stringify.valueAlloc(a, record, .{ .whitespace = .indent_2 });
    try writeNew(io, output, raw);
}

pub fn writeNew(io: std.Io, path: []const u8, raw: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true });
    defer file.close(io);
    try file.writeStreamingAll(io, raw);
}
