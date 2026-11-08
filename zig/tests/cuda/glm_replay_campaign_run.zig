//! A prescribed sequence of blocks sharing exactly one production rank session.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const campaign = @import("glm_replay_campaign.zig");
const runner = @import("glm_replay.zig");
const exec = @import("glm_replay_execute.zig");
const meta = @import("glm_replay_metadata.zig");
const result = @import("glm_replay_result.zig");
const binding = @import("glm_replay_binding.zig");
const Plan = struct { prep: []exec.Compiled, block: []exec.Compiled, reference: ?@import("glm_replay_evidence.zig").Logical = null };
const Progress = struct {
    rank: ?usize = null,
    completed: usize = 0,
    active: ?usize = null,
    ordered_hash: ?[64]u8 = null,
    session_open: bool = false,
};

pub fn run(init: std.process.Init, args: []const [:0]const u8) !void {
    var progress: Progress = .{};
    execute(init, args, &progress) catch |err| {
        failure(init, args, progress, err);
        return err;
    };
}

fn execute(init: std.process.Init, args: []const [:0]const u8, progress: *Progress) !void {
    if (args.len != 5 or !std.mem.eql(u8, args[0], "measure-many")) return error.ReplayCampaignArguments;
    const a = init.arena.allocator();
    const place = try glm.native.place();
    progress.rank = place.rank;
    if (place.world != 2) return error.ReplayNeedsTP2;
    const set = try runner.settings();
    try campaign.checkPages(glm.ranks.environment("TF_KV_PAGES"));
    const sharded = set.kv_shard.on;
    var all = try campaign.load(init.gpa, init.io, args[1], args[2], sharded);
    defer all.deinit();
    progress.ordered_hash = std.fmt.bytesToHex(all.ordered_hash, .lower);
    const receipt = try failurePath(a, args[4], place.rank);
    const canonical_receipt = try campaign.outputPath(a, init.io, receipt);
    const plans = try a.alloc(Plan, all.cases.len);
    var max_forwards: usize = 0;
    for (all.cases, plans, 0..) |c, *p, i| {
        if (std.mem.eql(u8, c.output, canonical_receipt)) return error.CampaignDuplicateOutput;
        try runner.checkModel(a, init.io, args[2], c.f);
        p.* = for (all.cases[0..i], 0..) |prior, j| {
            if (std.mem.eql(u8, c.f.fixture_sha256, prior.f.fixture_sha256)) break plans[j];
        } else .{ .prep = try exec.compile(a, c.f, c.f.preparation), .block = try exec.compile(a, c.f, c.f.forwards) };
        p.reference = try binding.load(a, init.io, c.m, c.f, sharded);
        max_forwards = @max(max_forwards, @max(p.prep.len, p.block.len));
    }
    const settings_hash = try glm.cuda_replay.runtime.settingsHash(a, set, &.{});
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    sha.update("glm-replay-campaign-settings-v1\x00");
    sha.update(&all.ordered_hash);
    sha.update(&settings_hash);
    sha.update(&.{@intFromBool(sharded)});
    const config_hash = sha.finalResult();
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const f = all.cases[0].f;
    const s = try glm.cuda_replay.Session.open(init.gpa, init.io, &ctx, args[2], .{
        .context = f.config.pool.configured_capacity - 16,
        .kv8 = true,
        .drafts = true,
        .drafter = args[3],
        .drafter_bits = 4,
    }, .{ .fixture_hash = all.ordered_hash, .config_hash = config_hash, .half_pages = f.config.pool.h, .max_forwards = max_forwards, .observe = false });
    defer s.close();
    progress.session_open = true;
    drive(init, s, all.cases, plans, sharded, progress) catch |err| {
        failure(init, args, progress.*, err);
        s.failed = true;
        cuda.tp_watch.fail("replay campaign failed");
        return err;
    };
}

fn drive(init: std.process.Init, s: *glm.cuda_replay.Session, cases: []const campaign.Case, plans: []const Plan, sharded: bool, progress: *Progress) !void {
    const runtime_hash = std.fmt.bytesToHex(s.runtime_sha256, .lower);
    const identity = try @import("glm_replay_provenance.zig").Provenance.fromSession(init.arena.allocator(), s);
    for (cases, plans) |c, p| {
        var gate_arena = std.heap.ArenaAllocator.init(init.gpa);
        defer gate_arena.deinit();
        if (c.m.expected_checkpoints) |pins| try pins.require(identity.checkpoints);
        try meta.checkRuntime(gate_arena.allocator(), init.io, c.m, sharded, &runtime_hash, identity.checkpoints);
        try binding.runtime(p.reference.?, &runtime_hash, identity.checkpoints);
    }
    try s.agree();
    if (progress.rank.? != 0) return s.follow();
    if (s.owned.e.capacity != cases[0].f.config.pool.configured_capacity) return error.LoadedConfiguration;
    const g = try exec.geometry(s.owned.e);
    if (g.layers != 11 or g.heads != 32 or g.latent != 512) return error.LoadedGeometry;
    for (cases, plans, 0..) |c, p, i| {
        progress.active = i;
        if (s.owned.e.c.vocab < c.f.provenance.tokenizer.vocab_size) return error.LoadedConfiguration;
        var block_arena = std.heap.ArenaAllocator.init(init.gpa);
        defer block_arena.deinit();
        var block_init = init;
        block_init.arena = &block_arena;
        try runner.measureBlock(block_init, c.output, s, c.f, c.budget, c.m, sharded, p.prep, p.block, g, &runtime_hash, p.reference.?.observations);
        progress.completed += 1;
        progress.active = null;
    }
    std.debug.print("PASS glm-replay measure-many: {d} blocks in one rank session, ordered SHA256 {s}\n", .{ cases.len, &progress.ordered_hash.? });
}

fn failurePath(a: std.mem.Allocator, prefix: []const u8, rank: usize) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}.rank-{d}.failed.json", .{ prefix, rank });
}

fn failure(init: std.process.Init, args: []const [:0]const u8, progress: Progress, err: anyerror) void {
    if (args.len != 5) return;
    const a = init.arena.allocator();
    const path = failurePath(a, args[4], progress.rank orelse 0) catch return;
    const raw = std.json.Stringify.valueAlloc(a, .{
        .schema = "glm-replay-native-campaign-attempt-v1",
        .status = "failed",
        .failure = @errorName(err),
        .measurement_complete = false,
        .rank = progress.rank,
        .session_open = progress.session_open,
        .completed_output_cases = if (progress.rank == 0) @as(?usize, progress.completed) else null,
        .active_case_index = progress.active,
        .ordered_sha256 = if (progress.ordered_hash) |*v| @as(?[]const u8, v) else null,
    }, .{}) catch return;
    result.writeNew(init.io, path, raw) catch {};
}
