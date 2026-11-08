//! Local GLM fixed-work replay command; malformed inputs fail before CUDA opens.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const replay = glm.cuda_replay;
const fixture = @import("glm_replay_fixture.zig");
const metadata = @import("glm_replay_metadata.zig");
const exec = @import("glm_replay_execute.zig");
const validation = @import("glm_replay_validate.zig");
const result = @import("glm_replay_result.zig");
const binding = @import("glm_replay_binding.zig");
const provenance = @import("glm_replay_provenance.zig");

pub fn command(init: std.process.Init, cmd: []const u8, args: []const [:0]const u8) !u8 {
    run(init, cmd, args) catch |err| {
        failure(init, cmd, args, err);
        std.debug.print("FAIL {s}: {t}\n", .{ cmd, err });
        return 1;
    };
    return 0;
}

fn run(init: std.process.Init, cmd: []const u8, args: []const [:0]const u8) !void {
    const a = init.arena.allocator();
    const compiler = @import("builtin").zig_version;
    if (compiler.major != 0 or compiler.minor != 17) return error.ReplayCompiler;
    if (std.mem.eql(u8, cmd, "glm-replay") and args.len != 0 and std.mem.eql(u8, args[0], "measure-many"))
        return @import("glm_replay_campaign_run.zig").run(init, args);
    if (std.mem.eql(u8, cmd, "glm-replay-compare")) return @import("glm_replay_compare.zig").run(a, init.io, args);
    if (std.mem.eql(u8, cmd, "glm-replay-check")) {
        if (args.len != 1) return error.ReplayArguments;
        const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, args[0], a, .limited(512 << 20));
        var loaded = try fixture.load(init.gpa, raw, true, false);
        defer loaded.deinit();
        std.debug.print("PASS glm-replay-check {s}: {d} forwards, four prepared slots, hashes and pool validated\n", .{ loaded.fixture.fixture_sha256, loaded.fixture.forwards.len });
        return;
    }
    if (args.len != 6) return error.ReplayArguments;
    const measure = std.mem.eql(u8, args[0], "measure");
    if (!measure and !std.mem.eql(u8, args[0], "validate")) return error.ReplayMode;
    const set = try settings();
    const sharded = set.kv_shard.on;
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(512 << 20));
    var loaded = try fixture.load(init.gpa, raw, false, sharded);
    defer loaded.deinit();
    const f = loaded.fixture;
    const preliminary_hash = try replay.runtime.settingsHash(a, set, &.{});
    const m = try metadata.read(a, init.io, args[5], f, measure, sharded);
    try metadata.checkTelemetry(a, init.io, m);
    const reference = if (measure) try binding.load(a, init.io, m, f, sharded) else null;
    const campaign = @import("glm_replay_campaign.zig");
    if (measure) try campaign.checkOutput(a, init.io, args[4]) else try campaign.checkValidationOutput(a, init.io, args[4]);
    try metadata.checkTokenizer(a, init.io, args[2], f);
    const prep = try exec.compile(a, f, f.preparation);
    const compiled = try exec.compile(a, f, f.forwards);
    try checkModel(a, init.io, args[2], f);
    const place = try glm.native.place();
    if (place.world != 2) return error.ReplayNeedsTP2;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const capacity = f.config.pool.configured_capacity;
    const s = try replay.Session.open(init.gpa, init.io, &ctx, args[2], .{ .context = capacity - 16, .kv8 = true, .drafts = true, .drafter = args[3], .drafter_bits = 4 }, .{ .fixture_hash = try hexHash(f.fixture_sha256), .config_hash = try binding.agreement(a, m, &preliminary_hash, sharded), .half_pages = loaded.budget.half_pages, .max_forwards = @max(compiled.len, prep.len), .observe = false });
    defer s.close();
    drive(init, args, s, place.rank, f, loaded.budget, m, reference, measure, sharded, prep, compiled) catch |err| {
        s.failed = true;
        cuda.tp_watch.fail("replay runner failed");
        failure(init, cmd, args, err);
        std.debug.print("FAIL glm-replay: {t}\n", .{err});
        return err;
    };
}

fn drive(init: std.process.Init, args: []const [:0]const u8, s: *replay.Session, rank: usize, f: fixture.Fixture, budget: @import("glm_replay_control.zig").Budget, m: metadata.Metadata, reference: ?validation.evidence.Logical, measure: bool, sharded: bool, prep: []exec.Compiled, compiled: []exec.Compiled) !void {
    const a = init.arena.allocator();
    const capacity = f.config.pool.configured_capacity;
    const runtime_hash = std.fmt.bytesToHex(s.runtime_sha256, .lower);
    const identity = try provenance.Provenance.fromSession(a, s);
    if (m.expected_checkpoints) |pins| try pins.require(identity.checkpoints);
    if (measure) {
        try metadata.checkRuntime(a, init.io, m, sharded, &runtime_hash, identity.checkpoints);
        try binding.runtime(reference.?, &runtime_hash, identity.checkpoints);
    }
    try s.agree();
    if (rank != 0) return s.follow();
    if (s.owned.e.capacity != capacity or s.owned.e.c.vocab < f.provenance.tokenizer.vocab_size) return error.LoadedConfiguration;
    const g = try exec.geometry(s.owned.e);
    if (g.layers != 11 or g.heads != 32 or g.latent != 512) return error.LoadedGeometry;
    if (!measure) {
        const evidence = try validation.run(a, s, f, budget, prep, compiled, sharded);
        try binding.retain(a, init.io, args[4], f, m, evidence.observations, evidence.comparisons, sharded, &runtime_hash, identity.checkpoints);
        std.debug.print("PASS glm-replay validation: {d} both-rank forward comparisons; default-server neutrality remains external\n", .{evidence.comparisons});
        return;
    }
    try measureBlock(init, args[4], s, f, budget, m, sharded, prep, compiled, g, &runtime_hash, reference.?.observations);
}

pub fn measureBlock(init: std.process.Init, output: []const u8, s: *replay.Session, f: fixture.Fixture, budget: @import("glm_replay_control.zig").Budget, m: metadata.Metadata, sharded: bool, prep: []exec.Compiled, compiled: []exec.Compiled, g: @import("glm_replay_control.zig").Geometry, runtime_hash: []const u8, reference: []const validation.Observed) !void {
    const a = init.arena.allocator();
    const measured = try @import("glm_replay_measure.zig").run(a, init.io, s, f, budget, prep, compiled, sharded, g, reference);
    defer measured.deinit(a);
    try metadata.checkTelemetry(a, init.io, m);
    const completed = measured.headline;
    const identity = try provenance.Provenance.fromSession(a, s);
    const anchor_hash = try result.anchor(a, init.io, output, f, m, measured.anchor, completed, sharded, runtime_hash, identity);
    try result.write(a, init.io, output, f, budget, m, completed, measured.anchor, &anchor_hash, measured.preparation_ns, sharded, s.owned.e, g, runtime_hash, identity);
    const seconds = @as(f64, @floatFromInt(@max(completed.ranks[0].elapsed_ns, completed.ranks[1].elapsed_ns))) / 1e9;
    const rate = @as(f64, @floatFromInt(completed.ranks[0].rows)) / seconds;
    std.debug.print("PASS glm-replay: both ranks completed {d} forwards; replay throughput {d:.3} forced tokens/s\n", .{ f.forwards.len, rate });
}

pub fn checkModel(a: std.mem.Allocator, io: std.Io, dir: []const u8, f: fixture.Fixture) !void {
    const config_path = try std.fs.path.join(a, &.{ dir, "config.json" });
    const config_raw = try std.Io.Dir.cwd().readFileAlloc(io, config_path, a, .limited(1 << 24));
    const model_config = try glm.config.parse(a, config_raw);
    const expected_model: glm.config.Config = .{};
    if (model_config.vocab < f.provenance.tokenizer.vocab_size or model_config.vocab != expected_model.vocab or model_config.layers != expected_model.layers or model_config.mla_heads != 64 or model_config.kv_lora != 512 or model_config.i_dim != 128) return error.LoadedConfiguration;
}

pub fn settings() !glm.ranks.Settings {
    var problem: []const u8 = "";
    const s = try glm.ranks.Settings.read(&glm.ranks.environment, &problem);
    if (glm.ranks.environment("TF_GLM_DFLASH2_PROF") != null) return error.ReplayIntrusiveProfiler;
    if (s.kv_shard.on and (!s.kv_shard.twins or !s.kv_shard.carry or s.kv_shard.check != glm.kv_shard.check_rounds)) return error.ReplayShardControls;
    if (s.dense_bits != 8 or s.head_bits != 8 or s.layers != 0 or s.numerics.bits() != 0) return error.ReplayConfiguration;
    for ([_][:0]const u8{ "TF_GLM_PROF", "TF_GLM_DFLASH2_PROF", "TF_GLM_DFLASH2_RING_DIGEST" }) |name| {
        if (glm.ranks.environment(name)) |v| if (!std.mem.eql(u8, v, "0")) return error.ReplayIntrusiveProfiler;
    }
    if (!try glm.startup.paging(glm.ranks.environment("TF_KV_PAGES"), 2, true)) return error.ReplayNeedsPages;
    return s;
}

fn hexHash(raw: []const u8) ![32]u8 {
    var out: [32]u8 = undefined;
    _ = try std.fmt.hexToBytes(&out, raw);
    return out;
}

fn failure(init: std.process.Init, cmd: []const u8, args: []const [:0]const u8, err: anyerror) void {
    if (!std.mem.eql(u8, cmd, "glm-replay") or args.len != 6) return;
    const a = init.arena.allocator();
    const path = std.fmt.allocPrint(a, "{s}.failed.json", .{args[4]}) catch return;
    const raw = std.json.Stringify.valueAlloc(a, .{ .schema = "glm-replay-native-attempt-v1", .status = "failed", .mode = args[0], .failure = @errorName(err), .measurement_complete = false }, .{}) catch return;
    result.writeNew(init.io, path, raw) catch {};
}
