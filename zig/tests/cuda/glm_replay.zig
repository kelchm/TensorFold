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
const clock = @import("check.zig").now;

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
    try metadata.checkTokenizer(a, init.io, args[2], f);
    const prep = try exec.compile(a, f, f.preparation);
    const compiled = try exec.compile(a, f, f.forwards);
    const config_path = try std.fs.path.join(a, &.{ args[2], "config.json" });
    const config_raw = try std.Io.Dir.cwd().readFileAlloc(init.io, config_path, a, .limited(1 << 24));
    const model_config = try glm.config.parse(a, config_raw);
    const expected_model: glm.config.Config = .{};
    if (model_config.vocab < f.provenance.tokenizer.vocab_size or model_config.vocab != expected_model.vocab or model_config.layers != expected_model.layers or model_config.mla_heads != 64 or model_config.kv_lora != 512 or model_config.i_dim != 128) return error.LoadedConfiguration;
    const place = try glm.native.place();
    if (place.world != 2) return error.ReplayNeedsTP2;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const capacity = f.config.pool.configured_capacity;
    const s = try replay.Session.open(init.gpa, init.io, &ctx, args[2], .{ .context = capacity - 16, .kv8 = true, .drafts = true, .drafter = args[3], .drafter_bits = 4 }, .{ .fixture_hash = try hexHash(f.fixture_sha256), .config_hash = agreement(f.config_sha256, &preliminary_hash), .half_pages = loaded.budget.half_pages, .max_forwards = @max(compiled.len, prep.len), .observe = true });
    defer s.close();
    drive(init, args, s, place.rank, f, loaded.budget, m, measure, sharded, prep, compiled) catch |err| {
        s.failed = true;
        cuda.tp_watch.fail("replay runner failed");
        failure(init, cmd, args, err);
        std.debug.print("FAIL glm-replay: {t}\n", .{err});
        return err;
    };
}

fn drive(init: std.process.Init, args: []const [:0]const u8, s: *replay.Session, rank: usize, f: fixture.Fixture, budget: @import("glm_replay_control.zig").Budget, m: metadata.Metadata, measure: bool, sharded: bool, prep: []exec.Compiled, compiled: []exec.Compiled) !void {
    const a = init.arena.allocator();
    const capacity = f.config.pool.configured_capacity;
    const runtime_hash = std.fmt.bytesToHex(s.runtime_sha256, .lower);
    if (measure) try metadata.checkRuntime(a, init.io, m, sharded, &runtime_hash);
    if (rank != 0) return s.follow();
    if (s.owned.e.capacity != capacity or s.owned.e.c.vocab < f.provenance.tokenizer.vocab_size) return error.LoadedConfiguration;
    const g = try exec.geometry(s.owned.e);
    if (g.layers != 11 or g.heads != 32 or g.latent != 512) return error.LoadedGeometry;
    if (!measure) {
        const evidence = try validation.run(a, s, f, budget, prep, compiled, sharded);
        const correct_path = try std.fmt.allocPrint(a, "{s}.correctness.json", .{args[4]});
        const timing_path = try std.fmt.allocPrint(a, "{s}.timing.json", .{args[4]});
        try result.gate(a, init.io, correct_path, f, m, "correctness", evidence.comparisons, sharded, &runtime_hash);
        try result.gate(a, init.io, timing_path, f, m, "timing_neutrality", evidence.comparisons, sharded, &runtime_hash);
        const logical_path = try std.fmt.allocPrint(a, "{s}.logical.json", .{args[4]});
        const logical_raw = try std.json.Stringify.valueAlloc(a, .{ .schema = "glm-replay-native-logical-v1", .fixture_sha256 = f.fixture_sha256, .config_sha256 = f.config_sha256, .binary_sha256 = m.source.binary_sha256, .runtime_sha256 = &runtime_hash, .kv_shard = sharded, .forwards = f.forwards, .owners = [4]u8{ f.streams[0].owner, f.streams[1].owner, f.streams[2].owner, f.streams[3].owner }, .observations = evidence.observations }, .{});
        try result.writeNew(init.io, logical_path, logical_raw);
        std.debug.print("PASS glm-replay validation: {d} both-rank forward comparisons; default-server neutrality remains external\n", .{evidence.comparisons});
        return;
    }
    try s.enableTiming(true);
    try exec.prepare(s, f, budget, prep);
    _ = try exec.block(s, compiled);
    const before = clock(init.io);
    try exec.prepare(s, f, budget, prep);
    const preparation_ns: u64 = @intCast(clock(init.io) - before);
    const completed = try exec.block(s, compiled);
    try exec.checkCompletion(completed, f.forwards, sharded, g);
    try metadata.checkTelemetry(a, init.io, m);
    try result.write(a, init.io, args[4], f, budget, m, completed, preparation_ns, sharded, s.owned.e, g, &runtime_hash);
    const seconds = @as(f64, @floatFromInt(completed.elapsed_ns)) / 1e9;
    const rate = @as(f64, @floatFromInt(completed.ranks[0].rows)) / seconds;
    std.debug.print("PASS glm-replay: both ranks completed {d} forwards; replay throughput {d:.3} forced tokens/s\n", .{ f.forwards.len, rate });
}

fn settings() !glm.ranks.Settings {
    var problem: []const u8 = "";
    const s = try glm.ranks.Settings.read(&glm.ranks.environment, &problem);
    if (glm.ranks.environment("TF_GLM_DFLASH2_PROF") != null) return error.ReplayIntrusiveProfiler;
    if (s.kv_shard.on and (!s.kv_shard.twins or !s.kv_shard.carry or s.kv_shard.check != glm.kv_shard.check_rounds)) return error.ReplayShardControls;
    if (s.dense_bits != 8 or s.head_bits != 8 or s.layers != 0 or s.numerics.bits() != 0) return error.ReplayConfiguration;
    for ([_][:0]const u8{ "TF_GLM_PROF", "TF_GLM_DFLASH2_PROF", "TF_GLM_DFLASH2_RING_DIGEST" }) |name| {
        if (glm.ranks.environment(name)) |v| if (!std.mem.eql(u8, v, "0")) return error.ReplayIntrusiveProfiler;
    }
    if (glm.ranks.environment("TF_KV_PAGES")) |v| {
        if (!std.mem.eql(u8, v, "on") and !std.mem.eql(u8, v, "auto")) return error.ReplayNeedsPages;
    }
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

fn agreement(config_hash: []const u8, settings_hash: []const u8) [32]u8 {
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    sha.update(config_hash);
    sha.update(settings_hash);
    return sha.finalResult();
}
