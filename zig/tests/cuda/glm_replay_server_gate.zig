//! Offline serving evidence has server identities, independent of replay binary/runtime identities.
const std = @import("std");
const hash = @import("glm_replay_hash.zig");
const fixture = @import("glm_replay_fixture.zig");
pub const Artifact = struct { path: []const u8, sha256: []const u8 };
pub const Gate = struct {
    schema: []const u8,
    status: []const u8,
    execution: []const u8,
    engine: []const u8,
    harness_commit: []const u8,
    fatbin_sha256: []const u8,
    original_server_sha256: []const u8,
    instrumented_server_sha256: []const u8,
    completed_ranks: [2]u32,
    token_evidence: Artifact,
    performance_evidence: Artifact,
    contract_sha256: []const u8,
};
pub const Contract = struct {
    schema: []const u8,
    engine: []const u8,
    engine_commit: []const u8,
    image_digest: []const u8,
    harness_commit: []const u8,
    fatbin_sha256: []const u8,
    original_server_sha256: []const u8,
    instrumented_server_sha256: []const u8,
    policy_sha256: []const u8,
    declared_unix_ns: u64,
};
pub const TokenCase = struct {
    case_id: []const u8,
    request_sha256: []const u8,
    config_sha256: []const u8,
    original_tokens: []const u32,
    instrumented_tokens: []const u32,
};
pub const Tokens = struct {
    schema: []const u8,
    execution: []const u8,
    engine: []const u8,
    harness_commit: []const u8,
    fatbin_sha256: []const u8,
    original_server_sha256: []const u8,
    instrumented_server_sha256: []const u8,
    completed_ranks: [2]u32,
    started_unix_ns: u64,
    ended_unix_ns: u64,
    cases: []const TokenCase,
};
pub const Sample = struct {
    case_id: []const u8,
    pair_id: []const u8,
    start_order: u8,
    request_sha256: []const u8,
    config_sha256: []const u8,
    original_elapsed_ns: u64,
    instrumented_elapsed_ns: u64,
};
pub const Policy = struct {
    schema: []const u8,
    predeclared_policy_sha256: []const u8,
    minimum_pairs: u64,
    max_slowdown_ppm: u64,
};
pub const Performance = struct {
    schema: []const u8,
    execution: []const u8,
    engine: []const u8,
    harness_commit: []const u8,
    fatbin_sha256: []const u8,
    original_server_sha256: []const u8,
    instrumented_server_sha256: []const u8,
    completed_ranks: [2]u32,
    started_unix_ns: u64,
    ended_unix_ns: u64,
    policy: Policy,
    samples: []const Sample,
};

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn read(a: std.mem.Allocator, io: std.Io, ref: anytype, contract: anytype, source: anytype) !void {
    const craw = try std.Io.Dir.cwd().readFileAlloc(io, contract.path, a, .limited(1 << 20));
    try hash.matches(contract.sha256, hash.bytes(craw));
    const pins = try std.json.parseFromSliceLeaky(Contract, a, craw, .{});
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, ref.path, a, .limited(1 << 20));
    try hash.matches(ref.sha256, hash.bytes(raw));
    const g = try std.json.parseFromSliceLeaky(Gate, a, raw, .{});
    const traw = try std.Io.Dir.cwd().readFileAlloc(io, g.token_evidence.path, a, .limited(64 << 20));
    const praw = try std.Io.Dir.cwd().readFileAlloc(io, g.performance_evidence.path, a, .limited(64 << 20));
    try hash.matches(g.token_evidence.sha256, hash.bytes(traw));
    try hash.matches(g.performance_evidence.sha256, hash.bytes(praw));
    const t = try std.json.parseFromSliceLeaky(Tokens, a, traw, .{});
    const p = try std.json.parseFromSliceLeaky(Performance, a, praw, .{});
    try check(a, g, t, p, pins, contract.sha256, source);
}

fn identity(value: anytype, source: anytype, c: Contract) !void {
    if (!eq(value.engine, "native") or !eq(value.execution, "device") or
        !std.mem.eql(u32, &value.completed_ranks, &.{ 0, 1 })) return error.ServerDeviceUnrun;
    try fixture.hex(value.harness_commit, 40);
    inline for (.{ "fatbin_sha256", "original_server_sha256", "instrumented_server_sha256" }) |key|
        try fixture.hex(@field(value, key), 64);
    if (!eq(value.harness_commit, source.harness_commit) or !eq(value.fatbin_sha256, source.fatbin_sha256))
        return error.ServerSourceIdentity;
    if (!eq(value.original_server_sha256, c.original_server_sha256) or
        !eq(value.instrumented_server_sha256, c.instrumented_server_sha256)) return error.ServerContractIdentity;
}

fn evidence(value: anytype, g: Gate, source: anytype, c: Contract) !void {
    try identity(value, source, c);
    if (value.started_unix_ns <= c.declared_unix_ns or value.ended_unix_ns < value.started_unix_ns)
        return error.ServerCaptureBeforeContract;
    inline for (.{ "engine", "harness_commit", "fatbin_sha256", "original_server_sha256", "instrumented_server_sha256" }) |key|
        if (!eq(@field(value, key), @field(g, key))) return error.ServerEvidenceIdentity;
}

pub fn check(a: std.mem.Allocator, g: Gate, t: Tokens, p: Performance, c: Contract, contract_hash: []const u8, source: anytype) !void {
    if (!eq(c.schema, "glm-replay-server-contract-v1") or !eq(c.engine, "native") or
        !eq(c.engine_commit, fixture.base_commit) or !eq(c.image_digest, source.image_digest) or
        !eq(c.harness_commit, source.harness_commit) or !eq(c.fatbin_sha256, source.fatbin_sha256) or
        c.declared_unix_ns == 0) return error.ServerContractIdentity;
    inline for (.{ "original_server_sha256", "instrumented_server_sha256", "policy_sha256" }) |key|
        try fixture.hex(@field(c, key), 64);
    if (eq(c.original_server_sha256, c.instrumented_server_sha256)) return error.ServerIdenticalBinaries;
    if (!eq(g.contract_sha256, contract_hash)) return error.ServerContractIdentity;
    if (!eq(g.schema, "glm-replay-server-neutrality-v2") or !eq(g.status, "pass")) return error.ServerNeutralityUnrun;
    try identity(g, source, c);
    if (!eq(t.schema, "glm-replay-server-tokens-v2") or t.cases.len == 0 or t.cases.len > 1024)
        return error.ServerTokenCases;
    if (!eq(p.schema, "glm-replay-server-performance-v2") or p.samples.len == 0 or p.samples.len > 100000)
        return error.ServerPerformanceCases;
    try evidence(t, g, source, c);
    try evidence(p, g, source, c);
    const policy = p.policy;
    if (!eq(policy.schema, "glm-replay-server-policy-v1") or policy.minimum_pairs < 3 or policy.max_slowdown_ppm >= 1000000)
        return error.ServerPolicy;
    const encoded = try std.json.Stringify.valueAlloc(a, .{ .schema = policy.schema, .minimum_pairs = policy.minimum_pairs, .max_slowdown_ppm = policy.max_slowdown_ppm }, .{});
    const v = try std.json.parseFromSliceLeaky(hash.Value, a, encoded, .{});
    try hash.matches(policy.predeclared_policy_sha256, try hash.digest(a, v));
    if (!eq(policy.predeclared_policy_sha256, c.policy_sha256)) return error.ServerPolicyIdentity;
    for (t.cases, 0..) |tc, i| {
        if (tc.case_id.len == 0 or tc.original_tokens.len == 0 or !std.mem.eql(u32, tc.original_tokens, tc.instrumented_tokens))
            return error.ServerTokensDiffer;
        try fixture.hex(tc.request_sha256, 64);
        try fixture.hex(tc.config_sha256, 64);
        for (tc.original_tokens) |id| if (id >= 154880) return error.ServerToken;
        for (t.cases[0..i]) |prior| if (eq(tc.case_id, prior.case_id)) return error.ServerDuplicateCase;
        var count: u64 = 0;
        var first: u64 = 0;
        var old: u128 = 0;
        var new: u128 = 0;
        for (p.samples) |s| if (eq(s.case_id, tc.case_id)) {
            if (!eq(s.request_sha256, tc.request_sha256) or !eq(s.config_sha256, tc.config_sha256)) return error.ServerChangedWork;
            count += 1;
            first += @intFromBool(s.start_order == 0);
            old += s.original_elapsed_ns;
            new += s.instrumented_elapsed_ns;
        };
        if (count < policy.minimum_pairs or first > count - first + 1 or count - first > first + 1)
            return error.ServerPairCoverage;
        if (new * 1000000 > old * (1000000 + policy.max_slowdown_ppm)) return error.ServerPerformanceRegression;
    }
    for (p.samples, 0..) |s, i| {
        if (s.pair_id.len == 0 or s.start_order > 1 or s.original_elapsed_ns == 0 or s.instrumented_elapsed_ns == 0)
            return error.ServerSample;
        const known = for (t.cases) |tc| {
            if (eq(tc.case_id, s.case_id)) break true;
        } else false;
        if (!known) return error.ServerUnknownCase;
        for (p.samples[0..i]) |prior| if (eq(s.case_id, prior.case_id) and eq(s.pair_id, prior.pair_id))
            return error.ServerDuplicatePair;
    }
}

test "server neutrality uses actual server identities and refuses changed tokens, work, policy and coverage" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const source = .{ .harness_commit = "1111111111111111111111111111111111111111", .fatbin_sha256 = "2222222222222222222222222222222222222222222222222222222222222222", .binary_sha256 = "9999999999999999999999999999999999999999999999999999999999999999", .image_digest = "sha256:0000000000000000000000000000000000000000000000000000000000000000" };
    var g: Gate = .{ .contract_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", .schema = "glm-replay-server-neutrality-v2", .status = "pass", .execution = "device", .engine = "native", .harness_commit = source.harness_commit, .fatbin_sha256 = source.fatbin_sha256, .original_server_sha256 = "3333333333333333333333333333333333333333333333333333333333333333", .instrumented_server_sha256 = "4444444444444444444444444444444444444444444444444444444444444444", .completed_ranks = .{ 0, 1 }, .token_evidence = .{ .path = "host-control", .sha256 = "5555555555555555555555555555555555555555555555555555555555555555" }, .performance_evidence = .{ .path = "host-control", .sha256 = "6666666666666666666666666666666666666666666666666666666666666666" } };
    var cases = [_]TokenCase{.{ .case_id = "host-control", .request_sha256 = "7777777777777777777777777777777777777777777777777777777777777777", .config_sha256 = "8888888888888888888888888888888888888888888888888888888888888888", .original_tokens = &.{ 1, 2 }, .instrumented_tokens = &.{ 1, 2 } }};
    var t: Tokens = undefined;
    var p: Performance = undefined;
    inline for (.{ "execution", "engine", "harness_commit", "fatbin_sha256", "original_server_sha256", "instrumented_server_sha256", "completed_ranks" }) |key| {
        @field(t, key) = @field(g, key);
        @field(p, key) = @field(g, key);
    }
    t.schema = "glm-replay-server-tokens-v2";
    t.cases = &cases;
    t.started_unix_ns = 2;
    t.ended_unix_ns = 3;
    p.started_unix_ns = 4;
    p.ended_unix_ns = 5;
    p.schema = "glm-replay-server-performance-v2";
    const policy_hash = hash.bytes("{\"max_slowdown_ppm\":50000,\"minimum_pairs\":3,\"schema\":\"glm-replay-server-policy-v1\"}");
    p.policy = .{ .schema = "glm-replay-server-policy-v1", .predeclared_policy_sha256 = &policy_hash, .minimum_pairs = 3, .max_slowdown_ppm = 50000 };
    var samples: [3]Sample = undefined;
    for (&samples, [_][]const u8{ "0", "1", "2" }, 0..) |*s, pair, i|
        s.* = .{ .case_id = "host-control", .pair_id = pair, .start_order = @intCast(i % 2), .request_sha256 = cases[0].request_sha256, .config_sha256 = cases[0].config_sha256, .original_elapsed_ns = 1000, .instrumented_elapsed_ns = 1020 };
    p.samples = &samples;
    var pins: Contract = .{ .schema = "glm-replay-server-contract-v1", .engine = "native", .engine_commit = fixture.base_commit, .image_digest = source.image_digest, .harness_commit = source.harness_commit, .fatbin_sha256 = source.fatbin_sha256, .original_server_sha256 = g.original_server_sha256, .instrumented_server_sha256 = g.instrumented_server_sha256, .policy_sha256 = &policy_hash, .declared_unix_ns = 1 };
    try check(a, g, t, p, pins, g.contract_sha256, source);
    cases[0].instrumented_tokens = &.{ 1, 3 };
    try std.testing.expectError(error.ServerTokensDiffer, check(a, g, t, p, pins, g.contract_sha256, source));
    cases[0].instrumented_tokens = cases[0].original_tokens;
    g.status = "unrun";
    try std.testing.expectError(error.ServerNeutralityUnrun, check(a, g, t, p, pins, g.contract_sha256, source));
    g.status = "pass";
    samples[0].request_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
    try std.testing.expectError(error.ServerChangedWork, check(a, g, t, p, pins, g.contract_sha256, source));
    samples[0].request_sha256 = cases[0].request_sha256;
    for (&samples) |*s| s.instrumented_elapsed_ns = 1060;
    try std.testing.expectError(error.ServerPerformanceRegression, check(a, g, t, p, pins, g.contract_sha256, source));
    for (&samples) |*s| s.instrumented_elapsed_ns = 1020;
    p.samples = samples[0..2];
    try std.testing.expectError(error.ServerPairCoverage, check(a, g, t, p, pins, g.contract_sha256, source));
    p.samples = &samples;
    pins.instrumented_server_sha256 = pins.original_server_sha256;
    try std.testing.expectError(error.ServerIdenticalBinaries, check(a, g, t, p, pins, g.contract_sha256, source));
    pins.instrumented_server_sha256 = g.instrumented_server_sha256;
    t.started_unix_ns = 1;
    try std.testing.expectError(error.ServerCaptureBeforeContract, check(a, g, t, p, pins, g.contract_sha256, source));
    t.started_unix_ns = 2;
    pins.policy_sha256 = g.contract_sha256;
    try std.testing.expectError(error.ServerPolicyIdentity, check(a, g, t, p, pins, g.contract_sha256, source));
    pins.policy_sha256 = &policy_hash;
    p.policy.max_slowdown_ppm = 60000;
    try std.testing.expectError(error.HashMismatch, check(a, g, t, p, pins, g.contract_sha256, source));
}
