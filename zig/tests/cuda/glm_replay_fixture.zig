//! Strict immutable fixture parsing and validation before any CUDA initialization.
const std = @import("std");
const ctl = @import("glm_replay_control.zig");
const hash = @import("glm_replay_hash.zig");
pub const target_revision = "078455ffe6472f9a52fbc1139f58b9db2881b25c";
pub const drafter_revision = "bf582e4eacc1810f76656d1811693ff6c6737d2a";
pub const base_commit = "e52eb42b52b1e50f3f23a5eae8115bbe80c5e7d9";
pub const Round = struct { round_id: u64, forward_id: u64, parts: []const ctl.Part };
pub const Config = struct {
    speed: []const u8,
    tp: u32,
    slots: u32,
    dense_bits: u32,
    head_bits: u32,
    kv: []const u8,
    reset: []const u8,
    selection: []const u8,
    proposer: bool,
    active_streams: []const u32,
    placement: []const u8,
    pool: struct { h: usize, page_tokens: usize, configured_capacity: usize, prompt_twins: usize },
};
pub const Stream = struct {
    stream_id: u32,
    owner: u8,
    source_id: []const u8,
    rendered_prompt: []const u8,
    rendered_tape: []const u8,
    rendered_prompt_sha256: []const u8,
    rendered_tape_sha256: []const u8,
    prompt_encoding: struct { token_ids: []const u32, used_tokens: usize },
    prompt_tokens: []const u32,
    tape_tokens: []const u32,
};
pub const Fixture = struct {
    schema: []const u8,
    fixture_id: []const u8,
    kind: []const u8,
    family: []const u8,
    provenance: struct {
        target_revision: []const u8,
        drafter_revision: []const u8,
        tokenizer: struct {
            kind: []const u8,
            revision: []const u8,
            vocab_size: u32,
            files: []const struct { path: []const u8, sha256: []const u8 },
            template: []const u8,
            template_sha256: []const u8,
        },
        sources: []const struct { source_id: []const u8, attribution: []const u8, license: []const u8, text: []const u8, sha256: []const u8 },
    },
    config: Config,
    streams: [4]Stream,
    preparation: []const Round,
    forwards: []const Round,
    tokens_sha256: []const u8,
    schedule_sha256: []const u8,
    config_sha256: []const u8,
    fixture_sha256: []const u8,

    pub fn streamsFor(f: Fixture) [4]ctl.Stream {
        var out: [4]ctl.Stream = undefined;
        for (f.streams, 0..) |s, i| out[i] = .{ .owner = s.owner, .prompt = s.prompt_tokens, .tape = s.tape_tokens };
        return out;
    }
};

pub const Loaded = struct {
    arena: std.heap.ArenaAllocator,
    fixture: Fixture,
    budget: ctl.Budget,

    pub fn deinit(l: *Loaded) void {
        l.arena.deinit();
    }
};

pub fn load(gpa: std.mem.Allocator, raw: []const u8, allow_synthetic: bool, sharded: bool) !Loaded {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const v = try std.json.parseFromSliceLeaky(hash.Value, a, raw, .{ .allocate = .alloc_always });
    const f = try std.json.parseFromSliceLeaky(Fixture, a, raw, .{ .allocate = .alloc_always });
    try validate(a, v, f, allow_synthetic);
    const streams = f.streamsFor();
    const all = try a.alloc(ctl.Forward, f.preparation.len + f.forwards.len);
    for (f.preparation, 0..) |r, i| all[i] = .{ .forward_id = i, .parts = r.parts };
    for (f.forwards, f.preparation.len..) |r, i| all[i] = .{ .forward_id = i, .parts = r.parts };
    const b = try ctl.budget(streams, all, f.config.pool.configured_capacity, sharded, f.config.pool.h);
    return .{ .arena = arena, .fixture = f, .budget = b };
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn hex(s: []const u8, len: usize) !void {
    if (s.len != len) return error.InvalidDigest;
    for (s) |c| if (!std.ascii.isDigit(c) and (c < 'a' or c > 'f')) return error.InvalidDigest;
}

fn validate(a: std.mem.Allocator, v: hash.Value, f: Fixture, allow_synthetic: bool) !void {
    if (!eq(f.schema, "glm-replay-fixture-v1") or f.fixture_id.len == 0) return error.Schema;
    if (!eq(f.family, "prose") and !eq(f.family, "code") and !eq(f.family, "pi")) return error.Family;
    if (!eq(f.kind, "campaign") and !(allow_synthetic and eq(f.kind, "synthetic"))) return error.SyntheticFixture;
    const c = f.config;
    if (!eq(c.speed, "standard") or c.tp != 2 or c.slots != 4 or c.dense_bits != 8 or c.head_bits != 8 or !eq(c.kv, "fp8") or !eq(c.reset, "fresh_prefill") or !eq(c.selection, "greedy") or c.proposer) return error.Configuration;
    if (c.pool.configured_capacity < 64 or c.pool.configured_capacity % 64 != 0) return error.Capacity;
    if (c.pool.page_tokens != ctl.page_rows or c.pool.prompt_twins != 1) return error.PoolConfig;
    if (!eq(c.placement, "balanced") and !eq(c.placement, "reversed")) return error.Placement;
    if (c.active_streams.len != 1 and c.active_streams.len != 4) return error.ActiveStreams;
    var active: [4]bool = @splat(false);
    for (c.active_streams) |id| {
        if (id >= 4 or active[id]) return error.ActiveStreams;
        active[id] = true;
    }
    const p = f.provenance;
    if (!eq(p.target_revision, target_revision) or !eq(p.drafter_revision, drafter_revision)) return error.Revision;
    const t = p.tokenizer;
    if (t.vocab_size == 0 or t.revision.len == 0) return error.Tokenizer;
    if (!eq(t.kind, "huggingface-local") and !(allow_synthetic and eq(t.kind, "synthetic-byte"))) return error.Tokenizer;
    if (eq(f.kind, "campaign") and (eq(t.kind, "synthetic-byte") or t.files.len == 0)) return error.Tokenizer;
    try hash.matches(t.template_sha256, hash.bytes(t.template));
    for (t.files) |file| {
        try hex(file.sha256, 64);
        if (file.path.len == 0 or std.fs.path.isAbsolute(file.path) or std.mem.indexOf(u8, file.path, "..") != null) return error.TokenizerPath;
    }
    for (p.sources, 0..) |source, i| {
        if (source.source_id.len == 0 or source.attribution.len == 0 or source.license.len == 0) return error.Attribution;
        for (p.sources[0..i]) |prior| if (eq(source.source_id, prior.source_id)) return error.SourceId;
        try hash.matches(source.sha256, hash.bytes(source.text));
    }
    for (f.streams, 0..) |s, i| {
        if (s.stream_id != i or s.owner != (@as(u8, @intCast(i / 2)) ^ @as(u8, @intFromBool(eq(c.placement, "reversed"))))) return error.Owner;
        const found = for (p.sources) |source| {
            if (eq(source.source_id, s.source_id)) break true;
        } else false;
        if (!found) return error.SourceId;
        try hash.matches(s.rendered_prompt_sha256, hash.bytes(s.rendered_prompt));
        try hash.matches(s.rendered_tape_sha256, hash.bytes(s.rendered_tape));
        const enc = s.prompt_encoding;
        if (enc.used_tokens != s.prompt_tokens.len or enc.used_tokens > enc.token_ids.len or s.prompt_tokens.len == 0) return error.PromptEncoding;
        if (!std.mem.eql(u32, enc.token_ids[0..enc.used_tokens], s.prompt_tokens)) return error.PromptEncoding;
        for ([_][]const u32{ enc.token_ids, s.tape_tokens }) |ids| for (ids) |id| if (id >= t.vocab_size) return error.Vocabulary;
        for (f.streams[0..i]) |prior| {
            if (std.mem.eql(u32, prior.prompt_tokens, s.prompt_tokens) or std.mem.eql(u32, prior.tape_tokens, s.tape_tokens)) return error.IdenticalStreams;
        }
    }
    if (f.preparation.len == 0 or f.forwards.len == 0) return error.EmptySchedule;
    var cursor = ctl.Cursor.init(f.streamsFor());
    for ([_][]const Round{ f.preparation, f.forwards }, 0..) |phase, phase_id| {
        cursor.next = 0;
        var prepared: [4]bool = @splat(false);
        for (phase, 0..) |r, i| {
            if (r.round_id != i or r.forward_id != i) return error.ForwardOrder;
            if (phase_id == 1 and r.parts.len != c.active_streams.len) return error.ActiveStreams;
            for (r.parts) |part| {
                if (part.stream_id >= 4 or (phase_id == 1 and !active[part.stream_id])) return error.ActiveStreams;
                prepared[part.stream_id] = true;
            }
            const fw: ctl.Forward = .{ .forward_id = r.forward_id, .parts = r.parts };
            try cursor.submitted(fw, f.streamsFor());
            try cursor.ingested(fw);
        }
        if (phase_id == 0) for (prepared) |done| if (!done) return error.UnpreparedSlot;
    }
    try checkHashes(a, v, f);
}

fn checkHashes(a: std.mem.Allocator, v: hash.Value, f: Fixture) !void {
    var root = try v.object.clone(a);
    _ = root.swapRemove("fixture_sha256");
    try hash.matches(f.fixture_sha256, try hash.digest(a, .{ .object = root }));
    try hash.matches(f.config_sha256, try hash.digest(a, v.object.get("config").?));
    var schedule: std.json.ObjectMap = .empty;
    try schedule.put(a, "preparation", v.object.get("preparation").?);
    try schedule.put(a, "forwards", v.object.get("forwards").?);
    try hash.matches(f.schedule_sha256, try hash.digest(a, .{ .object = schedule }));
    var token_streams: std.array_list.Managed(hash.Value) = .init(a);
    for (v.object.get("streams").?.array.items) |s| {
        var obj: std.json.ObjectMap = .empty;
        for ([_][]const u8{ "stream_id", "prompt_tokens", "tape_tokens" }) |key| try obj.put(a, key, s.object.get(key).?);
        try token_streams.append(.{ .object = obj });
    }
    try hash.matches(f.tokens_sha256, try hash.digest(a, .{ .array = token_streams }));
}

test "replay fixture refuses missing fields and duplicate keys before device access" {
    try std.testing.expectError(error.MissingField, load(std.testing.allocator, "{}", true, false));
    try std.testing.expectError(error.DuplicateField, load(std.testing.allocator, "{\"schema\":0,\"schema\":1}", true, false));
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 2) return error.ReplayFixtureArguments;
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(512 << 20));
    var f = try load(init.gpa, raw, true, false);
    defer f.deinit();
    std.debug.print("PASS fixture {s} ({s}): {d} forwards, four prepared slots, H={d}\n", .{ f.fixture.fixture_sha256, f.fixture.kind, f.fixture.forwards.len, f.budget.half_pages });
}
