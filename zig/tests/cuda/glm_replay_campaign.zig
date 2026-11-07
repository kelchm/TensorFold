//! Ordered, driver-free campaign preflight; one descriptor denotes one measured block.
const std = @import("std");
const fixture = @import("glm_replay_fixture.zig");
const meta = @import("glm_replay_metadata.zig");
pub const max_cases = 4096;
pub const Descriptor = struct { fixture: []const u8, metadata: []const u8, output: []const u8 };
pub const Manifest = struct { schema: []const u8, cases: []const Descriptor };
pub const Case = struct { f: fixture.Fixture, budget: @import("glm_replay_control.zig").Budget, m: meta.Metadata, output: []const u8 };
const Cached = struct { path: []const u8, loaded: fixture.Loaded };
pub const Campaign = struct {
    arena: std.heap.ArenaAllocator,
    fixtures: []Cached,
    cases: []Case,
    ordered_hash: [32]u8,

    pub fn deinit(c: *Campaign) void {
        for (c.fixtures) |*f| f.loaded.deinit();
        c.arena.deinit();
    }
};

pub fn checkPages(value: ?[]const u8) !void {
    for ([_][]const u8{ "on", "1", "all" }) |on| if (eq(value orelse "", on)) return;
    return error.ReplayNeedsPages;
}

pub fn parse(a: std.mem.Allocator, raw: []const u8) !Manifest {
    const m = try std.json.parseFromSliceLeaky(Manifest, a, raw, .{ .allocate = .alloc_always });
    if (!eq(m.schema, "glm-replay-native-campaign-v1") or m.cases.len == 0 or m.cases.len > max_cases) return error.CampaignSchema;
    for (m.cases) |c| {
        for ([_][]const u8{ c.fixture, c.metadata, c.output }) |path| {
            if (path.len == 0 or std.mem.indexOfScalar(u8, path, 0) != null) return error.CampaignPath;
        }
    }
    return m;
}

pub fn load(gpa: std.mem.Allocator, io: std.Io, path: []const u8, model: []const u8, sharded: bool) !Campaign {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(4 << 20));
    const manifest = try parse(a, raw);
    var cache: std.ArrayList(Cached) = .empty;
    errdefer for (cache.items) |*f| f.loaded.deinit();
    const cases = try a.alloc(Case, manifest.cases.len);
    for (manifest.cases, cases) |d, *c| {
        const canonical = try std.Io.Dir.cwd().realPathFileAlloc(io, d.fixture, a);
        const index = for (cache.items, 0..) |prior, i| {
            if (eq(canonical, prior.path)) break i;
        } else blk: {
            const contents = try std.Io.Dir.cwd().readFileAlloc(io, canonical, a, .limited(512 << 20));
            var loaded = try fixture.load(gpa, contents, false, sharded);
            errdefer loaded.deinit();
            try meta.checkTokenizer(a, io, model, loaded.fixture);
            try cache.append(a, .{ .path = canonical, .loaded = loaded });
            break :blk cache.items.len - 1;
        };
        const f = cache.items[index].loaded.fixture;
        const m = try meta.read(a, io, d.metadata, f, true, sharded);
        try meta.checkTelemetry(a, io, m);
        c.* = .{ .f = f, .budget = cache.items[index].loaded.budget, .m = m, .output = try outputPath(a, io, d.output) };
    }
    const ordered = try check(a, cases);
    return .{ .arena = arena, .fixtures = cache.items, .cases = cases, .ordered_hash = ordered };
}

pub fn outputPath(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const name = std.fs.path.basename(path);
    if (name.len == 0 or eq(name, ".") or eq(name, "..")) return error.CampaignPath;
    const parent = try std.Io.Dir.cwd().realPathFileAlloc(io, std.fs.path.dirname(path) orelse ".", a);
    defer a.free(parent);
    const output = try std.fs.path.join(a, &.{ parent, name });
    errdefer a.free(output);
    if (std.Io.Dir.cwd().statFile(io, output, .{ .follow_symlinks = false })) |_| {
        return error.CampaignOutputExists;
    } else |err| if (err != error.FileNotFound) return err;
    return output;
}

pub fn check(a: std.mem.Allocator, cases: []const Case) ![32]u8 {
    if (cases.len == 0 or cases.len > max_cases) return error.CampaignSchema;
    const first = cases[0];
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    sha.update("glm-replay-native-campaign-v1\x00");
    for (cases, 0..) |c, i| {
        const m = c.m;
        if (m.run_id.len == 0 or m.pair_id.len == 0 or m.start_id.len == 0 or m.cell_id.len == 0 or m.block_id.len == 0 or m.start_order > 1) return error.CampaignIdentity;
        if (!eq(m.fixture_sha256, c.f.fixture_sha256) or !eq(m.config_sha256, c.f.config_sha256)) return error.MetadataFixture;
        if (!eq(m.run_id, first.m.run_id) or !eq(m.pair_id, first.m.pair_id) or !eq(m.start_id, first.m.start_id) or m.start_order != first.m.start_order) return error.CampaignMixedStart;
        inline for (@typeInfo(meta.Source).@"struct".field_names) |field| {
            if (!eq(@field(m.source, field), @field(first.m.source, field))) return error.CampaignMixedSource;
        }
        if (!std.meta.eql(c.f.config.pool, first.f.config.pool)) return error.CampaignMixedPool;
        for (cases[0..i]) |prior| {
            if (eq(c.output, prior.output)) return error.CampaignDuplicateOutput;
            if (eq(m.cell_id, prior.m.cell_id)) {
                if (eq(m.block_id, prior.m.block_id)) return error.CampaignDuplicateBlock;
                if (!eq(c.f.fixture_sha256, prior.f.fixture_sha256)) return error.CampaignChangedCell;
            }
        }
        const encoded = try std.json.Stringify.valueAlloc(a, .{
            .fixture = c.f.fixture_sha256,
            .config = c.f.config_sha256,
            .tokens = c.f.tokens_sha256,
            .schedule = c.f.schedule_sha256,
            .run = m.run_id,
            .pair = m.pair_id,
            .start = m.start_id,
            .order = m.start_order,
            .cell = m.cell_id,
            .block = m.block_id,
            .source = m.source,
            .pool = c.f.config.pool,
            .telemetry = m.telemetry.sha256,
            .correctness = if (m.correctness) |v| v.sha256 else null,
            .sharding_equality = if (m.sharding_equality) |v| v.sha256 else null,
            .timing_neutrality = if (m.timing_neutrality) |v| v.sha256 else null,
            .default_server_neutrality = if (m.default_server_neutrality) |v| v.sha256 else null,
        }, .{});
        defer a.free(encoded);
        sha.update(encoded);
        sha.update("\x00");
    }
    return sha.finalResult();
}

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}
