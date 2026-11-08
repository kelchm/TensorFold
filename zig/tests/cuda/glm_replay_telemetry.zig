//! Post-block requests and measured samples; no telemetry IO occurs inside a replay block.
const std = @import("std");
const hash = @import("glm_replay_hash.zig");
const fixture = @import("glm_replay_fixture.zig");
pub const Ref = struct { path: []const u8 };
pub const Checkpoints = @import("glm_replay_provenance.zig").Checkpoints;
pub const Ids = struct { run_id: []const u8, pair_id: []const u8, start_id: []const u8, start_order: u8, cell_id: []const u8, block_id: []const u8 };
pub const Bindings = struct {
    fixture_sha256: []const u8,
    config_sha256: []const u8,
    schedule_sha256: []const u8,
    tokens_sha256: []const u8,
    arm: []const u8,
    source_sha256: []const u8,
    source_tree_sha256: []const u8,
    runtime_sha256: []const u8,
    checkpoints_sha256: []const u8,
};
pub const RankWindow = struct { rank: u8, realtime_start_ns: u64, realtime_end_ns: u64 };
pub const Window = struct {
    schema: []const u8 = "glm-replay-block-window-v1",
    bindings: Bindings,
    ids: Ids,
    completed_ranks: [2]u32 = .{ 0, 1 },
    ranks: [2]RankWindow,
    session_nonce: []const u8,
    campaign_sha256: []const u8,
    checkpoints: Checkpoints,
};
pub const Sample = struct {
    realtime_ns: u64,
    sm_clock_mhz: u64,
    memory_clock_mhz: u64,
    temperature_millicelsius: u64,
    power_milliwatts: u64,
    used_memory_bytes: u64,
};
pub const Rank = struct {
    rank: u8,
    device_uuid: []const u8,
    realtime_start_ns: u64,
    realtime_end_ns: u64,
    max_sample_gap_ns: u64,
    samples: []const Sample,
};
pub const Telemetry = struct {
    schema: []const u8,
    status: []const u8,
    window_sha256: []const u8,
    bindings: Bindings,
    ids: Ids,
    completed_ranks: [2]u32,
    ranks: [2]Rank,
    session_nonce: []const u8,
    campaign_sha256: []const u8,
    checkpoints: Checkpoints,
};

fn eq(a: []const u8, b: []const u8) bool {
    return std.mem.eql(u8, a, b);
}

pub fn digest(a: std.mem.Allocator, value: anytype) ![64]u8 {
    const raw = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(raw);
    const parsed = try std.json.parseFromSlice(hash.Value, a, raw, .{});
    defer parsed.deinit();
    return hash.digest(a, parsed.value);
}

pub fn checkPath(path: []const u8) !void {
    if (path.len == 0 or path[0] == '/' or std.mem.indexOfAny(u8, path, "\\\x00") != null) return error.TelemetryPath;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| if (part.len == 0 or eq(part, ".") or eq(part, "..")) return error.TelemetryPath;
}

pub fn safePath(io: std.Io, path: []const u8) !void {
    try checkPath(path);
    var end: usize = 0;
    while (end < path.len) {
        end = if (std.mem.indexOfScalarPos(u8, path, end, '/')) |i| i else path.len;
        if (std.Io.Dir.cwd().statFile(io, path[0..end], .{ .follow_symlinks = false })) |st| {
            if (st.kind == .sym_link) return error.TelemetrySymlink;
        } else |err| if (err != error.FileNotFound) return err;
        end += 1;
    }
}

pub fn preflight(a: std.mem.Allocator, io: std.Io, ref: Ref) !void {
    const request = try std.fmt.allocPrint(a, "{s}.window.json", .{ref.path});
    defer a.free(request);
    for ([_][]const u8{ ref.path, request }) |path| {
        try safePath(io, path);
        if (std.Io.Dir.cwd().statFile(io, path, .{ .follow_symlinks = false })) |_| {
            return error.TelemetryOutputExists;
        } else |err| if (err != error.FileNotFound) return err;
    }
}

pub fn checkWindow(w: Window) !void {
    if (!eq(w.schema, "glm-replay-block-window-v1") or !std.mem.eql(u32, &w.completed_ranks, &.{ 0, 1 })) return error.TelemetryWindow;
    try fixture.hex(w.session_nonce, 64);
    try fixture.hex(w.campaign_sha256, 64);
    inline for (@typeInfo(Bindings).@"struct".field_names) |key| {
        if (comptime !std.mem.eql(u8, key, "arm")) try fixture.hex(@field(w.bindings, key), 64);
    }
    if (!eq(w.bindings.arm, "A") and !eq(w.bindings.arm, "B")) return error.TelemetryArm;
    if (!eq(w.checkpoints.scheme, "tensorfold-full-identity-v1")) return error.TelemetryCheckpoint;
    for (w.checkpoints.target, w.checkpoints.drafter) |t, d| {
        try fixture.hex(t, 64);
        try fixture.hex(d, 64);
    }
    if (w.ids.start_order > 1) return error.TelemetryIds;
    inline for (.{ "run_id", "pair_id", "start_id", "cell_id", "block_id" }) |key|
        if (@field(w.ids, key).len == 0) return error.TelemetryIds;
    for (w.ranks, 0..) |r, i| if (r.rank != i or r.realtime_start_ns == 0 or r.realtime_end_ns <= r.realtime_start_ns)
        return error.TelemetryWindow;
}

pub fn check(a: std.mem.Allocator, t: Telemetry, w: Window, devices: [2][]const u8) !void {
    try checkWindow(w);
    if (!eq(t.schema, "glm-replay-telemetry-v1") or !eq(t.status, "measured") or
        !std.mem.eql(u32, &t.completed_ranks, &w.completed_ranks)) return error.TelemetryUnmeasured;
    try hash.matches(t.window_sha256, try digest(a, w));
    inline for (.{ "bindings", "ids", "checkpoints" }) |key| {
        if (!std.mem.eql(u8, &try digest(a, @field(t, key)), &try digest(a, @field(w, key)))) return error.TelemetryBinding;
    }
    if (!eq(t.session_nonce, w.session_nonce) or !eq(t.campaign_sha256, w.campaign_sha256)) return error.TelemetryBinding;
    if (eq(devices[0], devices[1]) or devices[0].len == 0 or devices[1].len == 0) return error.TelemetryDevice;
    for (t.ranks, w.ranks, devices, 0..) |r, want, device, rank| {
        if (r.rank != rank or !eq(r.device_uuid, device) or r.realtime_start_ns != want.realtime_start_ns or
            r.realtime_end_ns != want.realtime_end_ns) return error.TelemetryRank;
        if (r.max_sample_gap_ns == 0 or r.max_sample_gap_ns > 5000000000 or r.samples.len < 2 or r.samples.len > 100000)
            return error.TelemetrySamples;
        var inside = false;
        var previous: u64 = 0;
        for (r.samples, 0..) |s, i| {
            if (s.realtime_ns == 0 or s.sm_clock_mhz == 0 or s.memory_clock_mhz == 0 or
                s.temperature_millicelsius > 150000 or s.power_milliwatts == 0 or s.used_memory_bytes == 0)
                return error.TelemetrySample;
            if (i != 0 and (s.realtime_ns <= previous or s.realtime_ns - previous > r.max_sample_gap_ns)) return error.TelemetryGap;
            inside = inside or (s.realtime_ns >= want.realtime_start_ns and s.realtime_ns <= want.realtime_end_ns);
            previous = s.realtime_ns;
        }
        if (!inside or r.samples[0].realtime_ns > want.realtime_start_ns or r.samples[r.samples.len - 1].realtime_ns < want.realtime_end_ns)
            return error.TelemetryCoverage;
    }
}

pub fn receive(a: std.mem.Allocator, io: std.Io, ref: Ref, window: Window, devices: [2][]const u8, timeout_ms: u32) ![64]u8 {
    if (timeout_ms > 60000) return error.TelemetryWait;
    try checkWindow(window);
    const request = try std.fmt.allocPrint(a, "{s}.window.json", .{ref.path});
    try safePath(io, request);
    if (std.fs.path.dirname(request)) |parent| try std.Io.Dir.cwd().createDirPath(io, parent);
    const raw = try std.json.Stringify.valueAlloc(a, window, .{});
    const file = try std.Io.Dir.cwd().createFile(io, request, .{ .exclusive = true });
    defer file.close(io);
    try file.writeStreamingAll(io, raw);
    const start = std.Io.Clock.awake.now(io).toNanoseconds();
    while (true) {
        try safePath(io, ref.path);
        const incoming = std.Io.Dir.cwd().readFileAlloc(io, ref.path, a, .limited(64 << 20)) catch |err| {
            if (err != error.FileNotFound) return err;
            const elapsed = std.Io.Clock.awake.now(io).toNanoseconds() - start;
            if (elapsed >= @as(i96, timeout_ms) * 1000000) return error.TelemetryUnavailable;
            try std.Io.sleep(io, .fromMilliseconds(50), .awake);
            continue;
        };
        defer a.free(incoming);
        const parsed = try std.json.parseFromSlice(Telemetry, a, incoming, .{});
        defer parsed.deinit();
        try check(a, parsed.value, window, devices);
        return hash.bytes(incoming);
    }
}

pub fn sourceManifest(a: std.mem.Allocator, io: std.Io, ref: anytype, source: anytype) !void {
    const Entry = struct { path: []const u8, bytes: u64, sha256: []const u8 };
    const Manifest = struct { schema: []const u8, base_commit: []const u8, harness_commit: []const u8, files: []const Entry };
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, ref.path, a, .limited(8 << 20));
    try hash.matches(ref.sha256, hash.bytes(raw));
    const parsed = try std.json.parseFromSlice(Manifest, a, raw, .{});
    defer parsed.deinit();
    const m = parsed.value;
    if (!eq(m.schema, "glm-replay-native-source-v1") or !eq(m.base_commit, fixture.base_commit) or
        !eq(m.harness_commit, source.harness_commit) or m.files.len == 0 or m.files.len > 10000) return error.TelemetrySource;
    var runner = false;
    var session = false;
    for (m.files, 0..) |entry, i| {
        try checkPath(entry.path);
        try fixture.hex(entry.sha256, 64);
        runner = runner or eq(entry.path, "zig/tests/cuda/glm_replay.zig");
        session = session or eq(entry.path, "zig/src/families/glm/cuda_replay.zig");
        if (i != 0 and std.mem.order(u8, m.files[i - 1].path, entry.path) != .lt) return error.TelemetrySource;
    }
    if (!runner or !session) return error.TelemetrySource;
}
