//! GLM profiler lifetime operation used by unwind and close.
const std = @import("std");
const prof_mod = @import("cuda_prof.zig");
const Engine = @import("cuda_engine.zig").Engine;

pub fn initProfs(e: *Engine, layers: usize) !void {
    return initProfsFor(e, layers, try prof_mod.Prof.wanted());
}

pub fn initProfsFor(e: *Engine, layers: usize, profiled: bool) !void {
    if (!profiled) return;
    for (&e.profs) |*p| p.* = try prof_mod.Prof.init(e.d, e.io);
    e.shared_prof = try prof_mod.Prof.init(e.d, e.io);
    for (&e.prompt_profs, 0..) |*p, bucket| {
        p.* = try prof_mod.Prof.init(e.d, e.io);
        p.*.?.every = 4;
        const started = std.Io.Clock.awake.now(e.io).toNanoseconds();
        try p.*.?.initPrompt(layers, e.pieces.count);
        const create_ms = prof_mod.Prof.elapsedMs(e.io, started);
        const pairs = prof_mod.Prof.promptCapacity(layers, e.pieces.count);
        const bucket_rows = ([_]usize{ 64, 256, 1024, 2048 })[bucket];
        std.log.info("glm prof startup rank={d} bucket_rows={d} layers={d} pieces={d} pairs={d} events={d} create_ms={d:.4}", .{ e.share.rank, bucket_rows, layers, e.pieces.count, pairs, 2 * pairs, create_ms });
    }
    if (e.heat_gate.bands) |bands| {
        std.log.info("glm prof startup rank={d} heat=on high_c={d:.1} low_c={d:.1} root={s}", .{ e.share.rank, bands.high_c, bands.low_c, e.heat_gate.root });
    } else {
        std.log.info("glm prof startup rank={d} heat=off", .{e.share.rank});
    }
}

pub fn freeProfs(e: *Engine) void {
    for (&e.profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
    if (e.shared_prof) |v| v.deinit();
    e.shared_prof = null;
    for (&e.prompt_profs) |*p| {
        if (p.*) |v| v.deinit();
        p.* = null;
    }
}

pub fn heatRoot() []const u8 {
    const s = (@import("sys").environment.Process{}).get("TF_HEAT_ROOT") orelse return @import("heat").default_root;
    return if (s.len == 0) @import("heat").default_root else s;
}

pub fn verifyLabel(rows: usize) []const u8 {
    const l = [_][]const u8{ "verify 1 row", "verify 2 rows", "verify 3 rows", "verify 4 rows", "verify 5 rows", "verify 6 rows", "verify 7 rows", "verify 8 rows" };
    return l[@min(rows, l.len) - 1];
}

test "shared_prof: zero events when TF_GLM_PROF is off, balanced lifecycle when on" {
    const cuda = @import("cuda");
    const Fake = struct {
        var live: usize = 0;
        fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
            live += 1;
            out.* = @ptrFromInt(live);
            return 0;
        }
        fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
            live -= 1;
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuEventCreate = Fake.create;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    var e: Engine = undefined;
    e.d = &d;
    e.io = std.testing.io;
    e.profs = @splat(null);
    e.shared_prof = null;
    e.prompt_profs = @splat(null);
    e.pieces.count = 1;
    e.share.rank = 0;
    e.heat_gate = .{ .bands = null, .root = "" };
    Fake.live = 0;
    try initProfsFor(&e, 0, false);
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
    try std.testing.expect(e.shared_prof == null);
    for (e.profs) |p| try std.testing.expect(p == null);
    try initProfsFor(&e, 0, true);
    try std.testing.expect(e.shared_prof != null);
    try std.testing.expect(Fake.live > 0);
    freeProfs(&e);
    try std.testing.expect(e.shared_prof == null);
    try std.testing.expectEqual(@as(usize, 0), Fake.live);
}
