//! TF_GLM_GREEDY_ROWS: rank agreement, the shared window's decision and real draws on a recording TP=2 driver.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const pick = @import("cuda_engine_pick.zig");
const ranks = @import("cuda_ranks.zig");
const types = @import("cuda_engine_types.zig");
const fw = @import("cuda_forward.zig");
const t = std.testing;

test "GLM greedy rows: TF_GLM_GREEDY_ROWS is on by default, 0 or 1, refused otherwise, and every rank must agree" {
    const R = struct {
        var v: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_GREEDY_ROWS")) v else null;
        }
    };
    var problem: []const u8 = "";
    for ([_]?[]const u8{ null, "", "1" }) |v| {
        R.v = v;
        try t.expect((try ranks.Settings.read(R.get, &problem)).greedy_rows);
    }
    R.v = "0";
    const off = try ranks.Settings.read(R.get, &problem);
    try t.expect(!off.greedy_rows);
    for ([_][]const u8{ "on", "2", "true" }) |v| {
        R.v = v;
        try t.expectError(error.BadSetting, ranks.Settings.read(R.get, &problem));
        try t.expectEqualStrings("TF_GLM_GREEDY_ROWS: 0 or 1", problem);
    }
    const h: [32]u8 = @splat(0);
    var all = [_]ranks.Fingerprint{ .of(2, 0, .{}, 8192, true, h, h, h), .of(2, 1, .{}, 8192, true, h, h, h) };
    try t.expect(ranks.mismatch(&all) == null);
    all[1] = .of(2, 1, off, 8192, true, h, h, h);
    try t.expectEqual(ranks.Mismatch{ .field = .greedy_rows, .rank = 1 }, ranks.mismatch(&all).?);
}

test "GLM greedy rows: a shared window draws in one gather only with the switch on, no grammar and every part greedy" {
    var seqs: [3]types.Seq = undefined;
    for (&seqs) |*q| q.sampling = null;
    var e: engine.Engine = undefined;
    e.settings = .{};
    e.structured = false;
    e.seqs = .empty;
    defer e.seqs.deinit(t.allocator);
    try e.seqs.appendSlice(t.allocator, &.{ &seqs[0], null, &seqs[1], &seqs[2] });
    var out: [3]u32 = undefined;
    const parts = [_]engine.Part{ .{ .slot = 0, .ids = &.{1}, .out = out[0..1] }, .{ .slot = 2, .ids = &.{2}, .out = out[1..2] }, .{ .slot = 3, .ids = &.{3}, .out = out[2..3] } };
    try t.expect(pick.oneDraw(&e, &parts));
    e.structured = true; // one window under a grammar: the whole step draws whole rows
    try t.expect(!pick.oneDraw(&e, &parts));
    e.structured = false;
    seqs[1].sampling = .{ .seed = 3, .temperature = 0.7, .top_k = 0, .top_p = 1.0 };
    try t.expect(!pick.oneDraw(&e, &parts));
    try t.expect(pick.oneDraw(&e, &.{ parts[0], parts[2] }));
    seqs[1].sampling = null;
    try t.expect(!pick.oneDraw(&e, &.{.{ .slot = 1, .ids = &.{1}, .out = out[0..1] }})); // a slot never bound
    e.settings.greedy_rows = false;
    try t.expect(!pick.oneDraw(&e, &parts));
}

/// TP=2 on the host: both top-1 kernels as the host scan on each rank's slice; the gather puts rank 1's words after.
const Fake = struct {
    const V = 5; // each rank's vocabulary slice
    var base: u64 = 0; // rank 0's logits
    var peer_base: u64 = 0; // rank 1's, the same rows
    var peer: [pick.pick_rows][4]u32 = undefined;
    var launches: [2]usize = .{ 0, 0 }; // tf_glm_top1, tf_glm_top1_rows
    var grid: u32 = 0;
    var gathers: usize = 0;
    var gathered: usize = 0; // the last gather's bytes a rank
    var syncs: usize = 0;
    var reads: usize = 0;

    fn reset() void {
        launches = .{ 0, 0 };
        gathers = 0;
        syncs = 0;
        reads = 0;
    }

    fn bytes(p: u64, n: usize) []u8 {
        return @as([*]u8, @ptrFromInt(p))[0..n];
    }

    /// glm_top1's words: the first bf16 above every earlier one from -inf (a NaN never wins), its bits and lo + index.
    fn scan(logits: u64, n: usize, lo: u32) [4]u32 {
        const row: [*]const u16 = @ptrFromInt(logits);
        var top = -std.math.inf(f32);
        var best: ?usize = null;
        for (0..n) |i| {
            const v: f32 = @bitCast(@as(u32, row[i]) << 16);
            if (v > top) {
                top = v;
                best = i;
            }
        }
        return .{ @bitCast(top), lo + @as(u32, @intCast(best orelse 0)), 0, 0 };
    }

    fn arg(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }

    fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        launches[@intFromPtr(f) - 1] += 1;
        grid = gx;
        const logits = arg(u64, args, 0);
        const n: usize = @intCast(arg(i64, args, 1));
        const lo = arg(u32, args, 2);
        const out = arg(u64, args, 3);
        for (0..gx) |b| {
            const row = logits + b * n * 2;
            const w = scan(row, n, lo);
            @memcpy(bytes(out + b * 16, 8), std.mem.sliceAsBytes(w[0..2])); // words 0 and 1 only, as the kernel
            peer[b] = scan(peer_base + (row - base), n, lo + V);
        }
        return 0;
    }

    fn gather(mine: u64, all: u64, n: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
        gathers += 1;
        gathered = n;
        if (mine != all) return 1; // rank 0's words sit first already
        @memcpy(bytes(all + n, n), std.mem.sliceAsBytes(peer[0 .. n / 16]));
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        syncs += 1;
        return 0;
    }
    fn dtoh(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
        reads += 1;
        @memcpy(@as([*]u8, @ptrCast(dst.?))[0..n], bytes(src, n));
        return 0;
    }
    fn contextDevice(out: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0;
        return 0;
    }
    fn deviceAttribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 0; // a discrete card: cuda.memory.download copies straight to the host
        return 0;
    }
};

fn bf(x: f32) u16 {
    return @intCast(@as(u32, @bitCast(x)) >> 16);
}

/// Rank 1's max, a tie across ranks, a tie in a slice, a NaN, none above -inf (id 0), and a plain row.
const rows_n = 6;
const nan = std.math.nan(f32);
const inf = std.math.inf(f32);
const mine_rows = [rows_n][Fake.V]f32{ .{ 1, 2, 3, 0, -1 }, .{ 4, 0, 4, 1, 2 }, .{ 0, 7, 1, 7, 2 }, .{ nan, 1, 2, 9, 3 }, @splat(-inf), .{ 5, 6, 2, 1, 0 } };
const peer_rows = [rows_n][Fake.V]f32{ .{ 0, 8, 1, 1, 2 }, .{ 1, 4, 4, 0, 3 }, .{ 7, 1, 0, 7, 1 }, .{ 1, 2, nan, 3, 4 }, @splat(-inf), .{ 1, 2, 3, 4, 5 } };

/// The host's scan of the whole row, rank 0's slice then rank 1's.
fn want(r: usize) u32 {
    var top = -inf;
    var best: u32 = 0;
    for ([_][Fake.V]f32{ mine_rows[r], peer_rows[r] }, 0..) |slice, q| for (slice, 0..) |v, i| if (v > top) {
        top = v;
        best = @intCast(q * Fake.V + i);
    };
    return best;
}

const World = struct {
    d: cuda.Driver,
    lib: cuda.nccl.Library,
    e: engine.Engine,
    seqs: [3]types.Seq,
    pairs: [2 * pick.pick_rows][4]u32,
    mine: [rows_n][Fake.V]u16,
    other: [rows_n][Fake.V]u16,

    /// Rank 0 of two, on the fake driver: greedy streams, no grammar, the switch as given.
    fn init(w: *World, greedy_rows: bool) void {
        w.d = undefined;
        w.d.api.cuLaunchKernel = Fake.launch;
        w.d.api.cuStreamSynchronize = Fake.wait;
        w.d.api.cuMemcpyDtoH_v2 = Fake.dtoh;
        w.d.api.cuCtxGetDevice = Fake.contextDevice;
        w.d.api.cuDeviceGetAttribute = Fake.deviceAttribute;
        w.lib = undefined;
        w.lib.api.ncclAllGather = Fake.gather;
        for (&w.mine, &w.other, mine_rows, peer_rows) |*a, *b, x, y| for (a, b, x, y) |*p, *q, u, v| {
            p.* = bf(u);
            q.* = bf(v);
        };
        Fake.base = @intFromPtr(&w.mine);
        Fake.peer_base = @intFromPtr(&w.other);
        w.pairs = @splat(@splat(0));
        for (&w.seqs) |*q| q.* = .{ .st = undefined };
        const e = &w.e;
        e.* = undefined;
        e.gpa = t.allocator;
        e.d = &w.d;
        e.lib = &w.lib;
        e.comm = null;
        e.group = null;
        e.images = null;
        e.profiling_prompt = null;
        e.stream = .{ .d = &w.d, .handle = null };
        e.m.glue.top1 = .{ .d = &w.d, .handle = @ptrFromInt(1) };
        e.m.glue.top1_rows = .{ .d = &w.d, .handle = @ptrFromInt(2) };
        e.share.rank = 0;
        e.share.world = 2;
        e.share.vocab = .{ .lo = 0, .hi = Fake.V };
        e.settings = .{ .greedy_rows = greedy_rows };
        e.structured = false;
        e.c.hidden = 4;
        e.pairs = .{ .d = &w.d, .ptr = @intFromPtr(&w.pairs), .len = @sizeOf(@TypeOf(w.pairs)) };
        e.seqs = .empty;
        e.cur = &w.seqs[0];
        Fake.reset();
    }
};

test "GLM greedy rows: a verify window's rows take one launch, gather, sync and read, and the tokens of a row each" {
    var drawn: [2][rows_n]u32 = undefined;
    for ([_]bool{ true, false }, &drawn) |on, *out| {
        var w: World = undefined;
        w.init(on);
        try w.e.draw(Fake.base, rows_n, 1, 0, out);
        for (out, 0..) |tok, r| try t.expectEqual(want(r), tok);
        try t.expectEqual([2]usize{ if (on) 0 else rows_n, @intFromBool(on) }, Fake.launches);
        try t.expectEqual(@as(usize, if (on) 1 else rows_n), Fake.gathers);
        try t.expectEqual(@as(usize, if (on) rows_n * 16 else 16), Fake.gathered);
        try t.expectEqual(@as(usize, if (on) 1 else rows_n), Fake.syncs);
        try t.expectEqual(@as(usize, if (on) 1 else rows_n), Fake.reads);
        if (on) try t.expectEqual(@as(u32, rows_n), Fake.grid);
    }
    try t.expectEqualSlices(u32, &drawn[1], &drawn[0]);
    var w: World = undefined;
    w.init(true);
    var one: [1]u32 = undefined;
    try w.e.draw(Fake.base + Fake.V * 2, 1, 1, 0, &one); // a single row (a prompt's first token, an MTP draft): tf_glm_top1
    try t.expectEqual(want(1), one[0]);
    try t.expectEqual([2]usize{ 1, 0 }, Fake.launches);
    try t.expectEqual(@as(usize, 16), Fake.gathered);
}

test "GLM greedy rows: a shared window's parts take one gather, each part's tokens and rows_at as its own draw leaves" {
    var w: World = undefined;
    w.init(true);
    const e = &w.e;
    defer e.seqs.deinit(t.allocator);
    try e.seqs.appendSlice(t.allocator, &.{ &w.seqs[0], &w.seqs[1], &w.seqs[2] });
    var b: fw.Buffers = undefined;
    b.logits = Fake.base;
    b.fnormed = 1 << 20;
    var outs: [rows_n]u32 = @splat(99);
    const ids: [rows_n]u32 = @splat(0);
    const parts = [_]engine.Part{ .{ .slot = 2, .ids = ids[0..2], .out = outs[0..2] }, .{ .slot = 0, .ids = ids[2..5], .out = outs[2..5] }, .{ .slot = 1, .ids = ids[5..6], .out = outs[5..6] } };
    const segs = [_]fw.Seg{ .{ .st = undefined, .row0 = 0, .rows = 2 }, .{ .st = undefined, .row0 = 2, .rows = 3 }, .{ .st = undefined, .row0 = 5, .rows = 1 } };
    try t.expect(pick.oneDraw(e, &parts));
    try pick.drawShared(e, &b, &parts, &segs);
    for (outs, 0..) |tok, r| try t.expectEqual(want(r), tok);
    try t.expectEqual([2]usize{ 0, 1 }, Fake.launches);
    try t.expectEqual(@as(usize, 1), Fake.gathers);
    try t.expectEqual(@as(usize, rows_n * 16), Fake.gathered);
    try t.expectEqual(@as(usize, 1), Fake.syncs);
    for (parts, segs) |p, g| try t.expectEqual(b.fnormed + g.row0 * e.c.hidden * 2, w.seqs[p.slot].rows_at);
    try t.expectEqual(@as(u32, 1), e.slot); // the last part bound, as the per-part loop leaves it
}

test "GLM greedy rows: one gather's words in rank order pick each row as a gather a row does" {
    var prng = std.Random.DefaultPrng.init(0x6772);
    const rng = prng.random();
    const world = 3;
    const rows = 7;
    var all: [world * rows][4]u32 = undefined;
    for (&all, 0..) |*w, i| w.* = .{ @bitCast(@as(f32, @floatFromInt(rng.uintLessThan(u32, 4)))), @intCast(i), 0, 0 };
    for (0..rows) |r| {
        var row: [world][4]u32 = undefined;
        for (&row, 0..) |*w, q| w.* = all[q * rows + r];
        try t.expectEqual(@import("cuda_mtp_chance.zig").pick(&row, false).token, pick.rowPick(&all, world, rows, r));
    }
}
