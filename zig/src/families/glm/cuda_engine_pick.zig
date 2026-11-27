//! The existing rank-gathered greedy pick, kept separate from engine lifetime and prompt diagnostics.
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const chance = @import("cuda_mtp_chance.zig");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;
const pair_bytes = 16;
/// The most rows one batched pick takes (a decode window's, a shared forward's).
pub const pick_rows = @max(fw.window_rows, engine.shared_rows);

/// Whether a shared verify round's draws can run as one batched greedy top-1 gather across all parts.
pub fn canBatchGreedy(e: *const Engine, parts: []const engine.Part, rows: usize) bool {
    if (parts.len <= 1 or rows == 0 or rows > pick_rows or e.structured or e.local.len > 0) return false;
    for (parts) |p| {
        if (p.slot >= e.seqs.items.len) return false;
        const q = e.seqs.items[p.slot] orelse return false;
        if (q.sampling != null) return false;
    }
    return true;
}

/// Draws every part of a shared verify round: one batched greedy gather when all parts are greedy, else per-part draw.
pub fn drawShared(e: *Engine, b: *const fw.Buffers, parts: []const engine.Part, segs: []const fw.Seg, rows: usize) !void {
    if (canBatchGreedy(e, parts, rows)) {
        for (parts, segs) |p, g| {
            try e.bind(p.slot);
            e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        }
        var all: [pick_rows]u32 = undefined;
        try greedyRows(e, b.logits, rows, all[0..rows]);
        for (parts, segs) |p, g| @memcpy(p.out[0..g.rows], all[g.row0..][0..g.rows]);
        return;
    }
    for (parts, segs) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.bits, p.out);
    }
}
pub fn greedy(e: *Engine, logits: u64) !u32 {
    return (try pick(e, logits, false)).token;
}

/// `rows` rows' greedy picks, each as `greedy` gives it: their top-1s, then one gather and one read for them all.
pub fn greedyRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    if (rows == 1) {
        out[0] = try greedy(e, logits);
        return;
    }
    const world = e.share.world;
    const bytes = rows * pair_bytes;
    const mine = e.pairs.ptr + e.share.rank * bytes;
    if (e.m.glue.top1_rows) |f| {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(mine);
        try cuda_pdl.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.stream, &a);
    } else for (0..rows) |r| {
        var a: cuda.Args = .{};
        a.add(logits + r * e.share.vocab.len() * 2);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(mine + r * pair_bytes);
        try cuda_pdl.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    }
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world * pick_rows][4]u32 = undefined;
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. world * bytes]);
    var row: [ranks.max_world][4]u32 = undefined;
    for (0..rows) |r| {
        for (0..world) |w| row[w] = all[w * rows + r];
        out[r] = chance.pick(row[0..world], false).token;
    }
}

/// greedy with the pick's chance from every rank's softmax sum (tf_glm_top1_sums), for --mtp-confidence's drafts.
pub fn greedyChance(e: *Engine, logits: u64) !chance.Pick {
    return pick(e, logits, true);
}

fn pick(e: *Engine, logits: u64, sums: bool) !chance.Pick {
    const world = e.share.world;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(e.share.vocab.len())));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(e.pairs.ptr + e.share.rank * pair_bytes);
    try cuda_pdl.launch(if (sums) e.m.glue.top1_sums else e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    const mine = e.pairs.ptr + e.share.rank * pair_bytes;
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. world * pair_bytes]);
    return chance.pick(all[0..world], sums);
}

test "verifyShared drawShared batches greedy top-1 into one gather and falls back when sampled or structured" {
    const t = std.testing;
    const Seq = @import("cuda_engine_types.zig").Seq;
    const vocab_len: usize = 64;
    const base_logits: u64 = 0x100000;
    const h_top1: cuda.abi.Function = @ptrFromInt(1);
    const h_top1_rows: cuda.abi.Function = @ptrFromInt(2);
    const Mock = struct {
        var r0: [14][4]u32 = undefined;
        var r1: [14][4]u32 = undefined;
        var pairs_dev: [2 * 32][4]u32 = undefined;
        var row_for_slot: [32]usize = undefined;
        var top1_calls: usize = 0;
        var top1_rows_calls: usize = 0;
        var last_grid_x: c_uint = 0;
        var last_block_x: c_uint = 0;
        var last_logits: u64 = 0;
        var last_n: i64 = 0;
        var last_lo: u32 = 0;
        var last_out: u64 = 0;
        var gathers: usize = 0;
        var gather_bytes: usize = 0;
        var syncs: usize = 0;
        var downloads: usize = 0;
        var sampled_gathers: usize = 0;
        fn reset() void {
            top1_calls = 0;
            top1_rows_calls = 0;
            last_grid_x = 0;
            last_block_x = 0;
            last_logits = 0;
            last_n = 0;
            last_lo = 0;
            last_out = 0;
            gathers = 0;
            gather_bytes = 0;
            syncs = 0;
            downloads = 0;
            sampled_gathers = 0;
        }
        fn launch(func: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const logits = @as(*const u64, @ptrCast(@alignCast(params.?[0].?))).*;
            const n = @as(*const i64, @ptrCast(@alignCast(params.?[1].?))).*;
            const lo = @as(*const u32, @ptrCast(@alignCast(params.?[2].?))).*;
            const dst = @as(*const u64, @ptrCast(@alignCast(params.?[3].?))).*;
            last_grid_x = gx;
            last_block_x = bx;
            last_logits = logits;
            last_n = n;
            last_lo = lo;
            last_out = dst;
            const row0 = @as(usize, @intCast((logits - base_logits) / (vocab_len * 2)));
            const idx0 = @as(usize, @intCast((dst - @intFromPtr(&pairs_dev)) / pair_bytes));
            if (func == h_top1_rows) {
                for (0..@intCast(gx)) |r| {
                    pairs_dev[idx0 + r] = r0[row0 + r];
                    row_for_slot[idx0 + r] = row0 + r;
                }
                top1_rows_calls += 1;
                return 0;
            }
            pairs_dev[idx0] = r0[row0];
            row_for_slot[idx0] = row0;
            top1_calls += 1;
            return 0;
        }
        fn allGather(_: u64, _: u64, bytes: usize, _: cuda.nccl.DataType, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            const n = bytes / pair_bytes;
            for (0..n) |i| pairs_dev[n + i] = r1[row_for_slot[i]];
            gathers += 1;
            gather_bytes = bytes;
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            syncs += 1;
            return 0;
        }
        fn dtoh(dst: ?*anyopaque, src: u64, bytes: usize) callconv(.c) cuda.abi.Result {
            const out: [*]u8 = @ptrCast(dst.?);
            const inp: [*]const u8 = @ptrFromInt(src);
            @memcpy(out[0..bytes], inp[0..bytes]);
            downloads += 1;
            return 0;
        }
        fn groupOk() callconv(.c) cuda.nccl.Result {
            return 0;
        }
        fn bcast(_: u64, _: u64, _: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            sampled_gathers += 1;
            return 0;
        }
        fn copy2d(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.error_not_ready;
        }
        fn device(dev: *cuda.abi.Device) callconv(.c) cuda.abi.Result {
            dev.* = 0;
            return 0;
        }
        fn attribute(val: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            val.* = 0;
            return 0;
        }
    };
    cuda.download_stage.forget();
    defer cuda.download_stage.forget();
    // Build 14 rows across 4 parts: mix rank-0 wins, rank-1 wins, and exact ties favoring lower token id.
    var expected: [14]u32 = undefined;
    for (0..14) |r| {
        const v0: f32 = if (r % 3 == 0) 5.0 else if (r % 3 == 1) 2.0 else 4.0;
        const v1: f32 = if (r % 3 == 0) 3.0 else if (r % 3 == 1) 6.0 else 4.0;
        const id0: u32 = @intCast(10 + r);
        const id1: u32 = @intCast(100 + r);
        Mock.r0[r] = .{ @bitCast(v0), id0, 0, 0 };
        Mock.r1[r] = .{ @bitCast(v1), id1, 0, 0 };
        expected[r] = if (r % 3 == 1) id1 else id0;
    }
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Mock.launch;
    d.api.cuLaunchKernelEx = cuda_pdl.forwardEx(Mock.launch);
    d.api.cuStreamSynchronize = Mock.sync;
    d.api.cuMemcpyDtoH_v2 = Mock.dtoh;
    d.api.cuMemcpy2DAsync_v2 = Mock.copy2d;
    d.api.cuCtxGetDevice = Mock.device;
    d.api.cuDeviceGetAttribute = Mock.attribute;
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclAllGather = Mock.allGather;
    lib.api.ncclGroupStart = Mock.groupOk;
    lib.api.ncclGroupEnd = Mock.groupOk;
    lib.api.ncclBroadcast = Mock.bcast;
    var seqs = [4]Seq{ .{ .st = undefined }, .{ .st = undefined }, .{ .st = undefined }, .{ .st = undefined } };
    for (&seqs, 0..) |*q, i| {
        q.st.pos = 10 + i;
        q.sampling = null;
        q.rows_at = 0;
    }
    var spans = [_]@import("cuda_shard.zig").Span{ .{ .lo = 0, .hi = vocab_len }, .{ .lo = vocab_len, .hi = 2 * vocab_len } };
    var e: Engine = undefined;
    e.gpa = t.allocator;
    e.d = &d;
    e.lib = &lib;
    e.comm = null;
    e.group = null;
    e.collective = undefined;
    e.c.hidden = 4096;
    e.share.rank = 0;
    e.share.world = 2;
    e.share.vocab = spans[0];
    e.spans = &spans;
    e.local = &.{};
    e.images = null;
    e.structured = false;
    e.profiling_prompt = null;
    e.stream = .{ .d = &d, .handle = null };
    e.m.glue.top1 = .{ .d = &d, .handle = h_top1 };
    e.m.glue.top1_rows = .{ .d = &d, .handle = h_top1_rows };
    e.pairs = .{ .d = &d, .ptr = @intFromPtr(&Mock.pairs_dev), .len = @sizeOf(@TypeOf(Mock.pairs_dev)), .borrowed = true };
    e.smp.d = &d;
    e.smp.vocab = 2 * vocab_len;
    e.smp.stage = .{ .d = &d, .ptr = 0x200000, .len = 4096, .borrowed = true };
    e.smp.full = .{ .d = &d, .ptr = 0x300000, .len = 4096, .borrowed = true };
    e.seqs = .empty;
    defer e.seqs.deinit(t.allocator);
    try e.seqs.appendSlice(t.allocator, &.{ &seqs[0], &seqs[1], &seqs[2], &seqs[3] });
    e.cur = &seqs[0];
    e.slot = 0;
    var b: fw.Buffers = undefined;
    b.logits = base_logits;
    b.fnormed = 0x400000;
    const widths = [_]usize{ 3, 5, 2, 4 };
    var out_batched: [14]u32 = @splat(0);
    var out_single: [14]u32 = @splat(0);
    var parts: [4]engine.Part = undefined;
    var segs: [4]fw.Seg = undefined;
    var off: usize = 0;
    for (widths, 0..) |w, i| {
        parts[i] = .{ .slot = @intCast(i), .ids = &.{}, .out = out_batched[off..][0..w] };
        segs[i] = .{ .st = &seqs[i].st, .row0 = off, .rows = w };
        off += w;
    }
    // 1) All-greedy 4-stream shared draw with top1_rows: 1 launch (grid.x=14), 1 gather, 1 sync, 1 download.
    try t.expect(canBatchGreedy(&e, &parts, 14));
    Mock.reset();
    try drawShared(&e, &b, &parts, &segs, 14);
    try t.expectEqual(@as(usize, 1), Mock.top1_rows_calls);
    try t.expectEqual(@as(usize, 0), Mock.top1_calls);
    try t.expectEqual(@as(c_uint, 14), Mock.last_grid_x);
    try t.expectEqual(@as(c_uint, 1024), Mock.last_block_x);
    try t.expectEqual(base_logits, Mock.last_logits);
    try t.expectEqual(@as(i64, @intCast(vocab_len)), Mock.last_n);
    try t.expectEqual(@as(u32, 0), Mock.last_lo);
    try t.expectEqual(e.pairs.ptr, Mock.last_out);
    try t.expectEqual(@as(usize, 1), Mock.gathers);
    try t.expectEqual(@as(usize, 14 * pair_bytes), Mock.gather_bytes);
    try t.expectEqual(@as(usize, 1), Mock.syncs);
    try t.expectEqual(@as(usize, 1), Mock.downloads);
    try t.expectEqualSlices(u32, &expected, &out_batched);
    try t.expectEqual(@as(u32, 3), e.slot);
    try t.expectEqual(&seqs[3], e.cur);
    for (segs, 0..) |g, i| try t.expectEqual(b.fnormed + g.row0 * e.c.hidden * 2, seqs[i].rows_at);
    // Legacy fatbin fallback (top1_rows == null): 14 per-row top1 launches, 1 gather, identical outputs.
    e.m.glue.top1_rows = null;
    Mock.reset();
    @memset(&out_batched, 0);
    try drawShared(&e, &b, &parts, &segs, 14);
    try t.expectEqual(@as(usize, 0), Mock.top1_rows_calls);
    try t.expectEqual(@as(usize, 14), Mock.top1_calls);
    try t.expectEqual(@as(usize, 1), Mock.gathers);
    try t.expectEqualSlices(u32, &expected, &out_batched);
    // Single-row greedyRows still dispatches 1-row top1 even when top1_rows is loaded.
    e.m.glue.top1_rows = .{ .d = &d, .handle = h_top1_rows };
    Mock.reset();
    var one_tok: [1]u32 = .{0};
    try greedyRows(&e, base_logits, 1, &one_tok);
    try t.expectEqual(@as(usize, 0), Mock.top1_rows_calls);
    try t.expectEqual(@as(usize, 1), Mock.top1_calls);
    try t.expectEqual(expected[0], one_tok[0]);
    // Single-part loop produces the exact same tokens with 4 separate gathers/syncs/downloads.
    Mock.reset();
    for (0..4) |i| {
        var one_part = parts[i];
        one_part.out = out_single[segs[i].row0..][0..segs[i].rows];
        try t.expect(!canBatchGreedy(&e, &.{one_part}, segs[i].rows));
        try drawShared(&e, &b, &.{one_part}, &.{segs[i]}, segs[i].rows);
    }
    try t.expectEqual(@as(usize, 4), Mock.gathers);
    try t.expectEqual(@as(usize, 4), Mock.syncs);
    try t.expectEqual(@as(usize, 4), Mock.downloads);
    try t.expectEqualSlices(u32, &out_single, &out_batched);
    // 2) Mixed greedy + sampled falls back to per-part draw: part 0 draws 3 rows alone before part 1 enters sampler.
    seqs[1].sampling = .{ .seed = 1, .temperature = 0.8 };
    try t.expect(!canBatchGreedy(&e, &parts, 14));
    Mock.reset();
    @memset(&out_batched, 0);
    try t.expectError(error.NotReady, drawShared(&e, &b, &parts, &segs, 14));
    try t.expectEqual(@as(usize, 1), Mock.gathers);
    try t.expectEqual(@as(usize, 3 * pair_bytes), Mock.gather_bytes);
    try t.expectEqualSlices(u32, expected[0..3], out_batched[0..3]);
    try t.expect(Mock.sampled_gathers > 0);
    // 3) Structured mode falls back to drawWhole immediately without touching greedyRows.
    seqs[1].sampling = null;
    e.structured = true;
    try t.expect(!canBatchGreedy(&e, &parts, 14));
    Mock.reset();
    try t.expectError(error.NotReady, drawShared(&e, &b, &parts, &segs, 14));
    try t.expectEqual(@as(usize, 0), Mock.gathers);
    try t.expect(Mock.sampled_gathers > 0);
}
