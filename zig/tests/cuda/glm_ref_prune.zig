//! TF_GLM_DENSE_PRUNE on the GPU: kept dense rows equal the whole chunk's bytes, the rest unwritten, sparse past them.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Rig = qr.RigT;
const dsa = glm.dsa_block;
const latent = glm.latent;
const sparse = glm.sparse;

const lw = 512;
const heads = 32;
const most = 4096;

/// (host_pos, rows) chunks at the dense limit: first chunks, a window across it, one from inside the dense part.
const cases = [_][2]usize{ .{ 0, 2048 }, .{ 0, 2050 }, .{ 0, 2051 }, .{ 0, 2052 }, .{ 0, 4096 }, .{ 2048, 8 }, .{ 1000, 2048 } };

pub fn run(r: *Rig, set: glm.glue.Set, ops: cuda.Module) !void {
    var l: dsa.Layer = undefined; // denseAttention reads the heads, the latent width and the limit
    l.heads = heads;
    l.lw = lw;
    l.dense_limit = sparse.width;
    const k: dsa.Kernels = .{ .set = set, .ops = ops };
    const q = try r.dev(std.mem.sliceAsBytes(try r.bfs(most * heads * lw, 1.0)));
    const c: dsa.Caches = .{ .lc = try r.dev(std.mem.sliceAsBytes(try r.bfs(most * lw, 1.0))), .ik = 0, .ig = 0, .pk = 0, .pk_rows = 0 };
    const nch_most = latent.chunksFor(most);
    const part_rows = dsa.prompt_att_rows;
    var s = std.mem.zeroes(dsa.Scratch);
    s.qa = q;
    s.partials = .{ .po = try r.zeros(nch_most * part_rows * heads * lw * 4), .pm = try r.zeros(nch_most * part_rows * heads * 4), .pl = try r.zeros(nch_most * part_rows * heads * 4) };
    s.at = try r.zeros(4);
    s.dummy = try r.zeros(4);
    const row_bytes = heads * lw * 2;
    const outs = [2]u64{ try r.zeros(most * row_bytes), try r.zeros(most * row_bytes) };
    var name: [160]u8 = undefined;
    for (cases) |at| {
        const host_pos, const rows = at;
        const kept = dsa.denseRows(l, host_pos, rows, true);
        const nch = latent.chunksFor(host_pos + rows);
        const p0: i32 = @intCast(host_pos);
        const pos = try r.dev(std.mem.asBytes(&p0));
        for (outs) |o| {
            const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = o, .len = rows * row_bytes };
            try b.fill8(0xff, r.s.handle);
        }
        for (outs, [_]usize{ rows, kept }) |o, n| {
            s.ol = o;
            try dsa.denseAttention(k, r.s, l, c, s, pos, rows, n, nch, part_rows, 0.125);
        }
        const whole = try r.back(outs[0], rows * row_bytes);
        const pruned = try r.back(outs[1], rows * row_bytes);
        const what = try std.fmt.bufPrint(&name, "dense prune: {d} rows at {d}, the {d} kept rows equal the whole chunk's", .{ rows, host_pos, kept });
        try check.sameBytes(what, pruned[0 .. kept * row_bytes], whole[0 .. kept * row_bytes]);
        for (pruned[kept * row_bytes ..]) |b| try check.expect(b == 0xff, "dense prune: {d} rows at {d}, rows past {d} are left unwritten", .{ rows, host_pos, kept });
    }
    try counts(r, ops, l.dense_limit);
    check.pass("dense prune: kept rows equal the whole chunk's bytes in {d} chunks; sparse counts are nonzero from {d} on", .{ cases.len, l.dense_limit });
}

/// tf_glm_dsa_tokens' counts at positions 0 .. most: zero before the limit, nonzero from it (sparse writes those).
fn counts(r: *Rig, ops: cuda.Module, limit: usize) !void {
    const pools = try r.zeros(most * sparse.topk_pools * 8);
    const zero: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = pools, .len = most * sparse.topk_pools * 8 };
    try zero.fill8(0, r.s.handle);
    const tokens = try r.zeros(most * sparse.width * 4);
    const cnt = try r.zeros(most * 4);
    const p0: i32 = 0;
    var a: cuda.Args = .{};
    a.add(pools);
    a.add(try r.dev(std.mem.asBytes(&p0)));
    a.add(@as(c_int, sparse.topk_pools));
    a.add(tokens);
    a.add(cnt);
    a.add(@as(u64, 0));
    a.add(@as(i64, 0));
    try cuda.launch.launch(try ops.function("tf_glm_dsa_tokens"), .{ .grid = .{ .x = most }, .block = .{ .x = 256 } }, r.s, &a);
    const got = std.mem.bytesAsSlice(i32, try r.back(cnt, most * 4));
    for (got, 0..) |n, row| try check.expect((n != 0) == (row >= limit), "dense prune: position {d}'s sparse count is {d}", .{ row, n });
}
