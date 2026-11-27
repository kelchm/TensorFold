//! DFlash allocation boundaries and the production loader's partial owner, without a GPU.
const std = @import("std");
const glm = @import("glm");
const registry = @import("core").contract.registry;
const t = std.testing;

test "glm dflash: external-only keeps the draft projection and scheduling when the MTP layer is absent" {
    var e: glm.engine.Engine = undefined;
    var dr: glm.dflash.Drafter = undefined;
    e.drafter = &dr;
    e.mtp = false;
    e.w.mtp = null;
    e.w.external_head = null;
    try t.expect(e.w.draftHead() == null);
    const head: glm.tfq3.DraftHead = .{ .q4 = .{ .w = 11, .s = 12, .b = 13, .n = 64, .k = 128, .npad = 64 } };
    e.w.external_head = head;
    try t.expectEqualDeep(head, e.w.draftHead().?);
    var adapter: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    const standalone = adapter.factsWithCopies(true);
    try t.expect(standalone.speculate and standalone.draft_streams and standalone.hidden_rows);
    e.mtp = true;
    const both = adapter.factsWithCopies(true);
    try t.expectEqualDeep(standalone, both);

    const h: [32]u8 = @splat(0);
    const ranks = glm.ranks;
    const fp0 = ranks.Fingerprint.of(2, 0, .{}, 8192, true, h, h, h);
    const fp1 = ranks.Fingerprint.of(2, 1, .{ .dflash_only = true }, 8192, true, h, h, h);
    try t.expectEqual(ranks.Fingerprint.Field.mtp, ranks.mismatch(&.{ fp0, fp1 }).?.field);
}

test "glm dflash: selector skips out-of-vocabulary candidate ids and clamps a corrupt anchor" {
    const df = glm.dflash;
    var dr: df.Drafter = undefined;
    dr.world = 1;
    dr.temp = 1.0;
    const pred = try t.allocator.alloc(u16, df.VOCAB * df.RANK);
    defer t.allocator.free(pred);
    @memset(pred, 0);
    const succ = try t.allocator.alloc(u16, df.VOCAB * df.RANK);
    defer t.allocator.free(succ);
    @memset(succ, 0);
    var gh: [df.BLOCK * df.RANK]f32 = @splat(0);
    var cand: [df.BLOCK * df.CW]i32 = @splat(-1);
    // Row 1 has no-winner 0x7fffffff alongside one valid token id (42).
    cand[1 * df.CW + 0] = 0x7fff_ffff;
    cand[1 * df.CW + df.TOPK + 0] = @bitCast(@as(f32, 99.0));
    cand[1 * df.CW + 1] = 42;
    cand[1 * df.CW + df.TOPK + 1] = @bitCast(@as(f32, 1.5));
    // Row 2 has only out-of-vocab ids, so selection stops cleanly without indexing past the codebook.
    cand[2 * df.CW + 0] = @intCast(df.VOCAB);
    cand[2 * df.CW + df.TOPK + 0] = @bitCast(@as(f32, 5.0));
    dr.pred = pred;
    dr.succ = succ;
    dr.gh = &gh;
    dr.cand = &cand;
    var out: [df.BLOCK - 1]u32 = @splat(0);
    var probs: [df.BLOCK - 1]f64 = @splat(1.0);
    glm.dflash_select.select(&dr, 0, df.BLOCK, .{ .slot = 0, .pos = 0, .n = 0, .anchor = df.VOCAB + 7, .propose = true, .out = &out, .probs = &probs });
    try t.expectEqual(@as(u32, 42), out[0]);
    try t.expectEqual(@as(u32, 42), out[1]);
    try t.expectEqual(@as(f64, 0.0), probs[1]);
}

test "glm dflash: gather and read fit for every supported rank count" {
    for (1..17) |world| for ([_]usize{ 1, 8, glm.dflash.MAX_M }) |rows| {
        const n = glm.dflash_candidates.Layout.init(world, rows, glm.dflash.CW);
        const gathered = world * rows * glm.dflash.CW;
        try t.expect(gathered <= n.device);
        try t.expect(n.read <= n.host);
        try t.expectEqual(gathered, n.read);
    };
}

test "glm dflash: the raw scratch holds every 4-bit linear's words, scales and biases at every rank count" {
    const df = glm.dflash;
    var short: usize = 0;
    for (1..17) |world| for ([_]bool{ false, true }) |wide| {
        const ff = df.mlpSplit(world, wide).ff;
        // dflash_load's 4-bit linears, rows by columns: qkv, kv, o, gate and up, down, the kernel projections, fc
        const linears = [_][2]usize{ .{ df.QKV, df.D }, .{ 2 * df.KVW, df.D }, .{ df.D, df.D }, .{ 2 * ff, df.D }, .{ df.D, ff }, .{ df.KVW, df.D }, .{ df.D, 5 * df.D } };
        for (linears) |shape| {
            const nk = shape[0] * shape[1];
            const need = std.mem.alignForward(usize, nk / 2, 256) + std.mem.alignForward(usize, nk / 32, 256) + nk / 32;
            if (need <= df.rawBytes(4, world, wide)) continue;
            std.debug.print("world {d}: a {d}x{d} linear needs {d} raw bytes, the scratch has {d}\n", .{ world, shape[0], shape[1], need, df.rawBytes(4, world, wide) });
            short += 1;
        }
    };
    try t.expectEqual(@as(usize, 0), short);
}

test "glm dflash: the MLP splits at two ranks as on main; three or more only with TF_ROCE=1" {
    const df = glm.dflash;
    const whole: df.MlpSplit = .{ .split = false, .ff = df.FF };
    try t.expectEqual(whole, df.mlpSplit(1, true));
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = df.FF / 2 }, df.mlpSplit(2, false)); // 6144: main's split, no opt-in
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = df.FF / 2 }, df.mlpSplit(2, true));
    for ([_]usize{ 3, 4, 6, 8 }) |world| try t.expectEqual(whole, df.mlpSplit(world, false)); // main's whole MLP on every rank
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = df.FF / 4 }, df.mlpSplit(4, true)); // 3072 on four Sparks
    try t.expectEqual(df.MlpSplit{ .split = true, .ff = 4096 }, df.mlpSplit(3, true));
    try t.expectEqual(whole, df.mlpSplit(5, true)); // 12288 / 5 is no share
    try t.expectEqual(whole, df.mlpSplit(7, true));
    for (1..17) |world| for ([_]bool{ false, true }) |wide| {
        const m = df.mlpSplit(world, wide);
        try t.expect(m.ff % 256 == 0); // the q4 pack's rows, glm_mm's K tiles and df_swiglu's columns all fit
        if (m.split) {
            try t.expectEqual(df.FF, m.ff * world); // the shares are the whole width
            for (0..world) |rank| try t.expect(rank * m.ff + m.ff <= df.FF); // every rank's slice lies inside
        } else try t.expectEqual(df.FF, m.ff);
    };
}

const Resources = struct {
    pred: []u16 = &.{},
    succ: []u16 = &.{},
    scratch: []u8 = &.{},
    h_cpos: []i64 = &.{},
    h_crings: []u64 = &.{},
    cand: []i32 = &.{},
    gh: []f32 = &.{},
};
const Ops = struct {
    g: std.mem.Allocator,
    fail_read: ?usize = null,
    pub fn readBook(o: *Ops, i: usize, dst: []u16) !void {
        if (o.fail_read == i) return error.BadBook;
        @memset(dst, 7);
    }
    pub fn scratch(o: *Ops, r: *Resources) !void {
        r.scratch = try o.g.alloc(u8, 64);
    }
    pub fn freeScratch(o: *Ops, r: *Resources) void {
        o.g.free(r.scratch);
    }
};
fn release(g: std.mem.Allocator, r: Resources) void {
    g.free(r.pred);
    g.free(r.succ);
    g.free(r.scratch);
    g.free(r.h_cpos);
    g.free(r.h_crings);
    g.free(r.cand);
    g.free(r.gh);
}
fn attempt(fail_index: usize, fail_read: ?usize) !void {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var f = std.testing.FailingAllocator.init(arena.allocator(), .{ .fail_index = fail_index });
    const g = f.allocator();
    var r: Resources = .{};
    var o: Ops = .{ .g = g, .fail_read = fail_read };
    if (glm.dflash_load_owner.acquire(g, &r, &o, Ops, .{ .book = 16, .ctx = 8, .cand = 32, .hidden = 16 })) |_| {
        try t.expect(fail_index >= 7 and fail_read == null);
        try t.expect(f.allocated_bytes > f.freed_bytes);
        release(g, r);
    } else |err| {
        try t.expect(err == error.OutOfMemory or err == error.BadBook);
    }
    try t.expectEqual(f.allocated_bytes, f.freed_bytes);
}

test "glm dflash: every allocation failure releases earlier tail resources" {
    for (0..8) |i| try attempt(i, null);
}

test "glm dflash: either book read failure releases its allocated storage" {
    try attempt(99, 0);
    try attempt(99, 1);
}

test "glm dflash: every loaded symbol has a measured launch record" {
    inline for (@typeInfo(glm.dflash.Kernels).@"struct".field_names) |name| {
        const symbol = "df_" ++ name;
        var found = false;
        for (registry.all_cuda) |r| if (std.mem.eql(u8, symbol, r.launch.symbol)) {
            found = true;
            try t.expectEqualStrings("glm_dflash", r.launch.image);
            for (r.usage) |u| try t.expect(u.regs > 0);
        };
        try t.expect(found);
    }
}
