//! DFlash allocation boundaries and the production loader's partial owner, without a GPU.
const std = @import("std");
const glm = @import("glm");
const registry = @import("core").contract.registry;
const t = std.testing;

test "glm dflash: the live-pricing setting accepts only supported modes" {
    const Probe = struct {
        var raw: ?[]const u8 = null;
        fn get(key: [:0]const u8) ?[]const u8 {
            return if (std.mem.eql(u8, key, "TF_GLM_DFLASH_LIVE_PRICES")) raw else null;
        }
    };
    var issue: []const u8 = "";
    for ([_][]const u8{ "0", "1", "2" }, 0..) |input, i| {
        Probe.raw = input;
        const settings = try glm.ranks.Settings.read(Probe.get, &issue);
        try t.expectEqual(@as(u2, @intCast(i)), settings.dflash_live_prices);
    }
    for ([_][]const u8{ "3", "-1", "yes" }) |input| {
        Probe.raw = input;
        try t.expectError(error.BadSetting, glm.ranks.Settings.read(Probe.get, &issue));
        try t.expect(std.mem.startsWith(u8, issue, "TF_GLM_DFLASH_LIVE_PRICES"));
    }
}

test "glm dflash: live pricing needs copies and the external drafter; each mode has a distinct rank identity" {
    var e: glm.engine.Engine = undefined;
    var dr: glm.dflash.Drafter = undefined;
    e.drafter = &dr;
    e.mtp = true;
    var a: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    const h: [32]u8 = @splat(0);
    for (0..3) |n| {
        const mode: u2 = @intCast(n);
        a.live_prices = mode;
        const m = a.factsWithCopies(true);
        try t.expectEqual(@as(u32, if (n == 0) 0 else 4), m.lone_cost_tile);
        try t.expectEqual(@as(@TypeOf(m.lone_probe), switch (n) { 0 => .all, 1 => .observed_only, else => .positive_chances }), m.lone_probe);
        try t.expectEqual(@as(u32, 0), a.factsWithCopies(false).lone_cost_tile);
        const fp = glm.ranks.Fingerprint.of(2, 0, .{ .dflash_live_prices = mode }, 8192, true, h, h, h);
        try t.expectEqual(@as(u32, 1) + 8 * @as(u32, mode), fp.mtp);
    }
    e.drafter = null;
    try t.expectEqual(@as(u32, 0), a.factsWithCopies(true).lone_cost_tile);
}

test "glm dflash: each optional decode switch has its own rank fingerprint across all eight combinations" {
    const ranks = glm.ranks;
    const h: [32]u8 = @splat(0);
    for (0..8) |i| {
        const s: ranks.Settings = .{ .dflash_only = i & 1 != 0, .l2pf = .{ .shared_first = i & 2 != 0 }, .exl3_decode = true, .exl3_nmajor = i & 4 != 0 };
        const f = ranks.Fingerprint.of(2, 0, s, 8192, true, h, h, h);
        try t.expectEqual(@as(u32, 36), f.layout);
        try t.expectEqual(@as(u32, 1 | (@as(u32, @intFromBool(s.dflash_only)) << 2)), f.mtp);
        try t.expectEqual(@as(u32, 1 | (@as(u32, @intFromBool(s.l2pf.shared_first)) << 1)), f.l2pf[0]);
        try t.expectEqual(@as(u32, 2 | (@as(u32, @intFromBool(s.exl3_nmajor)) << 2)) | @intFromBool(s.pair_experts), f.pair_experts);
        try t.expect(ranks.mismatch(&.{f, ranks.Fingerprint.of(2, 1, s, 8192, true, h, h, h)}) == null);
    }
}

test "glm dflash: external-only keeps the draft projection and scheduling when the MTP layer is absent" {
    var e: glm.engine.Engine = undefined;
    var dr: glm.dflash.Drafter = undefined;
    e.drafter = &dr;
    e.mtp = false;
    e.w.mtp = null;
    e.w.external_head = null;
    try t.expect(e.w.draftHead() == null);
    const head: glm.q4.Q4 = .{ .w = 11, .s = 12, .b = 13, .n = 64, .k = 128, .npad = 64 };
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

test "glm dflash: calibrated greedy edge changes the candidate and its temperature changes the chance" {
    const df = glm.dflash;
    var dr: df.Drafter = undefined;
    dr.world = 1;
    dr.temp = 1;
    var cand: [df.BLOCK * df.CW]i32 = @splat(-1);
    var gh: [df.BLOCK * df.RANK]f32 = @splat(0);
    var pred: [2 * df.RANK]u16 = @splat(0);
    var succ: [2 * df.RANK]u16 = @splat(0);
    // The first predecessor feature is one for either token. Token 0 gains an edge of one.
    pred[0] = 0x3f80;
    pred[df.RANK] = 0x3f80;
    succ[0] = 0x3f80;
    for (1..df.BLOCK) |r| {
        const i = r * df.CW;
        cand[i] = 0;
        cand[i + 1] = 1;
        cand[i + df.TOPK] = @bitCast(@as(f32, 0));
        cand[i + df.TOPK + 1] = @bitCast(@as(f32, 0.75));
        gh[r * df.RANK] = 1;
    }
    dr.cand = &cand;
    dr.gh = &gh;
    dr.pred = &pred;
    dr.succ = &succ;
    var ids: [df.BLOCK - 1]u32 = undefined;
    var probs: [df.BLOCK - 1]f64 = undefined;
    var q: df.Req = .{ .slot = 0, .pos = 0, .n = 0, .anchor = 0, .out = &ids, .probs = &probs };
    for ([_]bool{ false, true }) |calibrated| {
        q.calibrated = calibrated;
        df.selector.select(&dr, 0, df.BLOCK, q);
        const delta: f64 = if (calibrated) (0.75 - 0.6) / 2.0 else (1.0 - 0.75);
        const expected = 1.0 / (1.0 + @exp(-delta));
        for (ids, probs) |id, p| {
            try t.expectEqual(@as(u32, @intFromBool(calibrated)), id);
            try t.expectApproxEqAbs(expected, p, 1e-8);
        }
    }
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
        // dflash_load's 4-bit linears, rows by columns: qkv, kv, o, gate and up, the kernel projections, fc
        const linears = [_][2]usize{ .{ df.QKV, df.D }, .{ 2 * df.KVW, df.D }, .{ df.D, df.D }, .{ 2 * ff, df.D }, .{ df.KVW, df.D }, .{ df.D, 5 * df.D } };
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
