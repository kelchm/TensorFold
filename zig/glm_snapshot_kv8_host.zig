//! Paged E4M3 snapshot round-trips (window <-> pages and stream -> stream on disjoint shuffled pages).
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const t = std.testing;

fn rowAt(st: *const glm.forward.State, x: glm.dsa_block.Caches, base: u64, r: usize, width: usize, per: usize) u64 {
    return if (x.pt == 0) base + r * width else base + @as(u64, st.pages[r / per]) * x.ps + (r % per) * width;
}

fn same(a: u64, b: u64, len: usize) !void {
    try t.expectEqualSlices(u8, rec.bytes(a, len), rec.bytes(b, len));
}

/// Every field an E4M3 prefix at `at` reads, in `a` and in `b` (codes, scales, rings, pooled keys), from either layout.
fn expectSamePrefix8(c: glm.config.Config, a: *const glm.forward.State, b: *const glm.forward.State, at: usize, head: ?usize) !void {
    try same(a.pos_dev, b.pos_dev, 4);
    try same(a.conv, b.conv, a.kda_layers * 3 * a.conv_c * 2);
    try same(a.rec, b.rec, 2 * a.kda_layers * a.rec_layer);
    for (0..a.caches.len + 1) |j| {
        const n = if (j < a.caches.len) at else head orelse break;
        const x = if (j < a.caches.len) a.caches[j] else a.mtp_caches;
        const y = if (j < b.caches.len) b.caches[j] else b.mtp_caches;
        try t.expect(x.ls != 0 and y.ls != 0);
        for (0..n) |r| {
            try same(rowAt(a, x, x.lc, r, c.kv_lora, 512), rowAt(b, y, y.lc, r, c.kv_lora, 512), c.kv_lora);
            try same(rowAt(a, x, x.ls, r, 4, 512), rowAt(b, y, y.ls, r, 4, 512), 4);
        }
        for (0..2) |k| try same(a.rings[2 * j + k], b.rings[2 * j + k], @min(n, a.ring_rows) * a.ring_row);
        for (0..@min(n / 4 + 2, x.pk_rows)) |p| try same(rowAt(a, x, x.pk, p, c.i_dim * 2, 128), rowAt(b, y, y.pk, p, c.i_dim * 2, 128), c.i_dim * 2);
    }
    try same(a.mtp_pos_dev, b.mtp_pos_dev, 4);
    try same(a.prompt_hidden, b.prompt_hidden, c.hidden * 2);
}

test "GLM snapshots: an E4M3 paged sequence keeps and resumes its codes, scales and pooled keys through its pages" {
    var d = rec.init();
    for ([_]bool{ false, true }) |mtp| {
        var f: Fixture = undefined;
        try f.initFormat(&d, 0, 2, mtp, true);
        defer f.deinit();
        const c = f.e.c;
        var w: glm.weights.Weights = undefined;
        w.config = c;
        w.share = f.e.share;
        var layers: [2]glm.weights.Layer = undefined;
        layers[0].mixer = .{ .mla = undefined };
        layers[1].mixer = .{ .kda = undefined };
        w.layers = &layers;
        const part = 512 * (c.kv_lora + 4) + 128 * c.i_dim * 2;
        try t.expectEqual(@as(u64, (1 + @as(usize, @intFromBool(mtp))) * part), glm.state.pageBytesOf(&w, true, mtp));
        const page = (1 + @as(usize, @intFromBool(mtp))) * part;
        const ids = [_]u32{ 5, 2, 9, 0, 7, 1, 3, 8, 4, 6 };
        var kv = try cuda.DeviceBuffer.alloc(&d, ids.len * page);
        defer kv.free();
        var paged = f.seq;
        paged.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 16448, .{ .kv = kv.ptr, .pages = glm.pages.tableLen(16448), .page_bytes = page }, true, mtp);
        defer paged.st.deinit();
        paged.st.pages = &ids;
        paged.st.capacity = ids.len * 512;
        const x = paged.st.caches[0];
        try t.expectEqual(kv.ptr, x.lc);
        try t.expectEqual(kv.ptr + 512 * c.kv_lora, x.ls);
        try t.expectEqual(kv.ptr + 512 * (c.kv_lora + 4), x.pk);
        try t.expectEqual(@as(u64, page), x.ps);
        if (mtp) try t.expectEqual(kv.ptr + part, paged.st.mtp_caches.lc);
        const window = &f.seq;
        try f.e.prepareSnapshots(3 * f.e.stateBytes(4096));
        for ([_]usize{ 511, 1537, 4096 }) |at| {
            const head: ?usize = if (mtp) at + 7 else null;
            f.e.cur = window;
            try f.mark(at, head orelse at, @intCast(at % 251));
            const id = try f.e.keepState();
            @memset(rec.bytes(paged.st.arena.buf.ptr, paged.st.arena.buf.len), 0xE9);
            @memset(rec.bytes(kv.ptr, kv.len), 0xE9);
            f.e.cur = &paged;
            try f.e.restoreState(id);
            try expectSamePrefix8(c, &window.st, &paged.st, at, head);
            try t.expectEqual(at, paged.st.pos);
            f.e.dropState(id);
            f.e.dropped.clearRetainingCapacity();
            const back = try f.e.keepState();
            @memset(rec.bytes(window.st.arena.buf.ptr, window.st.arena.buf.len), 0x5C);
            f.e.cur = window;
            try f.e.restoreState(back);
            try expectSamePrefix8(c, &paged.st, &window.st, at, head);
            f.e.dropState(back);
            f.e.dropped.clearRetainingCapacity();
        }
        f.e.cur = window;
    }
}

test "GLM snapshots: an E4M3 state kept on one paged stream restores into another stream on other pages, its table its own" {
    var d = rec.init();
    var f: Fixture = undefined;
    try f.initFormat(&d, 1, 2, false, true);
    defer f.deinit();
    const c = f.e.c;
    var w: glm.weights.Weights = undefined;
    w.config = c;
    w.share = f.e.share;
    var layers: [2]glm.weights.Layer = undefined;
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = &layers;
    const page = glm.state.pageBytesOf(&w, true, false);
    const ids_a = [_]u32{ 13, 2, 19, 7, 0, 11, 5, 16, 9, 3 };
    const ids_b = [_]u32{ 4, 17, 8, 1, 14, 10, 18, 6, 12, 15 };
    var kv = try cuda.DeviceBuffer.alloc(&d, 20 * page);
    defer kv.free();
    const at: glm.state.Paged = .{ .kv = kv.ptr, .pages = glm.pages.tableLen(16448), .page_bytes = page };
    var a = f.seq;
    a.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 16448, at, true, false);
    defer a.st.deinit();
    var b = f.seq;
    b.st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 16448, at, true, false);
    defer b.st.deinit();
    for ([_]*@TypeOf(a){ &a, &b }, [_][]const u32{ &ids_a, &ids_b }) |q, ids| {
        q.st.pages = ids;
        q.st.capacity = ids.len * 512;
        @memset(rec.bytes(q.st.table, ids.len * 4), 0x77);
    }
    try f.e.prepareSnapshots(2 * f.e.stateBytes(4096));
    for ([_]usize{ 1537, 4096 }) |at_row| {
        @memset(rec.bytes(kv.ptr, kv.len), 0);
        const arena = rec.bytes(a.st.arena.buf.ptr, a.st.arena.buf.len);
        for (arena, 0..) |*x, i| x.* = @truncate(i *% 29 +% at_row);
        @memset(rec.bytes(a.st.table, ids_a.len * 4), 0x77);
        for (ids_a) |id| for (rec.bytes(kv.ptr + id * page, page), 0..) |*x, i| {
            x.* = @truncate(i *% 13 +% id);
        };
        try a.st.restoreHost(f.e.stream, at_row, 0, &.{1}, .{ 0, 0 });
        f.e.cur = &a;
        const id = try f.e.keepState();
        for (ids_b) |pid| @memset(rec.bytes(kv.ptr + pid * page, page), 0xE9);
        const a_pages = try t.allocator.dupe(u8, rec.bytes(kv.ptr, kv.len));
        defer t.allocator.free(a_pages);
        f.e.cur = &b;
        try f.e.restoreState(id);
        try expectSamePrefix8(c, &a.st, &b.st, at_row, null);
        try t.expectEqual(at_row, b.st.pos);
        for (ids_a) |pid| try t.expectEqualSlices(u8, a_pages[pid * page ..][0..page], rec.bytes(kv.ptr + pid * page, page));
        for ([_]*@TypeOf(a){ &a, &b }, [_][]const u32{ &ids_a, &ids_b }) |q, ids| for (rec.bytes(q.st.table, ids.len * 4)) |x| try t.expectEqual(@as(u8, 0x77), x);
        f.e.dropState(id);
        f.e.dropped.clearRetainingCapacity();
    }
    f.e.cur = &f.seq;
}
