//! Actual GLM Plan, State, pool and capsule copies under a recording CUDA ABI, without a GPU.
const std = @import("std");
const core = @import("core");
const glm = @import("glm");
const cuda = @import("cuda");
const cp = glm.checkpoint;
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const t = std.testing;

fn weights(layers: []glm.weights.Layer, kv8: bool) !glm.weights.Weights {
    var w: glm.weights.Weights = undefined;
    w.config = .{ .layers = 2, .kda_heads = 64, .kv_lora = if (kv8) 512 else 16, .i_dim = 8, .hidden = 64 };
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = layers;
    return w;
}

test "checkpoint GLM: actual Plan streams FP8 scales, pooled lookahead, KDA and ring bases after live mutation" {
    for ([_]bool{ false, true }) |kv8| for ([_][2]usize{ .{ 1024, 1027 }, .{ 1025, 1028 }, .{ 1025, 1023 } }) |mark| {
        var d = rec.init();
        var layers: [2]glm.weights.Layer = undefined;
        var w = try weights(&layers, kv8);
        const page_bytes = glm.state.pageBytesOf(&w, kv8, true);
        var device = try cuda.DeviceBuffer.alloc(&d, page_bytes * 6);
        defer device.free();
        var pool = try core.kv_pool.Pool.init(t.allocator, 6, page_bytes);
        defer pool.deinit();
        var live: core.kv_pool.Lease = .{};
        defer pool.finish(&live);
        try pool.promise(&live, 5);
        try pool.cover(&live, 3, null);
        // Deliberately noncontiguous order; a linear-prefix copy cannot pass this case.
        std.mem.swap(u32, &live.ids.items[0], &live.ids.items[2]);
        var st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 2048, .{ .kv = device.ptr, .pages = 6, .page_bytes = page_bytes }, kv8, true);
        defer st.deinit();
        st.capacity = 1536;
        st.pages = live.ids.items;
        const stream: cuda.Stream = .{ .d = &d, .handle = null };
        for (rec.bytes(device.ptr, device.len), 0..) |*b, i| b.* = @truncate(i *% 137 +% i / 59);
        for (rec.bytes(st.arena.buf.ptr, st.arena.buf.len), 0..) |*b, i| b.* = @truncate(i *% 71 +% i / 131);
        try st.restoreHost(stream, mark[0], mark[1], &.{1}, .{ 512, 516 });
        const plan = try glm.snapshots.layout.Plan.init(&st, &w.config, st.pos, st.mtp_len);
        // Small fixture oracle only: the implementation never allocates this whole-prefix byte array.
        const expected = try t.allocator.alloc(u8, plan.len);
        defer t.allocator.free(expected);
        var off: usize = 0;
        for (plan.parts[0..plan.count]) |part| {
            var done: usize = 0;
            while (done < part.len) {
                const p = plan.piece(part, done);
                @memcpy(expected[off..][0..p.len], rec.bytes(p.live, p.len));
                off += p.len;
                done += p.len;
            }
        }
        var budget: core.checkpoint_capture.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 2, .pages_per_descriptor = 2, .capsule_bytes = 64 << 20 } };
        var freezer: glm.checkpoint_freeze.Device = .{ .gpa = t.allocator, .d = &d, .stream = stream };
        try t.expectError(error.UnsupportedMutableCapture, cp.Capture.create(&budget, &live, &st, &w.config, .none, null));
        var saved = try cp.Capture.create(&budget, &live, &st, &w.config, .none, freezer.adapter());
        defer saved.close() catch unreachable;
        try t.expectEqual(@min(mark[0], mark[1]) / 512, pool.npinned);
        try t.expectEqual(@as(usize, 2), pool.promised);
        try t.expectEqualSlices(usize, &.{ 512, 516 }, &saved.bases);
        // Every mutable byte changes while the completed page bytes remain owned by the descriptor.
        @memset(rec.bytes(st.arena.buf.ptr, st.arena.buf.len), 0xee);
        for (live.ids.items[saved.pages.ids.len..]) |id| @memset(rec.bytes(device.ptr + @as(u64, id) * page_bytes, page_bytes), 0xdd);
        st.cur[0] = 0;
        pool.finish(&live);
        try t.expectEqual(@as(usize, 0), pool.reclaimable());
        var cursor: cp.Cursor = .{ .capture = &saved };
        var slots: [2]core.checkpoint_stream.Slot = undefined;
        var storage: [8192]u8 = undefined;
        var ring = try core.checkpoint_stream.Ring.init(cursor.source(), &storage, &slots, 4096);
        while (ring.emitted < plan.len) {
            const ticket = try ring.acquire();
            try d.check(d.api.cuMemcpyDtoHAsync_v2(ticket.target.ptr, ticket.chunk.location.device, ticket.chunk.len, null), "test checkpoint transfer");
            try ring.dmaDone(ticket.key, true);
            const bytes = try ring.beginWrite(ticket.key);
            try t.expectEqualSlices(u8, expected[ticket.chunk.offset..][0..bytes.len], bytes);
            try ring.ioDone(ticket.key, true);
            try ring.retire(ticket.key);
        }
        try t.expect(ring.readyToSeal());
        try ring.close();
        try t.expectEqual(@as(u1, 1), saved.cur[0]);
        try t.expectEqual(mark[0], saved.pos);
    };
}

const Sizing = struct {
    var next: u64 = 1 << 30;
    var largest: usize = 0;
    var copied: usize = 0;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        largest = @max(largest, n);
        out.* = next;
        next += std.mem.alignForward(usize, n, 256);
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn copy(_: u64, _: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copied += n;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "checkpoint GLM: near 1M first save freezes a bounded capsule including 40 MiB DFlash2, not the prefix" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Sizing.alloc;
    d.api.cuMemFree_v2 = Sizing.free;
    d.api.cuMemsetD8_v2 = Sizing.zero;
    d.api.cuMemcpyDtoDAsync_v2 = Sizing.copy;
    d.api.cuStreamSynchronize = Sizing.sync;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*l, i| l.mixer = if (w.config.kind(@intCast(i)) == .mla) .{ .mla = undefined } else .{ .kda = undefined };
    const page_bytes = glm.state.pageBytesOf(&w, true, false);
    const at = 1_000_001;
    const pages = (at + 511) / 512 + 1;
    var pool = try core.kv_pool.Pool.init(t.allocator, pages + 8, page_bytes);
    defer pool.deinit();
    var live: core.kv_pool.Lease = .{};
    defer pool.finish(&live);
    try pool.promise(&live, pages + 8);
    try pool.cover(&live, pages, null);
    var st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 1 << 20, .{ .kv = 64 << 30, .pages = pages + 8, .page_bytes = page_bytes }, true, false);
    defer st.deinit();
    st.capacity = pages * 512;
    st.pages = live.ids.items;
    st.pos = at;
    st.base = at - 8192;
    var budget: core.checkpoint_capture.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = pages, .pages_per_descriptor = pages, .capsule_bytes = 512 << 20 } };
    var freezer: glm.checkpoint_freeze.Device = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null } };
    const draft: cp.Draft = .{ .dflash2 = .{ .live = 96 << 30, .from = at - 2048, .through = at } };
    Sizing.largest = 0;
    Sizing.copied = 0;
    var saved = try cp.Capture.create(&budget, &live, &st, &w.config, draft, freezer.adapter());
    defer if (!saved.pages.closed) saved.close() catch unreachable;
    try t.expectEqual(@as(u64, 40 << 20), glm.dflash.ring_bytes);
    try t.expectEqual(saved.frozen.len, Sizing.largest);
    try t.expectEqual(saved.frozen.len, Sizing.copied);
    try t.expect(saved.frozen.len < 512 << 20);
    try t.expect(saved.frozen.len > 40 << 20);
    try t.expect(saved.plan.len > 5 * @as(usize, 1 << 30));
    try t.expectEqual(@as(usize, at / 512), saved.pages.ids.len);
    try t.expectEqual(@as(usize, 8), pool.promised);
    try t.expectEqual(@as(?i64, at - 2048), saved.ring_from);
    var cursor: cp.Cursor = .{ .capture = &saved };
    const source = cursor.source();
    var total: u64 = 0;
    while (try source.next(source.ctx, 2 << 20)) |chunk| {
        try t.expect(chunk.len <= 2 << 20);
        try t.expectEqual(total, chunk.offset);
        total += chunk.len;
    }
    try t.expectEqual(saved.plan.len, total);
    try saved.close();
    st.pos = 8192 + at % 512;
    const small_draft: cp.Draft = .{ .dflash2 = .{ .live = 96 << 30, .from = @intCast(st.pos - 2048), .through = st.pos } };
    var small = try cp.Capture.create(&budget, &live, &st, &w.config, small_draft, freezer.adapter());
    defer small.close() catch unreachable;
    try t.expectEqual(saved.frozen.len, small.frozen.len);
}

test "checkpoint GLM: failed native freeze drains before releasing admission and page owners" {
    var d = rec.init();
    var layers: [2]glm.weights.Layer = undefined;
    var w = try weights(&layers, false);
    const page_bytes = glm.state.pageBytesOf(&w, false, false);
    var device = try cuda.DeviceBuffer.alloc(&d, page_bytes * 3);
    defer device.free();
    var pool = try core.kv_pool.Pool.init(t.allocator, 3, page_bytes);
    defer pool.deinit();
    var live: core.kv_pool.Lease = .{};
    defer pool.finish(&live);
    try pool.promise(&live, 3);
    try pool.cover(&live, 3, null);
    var st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 1024, .{ .kv = device.ptr, .pages = 3, .page_bytes = page_bytes }, false, false);
    defer st.deinit();
    st.capacity = 1536;
    st.pages = live.ids.items;
    st.pos = 512;
    var b: core.checkpoint_capture.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 3, .pages_per_descriptor = 3, .capsule_bytes = 1 } };
    var freezer: glm.checkpoint_freeze.Device = .{ .gpa = t.allocator, .d = &d, .stream = .{ .d = &d, .handle = null } };
    const allocs = rec.allocs;
    st.pending = 1;
    try t.expectError(error.NotAtMark, cp.Capture.create(&b, &live, &st, &w.config, .none, freezer.adapter()));
    st.pending = 0;
    const table = st.table;
    st.table = 0;
    try t.expectError(error.UnsupportedUnpagedCapture, cp.Capture.create(&b, &live, &st, &w.config, .none, freezer.adapter()));
    st.table = table;
    st.head = true;
    st.pos = 1536;
    try t.expectError(error.UnsupportedHeadLag, cp.Capture.create(&b, &live, &st, &w.config, .none, freezer.adapter()));
    st.head = false;
    st.pos = 512;
    try t.expectError(error.CheckpointBackpressure, cp.Capture.create(&b, &live, &st, &w.config, .none, freezer.adapter()));
    try t.expectEqual(allocs, rec.allocs);
    b.limits.capsule_bytes = 64 << 20;
    rec.fail_copy = 2;
    if (cp.Capture.create(&b, &live, &st, &w.config, .none, freezer.adapter())) |value| {
        var unexpected = value;
        try unexpected.close();
        return error.ExpectedFailure;
    } else |_| {}
    try t.expect(rec.syncs > 0);
    try t.expectEqual(@as(usize, 0), pool.npinned + b.active + b.capsule_used);
    try t.expectEqual(@as(u32, 1), pool.holders(0));
}
