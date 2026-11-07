//! Actual State, Plan, capture/capsule, rank writer and filesystem recovery under a recording CUDA ABI.
const std = @import("std");
const sys = @import("sys");
const core = @import("core");
const glm = @import("glm");
const cuda = @import("cuda");
const restore = glm.checkpoint_restore;
const cp = glm.checkpoint;
const epoch = core.checkpoint_epoch;
const wire = epoch.wire;
const rows = @import("src/core/rows_file.zig");
const Store = epoch.Store(rows);
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const t = std.testing;

const Directory = struct {
    path: [128]u8 = undefined,
    len: usize,
    fn init(label: []const u8) !Directory {
        var d: Directory = .{ .len = 0 };
        const p = try std.fmt.bufPrintSentinel(&d.path, "/tmp/tf-glm-restore-080-{d}-{s}", .{ sys.c.getpid(), label }, 0);
        d.len = p.len;
        if (sys.c.mkdir(p, 0o700) != 0) return error.TestDirectory;
        return d;
    }
    fn name(d: *const Directory) [:0]const u8 {
        return d.path[0..d.len :0];
    }
    fn deinit(d: *Directory) void {
        const dir = sys.c.opendir(d.name()) orelse unreachable;
        while (sys.c.readdir(dir)) |ent| {
            const n = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, n, ".") or std.mem.eql(u8, n, "..")) continue;
            var path: [1200]u8 = undefined;
            _ = sys.c.unlink(std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ d.name(), n }, 0) catch unreachable);
        }
        _ = sys.c.closedir(dir);
        _ = sys.c.rmdir(d.name());
    }
};

fn weights(layers: []glm.weights.Layer, kv8: bool) !glm.weights.Weights {
    var w: glm.weights.Weights = undefined;
    w.config = .{ .layers = 2, .kda_heads = 64, .kv_lora = if (kv8) 512 else 16, .i_dim = 8, .hidden = 64 };
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    layers[0].mixer = .{ .mla = undefined };
    layers[1].mixer = .{ .kda = undefined };
    w.layers = layers;
    return w;
}

const Fixture = struct {
    d: cuda.Driver,
    layers: [2]glm.weights.Layer,
    w: glm.weights.Weights,
    device: cuda.DeviceBuffer,
    pool: core.kv_pool.Pool,
    live: core.kv_pool.Lease,
    st: glm.forward.State,
    draft: ?cuda.DeviceBuffer,
    saved: ?cp.Capture = null,
    budget: core.checkpoint_capture.Budget,
    metadata: restore.Metadata,
    agreement: epoch.Agreement,

    fn init(f: *Fixture, kv8: bool, head: bool, draft: bool, mark: usize) !void {
        f.d = rec.init();
        f.w = try weights(&f.layers, kv8);
        const page_bytes = glm.state.pageBytesOf(&f.w, kv8, head);
        f.device = try cuda.DeviceBuffer.alloc(&f.d, page_bytes * 12);
        errdefer f.device.free();
        f.pool = try core.kv_pool.Pool.init(t.allocator, 12, page_bytes);
        errdefer f.pool.deinit();
        f.live = .{};
        errdefer f.pool.finish(&f.live);
        try f.pool.promise(&f.live, 4);
        try f.pool.cover(&f.live, 4, null);
        std.mem.swap(u32, &f.live.ids.items[0], &f.live.ids.items[2]);
        f.st = try glm.forward.State.initPagedFormat(t.allocator, &f.d, &f.w, 2048, .{ .kv = f.device.ptr, .pages = 12, .page_bytes = page_bytes }, kv8, head);
        errdefer f.st.deinit();
        f.st.pages = f.live.ids.items;
        f.st.capacity = 2048;
        fill(rec.bytes(f.device.ptr, f.device.len), 17);
        fill(rec.bytes(f.st.arena.buf.ptr, f.st.arena.buf.len), 23);
        try f.st.restoreHost(f.stream(), mark, if (head) mark + 3 else 0, &.{1}, .{ 512, if (head) 512 else 0 });
        f.draft = if (draft) try cuda.DeviceBuffer.alloc(&f.d, glm.dflash.ring_bytes) else null;
        errdefer if (f.draft) |*d| d.free();
        if (f.draft) |d| fill(rec.bytes(d.ptr, d.len), 31);
        f.budget = .{ .pool = &f.pool, .limits = .{ .descriptors = 1, .unique_pages = 4, .pages_per_descriptor = 4, .capsule_bytes = 128 << 20 } };
        var freezer: glm.checkpoint_freeze.Device = .{ .gpa = t.allocator, .d = &f.d, .stream = f.stream() };
        f.saved = try cp.Capture.create(&f.budget, &f.live, &f.st, &f.w.config, if (f.draft) |d| .{ .dflash2 = .{ .live = d.ptr, .from = @intCast(mark - 512), .through = mark } } else .none, freezer.adapter());
        errdefer f.saved.?.close() catch unreachable;
        var media: restore.Media = .{ .processor = @splat(42), .count = 1 };
        media.images[0] = .{ .digest = @splat(43), .at = 10, .tokens = 4, .gh = 4, .gw = 4 };
        f.metadata = try restore.Metadata.fromCapture(&f.saved.?, &f.st, &f.w.config, media);
        const binding: epoch.Binding = .{ .model = @splat(1), .drafter = @splat(if (draft) @as(u8, 2) else 0), .arithmetic = @splat(3), .prefix = @splat(4), .media = undefined, .layout = undefined };
        f.agreement = .{ .world = 2, .epoch = 1, .mark = mark, .binding = f.metadata.bind(binding) };
    }
    fn stream(f: *Fixture) cuda.Stream {
        return .{ .d = &f.d, .handle = null };
    }
    fn finishCapture(f: *Fixture) !void {
        if (f.saved) |*s| try s.close();
        f.saved = null;
    }
    fn deinit(f: *Fixture) void {
        f.finishCapture() catch unreachable;
        if (f.draft) |*d| d.free();
        f.st.deinit();
        f.pool.finish(&f.live);
        f.pool.deinit();
        f.device.free();
    }
    fn destination(f: *Fixture) !glm.forward.State {
        return glm.forward.State.initPagedFormat(t.allocator, &f.d, &f.w, 2048, .{ .kv = f.device.ptr, .pages = 12, .page_bytes = f.pool.page_bytes }, f.st.kv8, f.st.head);
    }
    fn persist(f: *Fixture, dir: [:0]const u8) !Store.Epoch {
        var slab = try cuda.HostBuffer.alloc(&f.d, 64 * 1024);
        defer slab.free();
        var c = try Store.Coordinator.init(.{ .dir = dir, .validate_detail = restore.validateEnvelope }, f.agreement);
        for (0..f.agreement.world) |rank_num| {
            var cursor: cp.Cursor = .{ .capture = &f.saved.? };
            var slots: [2]core.checkpoint_stream.Slot = undefined;
            var ring = try core.checkpoint_stream.Ring.init(cursor.source(), slab.slice(u8), &slots, 32 * 1024);
            defer ring.close() catch unreachable;
            var w = try core.checkpoint_file.Writer(rows).open(dir, f.agreement.identity(@intCast(rank_num)), 1, f.metadata.payload, epoch.max_metadata);
            defer w.close();
            while (ring.emitted < f.metadata.payload) {
                const ticket = try ring.acquire();
                try f.d.check(f.d.api.cuMemcpyDtoHAsync_v2(ticket.target.ptr, ticket.chunk.location.device, ticket.chunk.len, null), "checkpoint test download");
                try f.stream().synchronize();
                try ring.dmaDone(ticket.key, true);
                try w.consume(&ring, ticket.key);
                try ring.retire(ticket.key);
            }
            var eb: [wire.size(epoch.Envelope)]u8 = undefined;
            var mb: [wire.size(restore.Metadata)]u8 = undefined;
            const env = try f.metadata.envelope(f.agreement, @intCast(rank_num));
            try w.seal(&ring, @intCast(f.metadata.mark), &.{ try wire.encode(env, &eb), try wire.encode(f.metadata, &mb) });
            try c.accept(try w.publish(), slab.slice(u8));
        }
        return c.commit(2, slab.slice(u8));
    }
};
fn fill(bytes: []u8, salt: usize) void {
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% salt +% i / 59);
}
fn oracleBytes(f: *Fixture) ![]u8 {
    const saved = &f.saved.?;
    const result = try t.allocator.alloc(u8, saved.plan.len);
    errdefer t.allocator.free(result);
    var cursor: cp.Cursor = .{ .capture = saved };
    const source = cursor.source();
    while (try source.next(source.ctx, epoch.max_chunk)) |chunk| {
        @memcpy(result[@intCast(chunk.offset)..][0..chunk.len], rec.bytes(chunk.location.device, chunk.len));
    }
    return result;
}
fn equalPlan(st: *glm.forward.State, c: *const glm.config.Config, draft: ?cuda.DeviceBuffer, expected: []const u8) !void {
    var plan = try glm.snapshots.layout.Plan.init(st, c, st.pos, if (st.head) st.mtp_len else null);
    if (draft) |d| try plan.outside(d.ptr, d.len);
    var offset: usize = 0;
    for (plan.parts[0..plan.count]) |p| {
        var done: usize = 0;
        while (done < p.len) {
            const piece = plan.piece(p, done);
            try t.expectEqualSlices(u8, expected[offset..][0..piece.len], rec.bytes(piece.live, piece.len));
            done += piece.len;
            offset += piece.len;
        }
    }
    try t.expectEqual(expected.len, offset);
}

test "checkpoint restore: actual BF16 and FP8 Plan restores KDA, scales, rings, MTP and images onto different pages" {
    for ([_]bool{ false, true }) |kv8| for ([_]usize{ 1024, 1025 }) |mark| {
        var f: Fixture = undefined;
        try f.init(kv8, true, false, mark);
        defer f.deinit();
        var dir = try Directory.init("roundtrip");
        defer dir.deinit();
        const expected = try oracleBytes(&f); // test oracle only; production restore never has this buffer
        defer t.allocator.free(expected);
        _ = try f.persist(dir.name());
        try f.finishCapture();
        @memset(rec.bytes(f.device.ptr, f.device.len), 0xee);
        @memset(rec.bytes(f.st.arena.buf.ptr, f.st.arena.buf.len), 0xdd);
        f.st.cur[0] = 0;
        var target = try f.destination();
        defer target.deinit();
        var lease: core.kv_pool.Lease = .{};
        defer f.pool.finish(&lease);
        try f.pool.promise(&lease, 4);
        var inactive = try restore.Inactive.init(&f.pool, &lease, &target, f.stream(), null);
        defer inactive.release();
        var slab = try cuda.HostBuffer.alloc(&f.d, 65537);
        defer slab.free();
        const store: Store = .{ .dir = dir.name(), .validate_detail = restore.validateEnvelope };
        const recovered = (try store.recover(f.agreement.binding, 2, slab.slice(u8))).?;
        var reader = try recovered.openRank(1);
        defer reader.close();
        const before = rec.allocs;
        const ready = try inactive.load(&reader, &f.w.config, slab.slice(u8));
        try t.expectEqual(before, rec.allocs); // no device snapshot allocation on restore
        try t.expectEqual(mark, ready.mark);
        try t.expectEqual(@as(?i64, null), ready.ring_from);
        try t.expectEqual(.ready, inactive.phase);
        try t.expectEqual(@as(u1, 1), target.cur[0]);
        try t.expectEqual(@as(usize, 512), target.base);
        try t.expectEqual(@as(usize, 512), target.mtp_base);
        try t.expectEqual(mark + 3, target.mtp_len);
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(lease.ids.items), rec.bytes(target.table, lease.ids.items.len * 4));
        try t.expect(lease.ids.items[0] != f.live.ids.items[0]);
        try equalPlan(&target, &f.w.config, null, expected);
        try t.expectError(error.CheckpointRestoreState, inactive.load(&reader, &f.w.config, slab.slice(u8)));
    };
}

test "checkpoint restore: real DFlash2 ring bytes and ring_from survive an FP8 no-head restart" {
    var f: Fixture = undefined;
    try f.init(true, false, true, 1025);
    defer f.deinit();
    var dir = try Directory.init("draft");
    defer dir.deinit();
    const expected = try oracleBytes(&f);
    defer t.allocator.free(expected);
    const saved = try f.persist(dir.name());
    try f.finishCapture();
    var target = try f.destination();
    defer target.deinit();
    var draft = try cuda.DeviceBuffer.alloc(&f.d, glm.dflash.ring_bytes);
    defer draft.free();
    var lease: core.kv_pool.Lease = .{};
    defer f.pool.finish(&lease);
    try f.pool.promise(&lease, 4);
    var inactive = try restore.Inactive.init(&f.pool, &lease, &target, f.stream(), .{ .live = draft.ptr, .bytes = draft.len });
    defer inactive.release();
    var slab = try cuda.HostBuffer.alloc(&f.d, 2 << 20);
    defer slab.free();
    var reader = try saved.openRank(0);
    defer reader.close();
    const ready = try inactive.load(&reader, &f.w.config, slab.slice(u8));
    try t.expectEqual(@as(?i64, 513), ready.ring_from);
    try t.expectEqual(@as(usize, 0), target.mtp_len);
    try equalPlan(&target, &f.w.config, draft, expected);
}

const Stop = struct {
    checks: usize = 0,
    fn check(ptr: ?*anyopaque) !void {
        const s: *Stop = @ptrCast(@alignCast(ptr.?));
        s.checks += 1;
        if (s.checks == 4) return error.Cancelled;
    }
};

test "checkpoint restore: failed allocation, mid-copy, cancellation and late corruption cannot activate or leak destination pages" {
    var f: Fixture = undefined;
    try f.init(true, true, false, 1025);
    defer f.deinit();
    var dir = try Directory.init("failures");
    defer dir.deinit();
    const saved = try f.persist(dir.name());
    try f.finishCapture();
    var slab = try cuda.HostBuffer.alloc(&f.d, 65537);
    defer slab.free();
    for (0..6) |mode| {
        var target = try f.destination();
        defer target.deinit();
        var lease: core.kv_pool.Lease = .{};
        defer f.pool.finish(&lease);
        try f.pool.promise(&lease, 4);
        var inactive = try restore.Inactive.init(&f.pool, &lease, &target, f.stream(), null);
        defer inactive.release();
        var reader = try saved.openRank(0);
        defer reader.close();
        var stop: Stop = .{};
        if (mode == 0) rec.fail_copy = rec.copies + 3;
        if (mode == 1) reader.control = .{ .ctx = &stop, .check_fn = Stop.check };
        var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
        const gpa = f.pool.gpa;
        if (mode == 2) f.pool.gpa = failing.allocator();
        defer f.pool.gpa = gpa;
        if (mode == 3) {
            // File was fully verified at recovery, then changed before/during its restore.
            const stem = std.fmt.bytesToHex(f.agreement.identity(0).name(), .lower);
            var path: [1200]u8 = undefined;
            const fd = sys.c.open(try rows.path(&path, dir.name(), &stem, ".rank", false), .{ .ACCMODE = .WRONLY }, @as(sys.c.mode_t, 0));
            try t.expect(fd >= 0);
            defer _ = sys.c.close(fd);
            try t.expectEqual(@as(isize, 1), sys.c.pwrite(fd, "!", 1, @intCast(f.metadata.payload - 1)));
        }
        if (mode == 5) {
            const stem = std.fmt.bytesToHex(f.agreement.identity(0).name(), .lower);
            var path: [1200]u8 = undefined;
            const fd = sys.c.open(try rows.path(&path, dir.name(), &stem, ".rank", false), .{ .ACCMODE = .WRONLY }, @as(sys.c.mode_t, 0));
            try t.expect(fd >= 0);
            defer _ = sys.c.close(fd);
            try t.expectEqual(@as(c_int, 0), sys.c.ftruncate(fd, @intCast(f.metadata.payload / 2)));
        }
        if (mode == 4) reader.expected = null; // a rank file without the all-rank proof is refused before writing
        const syncs = rec.syncs;
        if (inactive.load(&reader, &f.w.config, slab.slice(u8))) |_| return error.ExpectedFailure else |err| {
            if (mode == 1) try t.expectEqual(error.Cancelled, err);
            if (mode == 2) try t.expectEqual(error.OutOfMemory, err);
            if (mode == 3) try t.expectEqual(error.RowsHash, err);
            if (mode == 4) try t.expectEqual(error.CheckpointNotDurable, err);
            if (mode == 5) try t.expectEqual(error.RowsRead, err);
        }
        rec.fail_copy = null;
        try t.expectEqual(.poisoned, inactive.phase);
        try t.expect(rec.syncs > syncs);
        try t.expectEqual(@as(usize, 0), target.pos);
        f.pool.gpa = gpa;
        inactive.release();
        try t.expectEqual(@as(usize, 0), lease.ids.items.len + lease.promised);
        try t.expectEqual(@as(usize, 4), f.pool.nleased);
    }
    rec.fail_alloc = true;
    if (f.destination()) |value| {
        var unexpected = value;
        unexpected.deinit();
        return error.ExpectedFailure;
    } else |_| {}
    rec.fail_alloc = false;
    const host_alloc = f.d.api.cuMemHostAlloc;
    f.d.api.cuMemHostAlloc = failHost;
    if (cuda.HostBuffer.alloc(&f.d, 1 << 20)) |value| {
        var unexpected = value;
        unexpected.free();
        return error.ExpectedFailure;
    } else |_| {}
    f.d.api.cuMemHostAlloc = host_alloc;
}

test "checkpoint restore: version, parity, media geometry, ring origin and arithmetic metadata reject before writes" {
    var f: Fixture = undefined;
    try f.init(false, true, false, 1025);
    defer f.deinit();
    var dir = try Directory.init("metadata");
    defer dir.deinit();
    const saved = try f.persist(dir.name());
    for (0..8) |kind| {
        var m = f.metadata;
        switch (kind) {
            0 => m.version += 1,
            1 => m.cur[0] = 2,
            2 => m.media.images[0].gw += 2,
            3 => m.media.images[0].digest[0] ^= 1,
            4 => m.ring_from = 1,
            5 => m.bases[0] = 3,
            6 => m.shape.flags ^= 1,
            7 => m.pages_required = std.math.maxInt(u64),
            else => unreachable,
        }
        var target = try f.destination();
        defer target.deinit();
        var lease: core.kv_pool.Lease = .{};
        defer f.pool.finish(&lease);
        try f.pool.promise(&lease, 4);
        var inactive = try restore.Inactive.init(&f.pool, &lease, &target, f.stream(), null);
        defer inactive.release();
        var reader = try saved.openRank(0);
        defer reader.close();
        _ = try wire.encode(m, reader.metadata[wire.size(epoch.Envelope)..]);
        var slab: [4096]u8 = undefined;
        const before = rec.copies;
        if (inactive.load(&reader, &f.w.config, &slab)) |_| return error.ExpectedFailure else |_| {}
        try t.expectEqual(before, rec.copies);
        try t.expectEqual(@as(usize, 0), lease.ids.items.len);
    }
    var partial = f.metadata.media;
    partial.images[0].at = 1024;
    try t.expectError(error.CheckpointMedia, partial.validate(1025));
}

fn failHost(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
    return cuda.abi.error_not_ready;
}

const Logical = struct {
    var next: u64 = 256 << 30;
    var allocated: usize = 0;
    var copied: u64 = 0;
    var largest: usize = 0;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        allocated += n;
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
    fn deviceCopy(_: u64, _: u64, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn upload(_: u64, _: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        copied += n;
        largest = @max(largest, n);
        return 0;
    }
    fn set(_: u64, _: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

// Only a logical geometry/budget probe. The real filesystem tests above exercise the verified Reader.
const LogicalReader = struct {
    metadata: [wire.size(restore.Metadata)]u8,
    envelope: epoch.Envelope,
    offset: u64 = 0,
    largest: usize = 0,
    pub fn check(_: *const LogicalReader) !void {}
    pub fn durable(_: *const LogicalReader) bool {
        return true;
    }
    pub fn detail(r: *const LogicalReader) []const u8 {
        return &r.metadata;
    }
    pub fn read(r: *LogicalReader, into: []u8) !usize {
        const n: usize = @intCast(@min(into.len, r.envelope.payload - r.offset));
        r.largest = @max(r.largest, n);
        r.offset += n;
        return n;
    }
    pub fn finish(r: *LogicalReader) !void {
        try t.expectEqual(r.envelope.payload, r.offset);
    }
};

test "checkpoint restore: 1M actual GLM FP8 Plan restores over six GiB through two MiB with zero snapshot allocation" {
    var d = rec.init();
    d.api.cuMemAlloc_v2 = Logical.alloc;
    d.api.cuMemFree_v2 = Logical.free;
    d.api.cuMemsetD8_v2 = Logical.zero;
    d.api.cuMemcpyDtoDAsync_v2 = Logical.deviceCopy;
    d.api.cuMemcpyHtoDAsync_v2 = Logical.upload;
    d.api.cuMemsetD32Async = Logical.set;
    d.api.cuStreamSynchronize = Logical.sync;
    var w: glm.weights.Weights = undefined;
    w.config = .{};
    w.share = try glm.shard.Share.of(&w.config, 0, 2);
    var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
    w.layers = layers[0..w.config.layers];
    for (w.layers, 0..) |*l, i| l.mixer = if (w.config.kind(@intCast(i)) == .mla) .{ .mla = undefined } else .{ .kda = undefined };
    const page_bytes = glm.state.pageBytesOf(&w, true, false);
    const at = 1_000_001;
    const pages = (at + 511) / 512 + 1;
    var pool = try core.kv_pool.Pool.init(t.allocator, 2 * pages + 8, page_bytes);
    defer pool.deinit();
    var source: core.kv_pool.Lease = .{};
    defer pool.finish(&source);
    try pool.promise(&source, pages);
    try pool.cover(&source, pages, null);
    const paged: glm.state.Paged = .{ .kv = 64 << 30, .pages = 2 * pages + 8, .page_bytes = page_bytes };
    var st = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 1 << 20, paged, true, false);
    defer st.deinit();
    st.capacity = pages * 512;
    st.pages = source.ids.items;
    st.pos = at;
    st.base = (at - 4096) / 4 * 4;
    var budget: core.checkpoint_capture.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = pages, .pages_per_descriptor = pages, .capsule_bytes = 512 << 20 } };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    var freezer: glm.checkpoint_freeze.Device = .{ .gpa = t.allocator, .d = &d, .stream = stream };
    var captured = try cp.Capture.create(&budget, &source, &st, &w.config, .{ .dflash2 = .{ .live = 96 << 30, .from = at - 2048, .through = at } }, freezer.adapter());
    defer captured.close() catch unreachable;
    const m = try restore.Metadata.fromCapture(&captured, &st, &w.config, .{});
    const binding: epoch.Binding = .{ .model = @splat(1), .drafter = @splat(2), .arithmetic = @splat(3), .layout = undefined, .media = undefined, .prefix = @splat(4) };
    const a: epoch.Agreement = .{ .world = 2, .epoch = 1, .mark = at, .binding = m.bind(binding) };
    var reader: LogicalReader = .{ .metadata = undefined, .envelope = try m.envelope(a, 0) };
    _ = try wire.encode(m, &reader.metadata);
    var target = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 1 << 20, paged, true, false);
    defer target.deinit();
    var lease: core.kv_pool.Lease = .{};
    defer pool.finish(&lease);
    try pool.promise(&lease, pages);
    var inactive = try restore.Inactive.init(&pool, &lease, &target, stream, .{ .live = 128 << 30, .bytes = glm.dflash.ring_bytes });
    defer inactive.release();
    const slab = try t.allocator.alloc(u8, epoch.max_chunk);
    defer t.allocator.free(slab);
    @memset(slab, 0);
    Logical.allocated = 0;
    Logical.copied = 0;
    Logical.largest = 0;
    const ready = try inactive.load(&reader, &w.config, slab);
    try t.expectEqual(@as(usize, at), ready.mark);
    try t.expectEqual(@as(?i64, at - 2048), ready.ring_from);
    try t.expectEqual(@as(usize, 0), Logical.allocated);
    try t.expectEqual(@as(u64, 6_649_248_044), m.payload);
    try t.expectEqual(m.payload + 4 * m.pages_required, Logical.copied);
    try t.expect(Logical.largest <= epoch.max_chunk);
    try t.expectEqual(@as(usize, epoch.max_chunk), reader.largest);
    try t.expectEqual(m.pages_required, lease.ids.items.len);
    try t.expectEqual(@as(usize, 0), pool.reclaimable());
}

fn failSet(_: u64, _: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    return cuda.abi.error_not_ready;
}

test "checkpoint restore: a final host-position upload failure poisons the copied state until its lease is discarded" {
    var f: Fixture = undefined;
    try f.init(false, true, false, 1025);
    defer f.deinit();
    var dir = try Directory.init("finalize");
    defer dir.deinit();
    const saved = try f.persist(dir.name());
    var target = try f.destination();
    defer target.deinit();
    var lease: core.kv_pool.Lease = .{};
    defer f.pool.finish(&lease);
    try f.pool.promise(&lease, 4);
    var inactive = try restore.Inactive.init(&f.pool, &lease, &target, f.stream(), null);
    defer inactive.release();
    var reader = try saved.openRank(0);
    defer reader.close();
    var slab = try cuda.HostBuffer.alloc(&f.d, 65536);
    defer slab.free();
    f.d.api.cuMemsetD32Async = failSet;
    try t.expectError(error.NotReady, inactive.load(&reader, &f.w.config, slab.slice(u8)));
    try t.expect(reader.finished); // payload and hashes succeeded; host/device cursor publication did not
    try t.expectEqual(.poisoned, inactive.phase);
    try t.expectEqual(@as(usize, 1025), f.st.pos); // the source/active lease is untouched
    inactive.release();
    try t.expectEqual(@as(usize, 0), lease.ids.items.len + lease.promised);
    try t.expectEqual(@as(usize, 4), f.pool.nleased);
}

test "checkpoint restore: uneven TP rank arenas bind one global layout while retaining exact local geometry" {
    var d = rec.init();
    var layers: [2]glm.weights.Layer = undefined;
    var w = try weights(&layers, true);
    w.share = try glm.shard.Share.of(&w.config, 0, 3);
    const page_bytes = glm.state.pageBytesOf(&w, true, true);
    var first = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 2048, .{ .kv = 64 << 30, .pages = 4, .page_bytes = page_bytes }, true, true);
    defer first.deinit();
    const a = try restore.Shape.of(&first, &w.config);
    w.share = try glm.shard.Share.of(&w.config, 2, 3);
    var last = try glm.forward.State.initPagedFormat(t.allocator, &d, &w, 2048, .{ .kv = 96 << 30, .pages = 4, .page_bytes = page_bytes }, true, true);
    defer last.deinit();
    const b = try restore.Shape.of(&last, &w.config);
    try t.expectEqual(@as(u64, 22), a.heads);
    try t.expectEqual(@as(u64, 21), b.heads);
    try t.expect(a.arena_bytes != b.arena_bytes);
    try t.expect(!std.meta.eql(a, b));
    try t.expectEqualSlices(u8, &a.layoutIdentity(), &b.layoutIdentity());
}
