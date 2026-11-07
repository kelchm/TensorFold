//! Test fixtures adapted from prefix080's GLM restore host tests; host-backed CUDA ABI, no GPU access.
const std = @import("std");
const sys = @import("sys");
pub const core = @import("core");
pub const glm = @import("glm");
pub const cuda = @import("cuda");
pub const rec = @import("../test_support/glm_snap_driver.zig").Driver;
pub const f = @import("format.zig");
pub const fs = @import("files.zig");
pub const capture = @import("capture.zig");
pub const local = @import("local.zig");
pub const protocol = @import("protocol.zig");
pub const t = std.testing;

pub const Directory = struct {
    path: [128]u8 = undefined,
    len: usize,
    pub fn init(label: []const u8) !Directory {
        var d: Directory = .{ .len = 0 };
        const p = try std.fmt.bufPrintSentinel(&d.path, "/tmp/tf-glm-block-0950-{d}-{s}", .{ sys.c.getpid(), label }, 0);
        d.len = p.len;
        if (sys.c.mkdir(p, 0o700) != 0) return error.TestDirectory;
        return d;
    }
    pub fn name(d: *const Directory) [:0]const u8 {
        return d.path[0..d.len :0];
    }
    pub fn deinit(d: *Directory) void {
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

pub fn fill(bytes: []u8, salt: usize) void {
    for (bytes, 0..) |*b, i| b.* = @truncate(i *% salt +% i / 59);
}

pub const Fixture = struct {
    d: cuda.Driver,
    layers: [2]glm.weights.Layer,
    w: glm.weights.Weights,
    device: cuda.DeviceBuffer,
    pool: core.kv_pool.Pool,
    live: core.kv_pool.Lease,
    st: glm.forward.State,
    draft: ?cuda.DeviceBuffer,
    saved: ?glm.checkpoint.Capture = null,
    budget: core.checkpoint_capture.Budget,
    metadata: glm.checkpoint_restore.Metadata,
    agreement: f.epoch.Agreement,
    tokens: [2049]u32,
    starts: [8]u32 = undefined,
    starts_len: usize = 0,
    slab: cuda.HostBuffer,
    transfer: capture.Device,
    stats: capture.Stats = .{},
    admission_fail: bool = false,

    pub fn init(x: *Fixture, kv8: bool, head: bool, draft: bool, world: u32) !void {
        x.d = rec.init();
        x.w = undefined;
        x.w.config = .{ .layers = 2, .kda_heads = 64, .kv_lora = if (kv8) 512 else 16, .i_dim = 8, .hidden = 64 };
        x.w.share = try glm.shard.Share.of(&x.w.config, 0, world);
        x.layers[0].mixer = .{ .mla = undefined };
        x.layers[1].mixer = .{ .kda = undefined };
        x.w.layers = &x.layers;
        const page_bytes = glm.state.pageBytesOf(&x.w, kv8, head);
        x.device = try cuda.DeviceBuffer.alloc(&x.d, page_bytes * 12);
        errdefer x.device.free();
        x.pool = try core.kv_pool.Pool.init(t.allocator, 12, page_bytes);
        errdefer x.pool.deinit();
        x.live = .{};
        errdefer x.pool.finish(&x.live);
        try x.pool.promise(&x.live, 4);
        try x.pool.cover(&x.live, 4, null);
        std.mem.swap(u32, &x.live.ids.items[0], &x.live.ids.items[2]);
        x.st = try glm.forward.State.initPagedFormat(t.allocator, &x.d, &x.w, 2048, .{ .kv = x.device.ptr, .pages = 12, .page_bytes = page_bytes }, kv8, head);
        errdefer x.st.deinit();
        x.st.pages = x.live.ids.items;
        x.st.capacity = 2048;
        fill(rec.bytes(x.device.ptr, x.device.len), 17);
        x.draft = if (draft) try cuda.DeviceBuffer.alloc(&x.d, glm.dflash.ring_bytes) else null;
        errdefer if (x.draft) |*d| d.free();
        if (x.draft) |d| fill(rec.bytes(d.ptr, d.len), 31);
        x.budget = .{ .pool = &x.pool, .limits = .{ .descriptors = 1, .unique_pages = 4, .pages_per_descriptor = 4, .capsule_bytes = 128 << 20 } };
        x.slab = try cuda.HostBuffer.alloc(&x.d, 65537);
        x.transfer = .{ .stream = x.stream() };
        x.saved = null;
        x.stats = .{};
        x.admission_fail = false;
        for (&x.tokens, 0..) |*token, i| token.* = @intCast(i + 1);
        x.agreement = .{ .world = world, .epoch = 1, .mark = 0, .binding = .{ .model = @splat(1), .drafter = @splat(if (draft) @as(u8, 2) else 0), .arithmetic = @splat(3), .prefix = @splat(4), .media = undefined, .layout = undefined } };
        try x.mark(512);
    }
    pub fn stream(x: *Fixture) cuda.Stream {
        return .{ .d = &x.d, .handle = null };
    }
    pub fn finishCapture(x: *Fixture) !void {
        if (x.saved) |*s| try s.close();
        x.saved = null;
    }
    pub fn mark(x: *Fixture, at: usize) !void {
        try x.finishCapture();
        fill(rec.bytes(x.st.arena.buf.ptr, x.st.arena.buf.len), 23 + at);
        try x.st.restoreHost(x.stream(), at, if (x.st.head) at + 3 else 0, &.{1}, .{ 0, 0 });
        var freezer: glm.checkpoint_freeze.Device = .{ .gpa = t.allocator, .d = &x.d, .stream = x.stream() };
        x.saved = try glm.checkpoint.Capture.create(&x.budget, &x.live, &x.st, &x.w.config, if (x.draft) |d| .{ .dflash2 = .{ .live = d.ptr, .from = @intCast(at - 512), .through = at } } else .none, freezer.adapter());
        x.metadata = try glm.checkpoint_restore.Metadata.fromCapture(&x.saved.?, &x.st, &x.w.config, .{});
        x.starts_len = 0;
        var start: u32 = 0;
        while (start < at) : (start += 512) {
            x.starts[x.starts_len] = start;
            x.starts_len += 1;
        }
        x.starts[x.starts_len] = @intCast(at);
        x.starts_len += 1;
        x.agreement.mark = at;
        x.agreement.binding.prefix = try @import("names.zig").prefix(x.input(), &x.metadata);
        x.agreement.binding = x.metadata.bind(x.agreement.binding);
        x.stats = .{};
    }
    pub fn deinit(x: *Fixture) void {
        x.finishCapture() catch unreachable;
        x.slab.free();
        if (x.draft) |*d| d.free();
        x.st.deinit();
        x.pool.finish(&x.live);
        x.pool.deinit();
        x.device.free();
    }
    fn admit(ptr: *anyopaque, demand: capture.Demand) !void {
        const x: *Fixture = @ptrCast(@alignCast(ptr));
        if (x.admission_fail or demand.pages > 4 or demand.slab_bytes > 65537 or demand.maintenance_bytes > 8 << 20) return error.AdmissionRefused;
    }
    pub fn input(x: *Fixture) @import("names.zig").Input {
        return .{ .tokens = &x.tokens, .spans = &.{}, .starts = x.starts[0..x.starts_len] };
    }
    pub fn request(x: *Fixture, rank: u32) capture.Request {
        return .{ .gpa = t.allocator, .saved = &x.saved.?, .metadata = &x.metadata, .agreement = x.agreement, .rank = rank, .input = x.input(), .slab = x.slab.slice(u8), .transfer = x.transfer.adapter(), .admission = .{ .ctx = x, .admit = admit }, .stats = &x.stats };
    }
    pub fn save(x: *Fixture, s: *fs.Files, rank: u32) !f.Receipt {
        return (try capture.save(.experimental, s, x.request(rank))).?;
    }
    pub fn oracle(x: *Fixture) ![]u8 {
        const result = try t.allocator.alloc(u8, x.saved.?.plan.len);
        errdefer t.allocator.free(result);
        var cursor: glm.checkpoint.Cursor = .{ .capture = &x.saved.? };
        const source = cursor.source();
        while (try source.next(source.ctx, f.epoch.max_chunk)) |chunk| {
            @memcpy(result[@intCast(chunk.offset)..][0..chunk.len], rec.bytes(chunk.location.device, chunk.len));
        }
        return result;
    }
    pub fn restore(x: *Fixture, e: *const protocol.Epoch, expected: []const u8) !void {
        var target = try glm.forward.State.initPagedFormat(t.allocator, &x.d, &x.w, 2048, .{ .kv = x.device.ptr, .pages = 12, .page_bytes = x.pool.page_bytes }, x.st.kv8, x.st.head);
        defer target.deinit();
        var lease: core.kv_pool.Lease = .{};
        defer x.pool.finish(&lease);
        try x.pool.promise(&lease, 4);
        var draft = if (x.draft != null) try cuda.DeviceBuffer.alloc(&x.d, glm.dflash.ring_bytes) else null;
        defer if (draft) |*d| d.free();
        var inactive = try glm.checkpoint_restore.Inactive.init(&x.pool, &lease, &target, x.stream(), if (draft) |d| .{ .live = d.ptr, .bytes = d.len } else null);
        defer inactive.release();
        var reader = try e.reader(x.slab.slice(u8));
        defer reader.close();
        const allocs = rec.allocs;
        const ready = try inactive.load(&reader, &x.w.config, x.slab.slice(u8));
        try t.expectEqual(allocs, rec.allocs);
        try t.expectEqual(e.manifest.agreement.mark, ready.mark);
        var plan = try glm.snapshots.layout.Plan.init(&target, &x.w.config, target.pos, if (target.head) target.mtp_len else null);
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
        try t.expectEqual(@as(u1, 1), target.cur[0]);
        try t.expect(!std.mem.eql(u32, target.pages, x.st.pages));
    }
};

pub fn loopback(_: *anyopaque, _: protocol.Phase, _: u64, send: []const u8, recv: []u8) !void {
    if (send.len != recv.len) return error.TestTransport;
    @memcpy(recv, send);
}
pub fn single() protocol.Transport {
    return .{ .ctx = @ptrFromInt(1), .world = 1, .rank = 0, .session = @splat(7), .scope = @splat(8), .exchange = loopback };
}
pub fn commitOne(s: *fs.Files, receipt: f.Receipt, slab: []u8) !protocol.Epoch {
    var transport = single();
    return protocol.commit(s, receipt.agreement, receipt, &transport, slab);
}
pub fn accepts(ptr: *const anyopaque, a: f.epoch.Agreement) bool {
    const b: *const f.epoch.Binding = @ptrCast(@alignCast(ptr));
    // Synthetic histories in these tests share tokens through earlier marks; live selection must prove that itself.
    var want = b.*;
    want.prefix = a.binding.prefix;
    return std.meta.eql(want, a.binding);
}
pub fn selection(binding: *const f.epoch.Binding) local.Selection {
    return .{ .ctx = binding, .accepts = accepts };
}
