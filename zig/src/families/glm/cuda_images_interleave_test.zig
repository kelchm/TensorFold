//! Real bounded CUDA media ownership, with recording driver calls and a counted CPU allocator.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const media = @import("cuda_images.zig");
const follow = @import("cuda_follow.zig");
const Images = media.Images;
const Image = lanes.stream.Image;
const t = std.testing;

const Counted = struct {
    live: usize = 0,
    peak: usize = 0,
    allocations: usize = 0,
    fail_at: usize = std.math.maxInt(usize),
    fn allocator(c: *Counted) std.mem.Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = std.mem.Allocator.noResize, .remap = std.mem.Allocator.noRemap, .free = free } };
    }
    fn alloc(ctx: *anyopaque, n: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const c: *Counted = @ptrCast(@alignCast(ctx));
        if (c.allocations == c.fail_at) return null;
        const p = t.allocator.rawAlloc(n, alignment, ret) orelse return null;
        c.allocations += 1;
        c.live += n;
        c.peak = @max(c.peak, c.live);
        return p;
    }
    fn free(ctx: *anyopaque, bytes: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const c: *Counted = @ptrCast(@alignCast(ctx));
        c.live -= bytes.len;
        t.allocator.rawFree(bytes, alignment, ret);
    }
};

const Fake = struct {
    const feature_base = 0x100000000;
    const map_base = 0x2000;
    const link_base = 0x3000;
    var driver_allocs: usize = 0;
    var driver_live: usize = 0;
    var driver_peak: usize = 0;
    var feature_bytes: usize = 0;
    var queued = false;
    var syncs: usize = 0;
    var encoded: usize = 0;
    var encoded_ids: [256]u32 = undefined;
    var values: [media.step_rows + 16384]u32 = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    var injected: [media.step_rows / 2 + 1]u32 = undefined;
    var header: follow.Header = undefined;
    var broadcasts: [128]struct { kind: u32, bytes: usize, rows: bool } = undefined;
    var broadcasts_len: usize = 0;
    var encode_failure = false;

    fn reset() void {
        driver_allocs = 0;
        driver_live = 0;
        driver_peak = 0;
        feature_bytes = 0;
        queued = false;
        syncs = 0;
        encoded = 0;
        broadcasts_len = 0;
        encode_failure = false;
        @memset(&values, 0);
    }
    fn alloc(out: *u64, n: usize) callconv(.c) c_int {
        out.* = feature_base;
        driver_allocs += 1;
        driver_live += n;
        feature_bytes = n;
        driver_peak = @max(driver_peak, driver_live);
        return 0;
    }
    fn free(_: u64) callconv(.c) c_int {
        std.debug.assert(!queued);
        driver_live -= feature_bytes;
        return 0;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        queued = false;
        syncs += 1;
        return 0;
    }
    fn uploadAsync(dst: u64, ptr: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        queued = true;
        const src: [*]const u8 = @ptrCast(ptr.?);
        if (dst == map_base) @memcpy(std.mem.sliceAsBytes(&map)[0..n], src[0..n]);
        if (dst == link_base and n == @sizeOf(follow.Header)) @memcpy(std.mem.asBytes(&header), src[0..n]);
        return 0;
    }
    fn copy(dst: u64, src: u64, bytes: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        queued = true;
        const d = (dst - feature_base) / media.row_bytes;
        const s = (src - feature_base) / media.row_bytes;
        const n = bytes / media.row_bytes;
        @memcpy(values[d..][0..n], values[s..][0..n]);
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) c_int {
        const bytes = t.allocator.alignedAlloc(u8, .@"16", n + 16) catch return cuda.abi.error_not_ready;
        std.mem.writeInt(usize, bytes[0..@sizeOf(usize)], n + 16, .little);
        out.* = bytes.ptr + 16;
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) c_int {
        const base: [*]align(16) u8 = @alignCast(@as([*]u8, @ptrCast(ptr.?)) - 16);
        const n = std.mem.readInt(usize, base[0..@sizeOf(usize)], .little);
        t.allocator.free(base[0..n]);
        return 0;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn eventDone(_: cuda.abi.Event) callconv(.c) c_int {
        return cuda.abi.error_not_ready;
    }
    fn eventDestroy(_: cuda.abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn launch(function: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        queued = true;
        if (function == @as(cuda.abi.Function, @ptrFromInt(2))) {
            const w: *const follow.Words = @ptrCast(@alignCast(args.?[2].?));
            @memcpy(std.mem.asBytes(&header), std.mem.sliceAsBytes(w.w[0..16]));
        } else {
            const rows: *const c_int = @ptrCast(@alignCast(args.?[3].?));
            for (map[0..@intCast(rows.*)], 0..) |m, i| injected[i] = if (m < 0) 0 else values[@intCast(m)];
        }
        return 0;
    }
    fn broadcast(ptr: u64, _: u64, n: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) c_int {
        broadcasts[broadcasts_len] = .{ .kind = header.kind, .bytes = n, .rows = ptr == feature_base };
        broadcasts_len += 1;
        queued = true;
        return 0;
    }
    fn driver(d: *cuda.Driver) void {
        d.* = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuStreamSynchronize = sync;
        d.api.cuMemcpyHtoDAsync_v2 = uploadAsync;
        d.api.cuMemcpyDtoDAsync_v2 = copy;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventDestroy_v2 = eventDestroy;
        d.api.cuLaunchKernel = launch;
    }
    fn encode(_: void, pixels: []const f32, img: Image, out: u64) !void {
        encoded_ids[encoded] = @intFromFloat(pixels[0]);
        encoded += 1;
        queued = true;
        if (encode_failure) return error.RecordedEncodeFailure;
        const base = (out - feature_base) / media.row_bytes;
        for (values[base..][0..img.tokens], 0..) |*v, i| v.* = @as(u32, @intFromFloat(pixels[0])) * 100000 + @as(u32, @intCast(i));
    }
};

const Source = struct {
    id: f32,
    bytes: usize = 4096,
    calls: usize = 0,
    fail: bool = false,
    cancel: bool = false,
    cancel_during: bool = false,
    fn prepare(ctx: *const anyopaque, a: std.mem.Allocator) anyerror![]const f32 {
        const self: *Source = @ptrCast(@alignCast(@constCast(ctx)));
        self.calls += 1;
        const pixels = try a.alloc(f32, self.bytes / 4);
        pixels[0] = self.id;
        if (self.cancel_during) self.cancel = true;
        if (self.fail) return error.RecordedMaterializationFailure;
        return pixels;
    }
    fn cancelled(ctx: ?*anyopaque) bool {
        const self: *const Source = @ptrCast(@alignCast(ctx.?));
        return self.cancel;
    }
    fn image(self: *Source, at: u32, tokens: u32) Image {
        return .{ .pixels = &.{}, .deferred = .{ .ctx = self, .prepare = prepare }, .at = at, .tokens = tokens, .gh = 2, .gw = tokens * 2, .hash = @intFromFloat(self.id) };
    }
};

fn setup(v: *Images, d: *cuda.Driver, a: std.mem.Allocator, host: []i32) !void {
    Fake.reset();
    Fake.driver(d);
    v.* = .{ .gpa = a, .io = undefined, .d = d, .module = undefined, .run = .{ .stream = .{ .d = d, .handle = @ptrFromInt(0x55) }, .f = undefined }, .rows = undefined, .map = .{ .d = d, .ptr = Fake.map_base, .len = host.len * 4, .borrowed = true }, .host_map = host, .limits = .{ .image_tokens = 102400, .max_images = 50, .video_tokens = 32768, .max_videos = 4, .max_frames = 128 }, .video_ids = .{ 1, 2, 3 } };
    v.run.f.inject = .{ .d = d, .handle = null };
    v.rows = try cuda.DeviceBuffer.alloc(d, v.scratchBytes());
}
fn cleanup(v: *Images) void {
    v.run.stream.synchronize() catch unreachable;
    v.rows.free();
    v.spans.deinit(v.gpa);
    v.saved.deinit(v.gpa);
}

test "media lifecycle: 50 full-quality images keep one global feature reservation and one CPU materialization" {
    var cpu: Counted = .{};
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, cpu.allocator(), &map);
    defer cleanup(&v);
    var source: Source = .{ .id = 7, .bytes = 2048 * 4 * 1176 * 4 };
    var imgs: [50]Image = undefined;
    for (&imgs, 0..) |*img, i| img.* = source.image(@intCast(i * 2048), 2048);
    try v.begin(1, &imgs, 0);
    try t.expectEqual(@as(usize, 0), source.calls);
    for (0..25) |i| {
        try v.windowWith(i * 4096, (i + 1) * 4096, null, {}, Fake.encode);
        try t.expectEqual(@as(usize, 4096), v.rowCount());
        try v.splice(v.run.stream, 1, i * 4096, (i + 1) * 4096, 0x4000, 4);
        try t.expectEqual(@as(u32, 702047), Fake.injected[2047]);
        v.finishWindow();
    }
    try t.expectEqual(@as(usize, 50), source.calls);
    try t.expectEqual(@as(usize, 1), Fake.driver_allocs);
    try t.expectEqual(@as(usize, 192 << 20), Fake.driver_peak);
    try t.expectEqual(@as(usize, 0), v.streamBytes());
    try t.expect(cpu.peak < source.bytes * 2);
    std.debug.print("media receipt: shared feature peak={d} CPU arena peak={d} one image payload={d} calls={d}\n", .{ Fake.driver_peak, cpu.peak, source.bytes, source.calls });
    try v.begin(1, &imgs, 49 * 2048);
    try v.windowWith(0, 2048, null, {}, Fake.encode);
    try t.expectEqual(@as(usize, 50), source.calls);
    try v.windowWith(49 * 2048, 50 * 2048, null, {}, Fake.encode);
    try t.expectEqual(@as(usize, 51), source.calls);
    v.clear();
    try t.expectEqual(@as(usize, 0), v.source.images.len);
}

test "media lifecycle: chunk boundaries repeated images A B A and cached-prefix replay preserve every row" {
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, t.allocator, &map);
    defer cleanup(&v);
    var a: Source = .{ .id = 1 };
    var b: Source = .{ .id = 2 };
    const aa = [_]Image{ a.image(2047, 4), a.image(4096, 4) };
    const bb = [_]Image{b.image(1, 2)};
    try v.begin(7, &aa, 0);
    try v.windowWith(0, 2048, null, {}, Fake.encode);
    try v.splice(v.run.stream, 7, 0, 2048, 0x4000, 1);
    try t.expectEqual(@as(u32, 100000), Fake.injected[2047]);
    v.finishWindow();
    try v.begin(8, &bb, 0);
    try v.windowWith(0, 2048, null, {}, Fake.encode);
    try v.splice(v.run.stream, 8, 0, 3, 0x4000, 1);
    try t.expectEqualSlices(u32, &.{ 0, 200000, 200001 }, Fake.injected[0..3]);
    v.finishWindow();
    try v.activate(7);
    try v.windowWith(2048, 4096, null, {}, Fake.encode);
    try v.splice(v.run.stream, 7, 2048, 2052, 0x4000, 1);
    try t.expectEqualSlices(u32, &.{ 100001, 100002, 100003, 0 }, Fake.injected[0..4]);
    v.finishWindow();
    // Suspension releases device contents, not the request-owned descriptors. Replay is deterministic.
    try v.windowWith(2048, 4096, null, {}, Fake.encode);
    try t.expectEqual(@as(u32, 100001), Fake.values[0]);
    v.finishWindow();
    // Rewind begins anew; resume skips exactly the cached image's tokens, including a partial image.
    try v.begin(7, &aa, 2049);
    try v.windowWith(2048, 4096, null, {}, Fake.encode);
    try t.expectEqual(@as(u32, 2049), v.spans.items[0].at);
    try t.expectEqual(@as(u32, 100002), Fake.values[0]);
    v.release(7);
    try t.expectEqual(@as(usize, 0), v.source.images.len);
    try v.activate(8);
    try t.expectEqual(@as(usize, 1), v.source.images.len);
    v.release(0); // releasing an unbound zero slot cannot erase B's source
    v.release(8);
    try t.expectEqual(@as(usize, 0), v.source.images.len);
    try v.begin(7, &aa, 0);
    try v.windowWith(4096, 4100, null, {}, Fake.encode);
    try t.expectEqual(@as(u32, 100003), Fake.values[3]);
    try t.expectEqual(@as(usize, 1), Fake.driver_allocs);
}

test "media lifecycle: cancellation materialization failure tower failure and OOM discard partial rows" {
    var cpu: Counted = .{};
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, cpu.allocator(), &map);
    defer cleanup(&v);
    var a: Source = .{ .id = 1 };
    var b: Source = .{ .id = 2, .fail = true };
    const imgs = [_]Image{ a.image(0, 2), b.image(2, 2) };
    try v.begin(1, &imgs, 0);
    try t.expectError(error.RecordedMaterializationFailure, v.windowWith(0, 4, null, {}, Fake.encode));
    try t.expectEqual(@as(usize, 0), v.rowCount());
    try t.expect(!Fake.queued);
    const held = cpu.live;
    b.fail = false;
    a.cancel = true;
    const cancel: Images.Cancel = .{ .ptr = &a, .check = Source.cancelled };
    try t.expectError(error.Cancelled, v.windowWith(0, 4, cancel, {}, Fake.encode));
    try t.expectEqual(held, cpu.live);
    a.cancel = false;
    a.cancel_during = true;
    try t.expectError(error.Cancelled, v.windowWith(0, 4, cancel, {}, Fake.encode));
    try t.expectEqual(held, cpu.live);
    a.cancel = false;
    a.cancel_during = false;
    cpu.fail_at = cpu.allocations;
    try t.expectError(error.OutOfMemory, v.windowWith(0, 4, null, {}, Fake.encode));
    cpu.fail_at = std.math.maxInt(usize);
    Fake.encode_failure = true;
    try t.expectError(error.RecordedEncodeFailure, v.windowWith(0, 4, null, {}, Fake.encode));
    try t.expect(!Fake.queued);
    try t.expectEqual(held, cpu.live);
    Fake.encode_failure = false;
    try v.windowWith(0, 4, null, {}, Fake.encode);
    try t.expectEqualSlices(u32, &.{ 100000, 100001, 200000, 200001 }, Fake.values[0..4]);
    v.clear();
    try t.expectEqual(@as(usize, 0), v.rowCount());
    try t.expectEqual(@as(usize, 0), v.source.images.len);
}

// This tests the actual wire sender and receiver, not a substitute collective protocol.
test "media lifecycle: actual TP image spans then rows then chunk and cancellation before forward" {
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, t.allocator, &map);
    defer cleanup(&v);
    var lib: cuda.nccl.Library = undefined;
    lib.api.ncclBroadcast = Fake.broadcast;
    var words: [8192]u32 = undefined;
    var link: follow.Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = v.run.stream, .buf = .{ .d = &d, .ptr = Fake.link_base, .len = 64 + words.len * 4, .borrowed = true }, .ids = &words, .gpa = t.allocator, .wait = false };
    var e: @import("cuda_engine.zig").Engine = undefined;
    e.slot = 1;
    e.images = &v;
    e.dropped = .empty;
    e.prompt_logprobs = null;
    e.m.glue.put_words = .{ .d = &d, .handle = @ptrFromInt(2) };
    const x: follow.Leader = .{ .e = &e, .link = &link };
    var a: Source = .{ .id = 1 };
    const imgs = [_]Image{a.image(0, 2)};
    try v.begin(1, &imgs, 0);
    try t.expect(!try x.announceChunkWith(&.{ 1, 2 }, 0, 2, 1, null, null, false, recordedWindow));
    try t.expectEqual(@as(usize, 3), Fake.broadcasts_len);
    try t.expectEqual(@as(u32, @backingInt(follow.Kind.image)), Fake.broadcasts[0].kind);
    try t.expect(Fake.broadcasts[1].rows);
    try t.expectEqual(@as(usize, 2 * media.row_bytes), Fake.broadcasts[1].bytes);
    try t.expectEqual(@as(u32, @backingInt(follow.Kind.chunk)), Fake.broadcasts[2].kind);
    const packet = try follow.spanWords(t.allocator, v.spans.items, words.len);
    defer t.allocator.free(packet);
    v.finishWindow();
    try v.take(1, packet);
    try t.expectEqual(@as(usize, 2), v.rowCount());
    v.finishWindow();
    a.cancel = true;
    Fake.broadcasts_len = 0;
    try t.expect(try x.announceChunk(&.{ 1, 2 }, 0, 2, 1, null, .{ .ptr = &a, .check = Source.cancelled }, false));
    try t.expectEqual(@as(usize, 2), Fake.broadcasts_len); // empty image header, chunk header, no row collective
    try t.expectEqual(@as(usize, 0), v.rowCount());
    a.cancel = false;
    var repeated: [50]Image = undefined;
    for (&repeated, 0..) |*img, i| img.* = a.image(@intCast(2 * i), 1);
    try v.begin(1, &repeated, 0);
    const long_part: [101]u32 = @splat(1);
    Fake.broadcasts_len = 0;
    link.keep_pending = .{ .slot = 1, .rows = 1 };
    try t.expect(!try x.announceChunkWith(&long_part, 0, long_part.len, 1, null, null, true, recordedWindow));
    try t.expectEqual(@as(usize, 6), Fake.broadcasts_len);
    try t.expectEqual(@as(u32, @backingInt(follow.Kind.keep)), Fake.broadcasts[0].kind);
    try t.expectEqual(@as(u32, @backingInt(follow.Kind.image)), Fake.broadcasts[1].kind);
    try t.expectEqual(follow.head_bytes, Fake.broadcasts[1].bytes);
    try t.expectEqual(@as(usize, 64 + 150 * 4), Fake.broadcasts[2].bytes);
    try t.expect(Fake.broadcasts[3].rows);
    try t.expectEqual(@as(usize, 50 * media.row_bytes), Fake.broadcasts[3].bytes);
    try t.expectEqual(@as(u32, @backingInt(follow.Kind.chunk)), Fake.broadcasts[4].kind);
    try t.expectEqual(@as(usize, 64 + 101 * 4), Fake.broadcasts[5].bytes);
    v.finishWindow();
    try v.begin(1, &imgs, 0);
    a.fail = true;
    Fake.broadcasts_len = 0;
    try t.expectError(error.RecordedMaterializationFailure, x.announceChunk(&.{ 1, 2 }, 0, 2, 1, null, null, false));
    try t.expectEqual(@as(usize, 0), Fake.broadcasts_len); // no peer was told to enter a forward pass
}

fn recordedWindow(v: *Images, lo: usize, hi: usize, cancel: ?Images.Cancel) !void {
    try v.windowWith(lo, hi, cancel, {}, Fake.encode);
}

test "media lifecycle: a full 16384-token group crosses native single and paired chunk boundaries" {
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, t.allocator, &map);
    defer cleanup(&v);
    var a: Source = .{ .id = 4 };
    const c = @import("cuda_forward.zig").prefill_rows;
    const imgs = [_]Image{a.image(c - 1, 16384)};
    try v.begin(3, &imgs, 0);
    try v.windowWith(0, 2 * c, null, {}, Fake.encode);
    try t.expectEqual(@as(usize, c + 1), v.rowCount());
    try v.splice(v.run.stream, 3, c, 2 * c, 0x4000, 1);
    try t.expectEqual(@as(u32, 400001), Fake.injected[0]);
    try t.expectEqual(@as(u32, 400000 + c), Fake.injected[c - 1]);
    v.finishWindow();
    try v.windowWith(2 * c, 4 * c, null, {}, Fake.encode);
    try t.expectEqual(@as(u32, 400001 + c), Fake.values[0]);
    try t.expectEqual(@as(u32, 400000 + 3 * c), Fake.values[2 * c - 1]);
    v.finishWindow();
    try v.windowWith(4 * c, 5 * c, null, {}, Fake.encode);
    try t.expectEqual(@as(usize, c - 1), v.rowCount());
    try t.expectEqual(@as(u32, 400000 + 16383), Fake.values[c - 2]);
    try t.expectEqual(@as(usize, 3), a.calls);
    v.finishWindow();
    try t.expectError(error.ImageRows, v.take(3, &.{ 0, media.step_rows + 1, 0 }));
    try t.expectError(error.ImageRows, v.take(3, &.{ 0, 1, 1 }));
    try t.expectError(error.ImageRows, v.take(3, &.{ 1, 2, 0, 2, 1, 2 }));
    try t.expectError(error.ImageRows, v.take(3, &.{ std.math.maxInt(u32), 1, 0 }));
    try t.expectError(error.BadStep, v.take(3, &.{ 0, 1 }));
}

const video = @import("video_core.zig").For(@import("core").picture);
const config: @import("vision_config.zig").Config = .{ .image_token = 1, .image_start = 2, .image_end = 3, .mean = @splat(0.5), .std = @splat(0.5), .min_tokens = 16, .max_tokens = 2048 };
const Frames = struct {
    calls: usize = 0,
    active: usize = 0,
    max_active: usize = 0,
    max_indices: usize = 0,
    fail_after: ?usize = null,
    cancelled: bool = false,
    cancel_after: ?usize = null,
    fn decode(ctx: *anyopaque, indices: []const u32, sink: *video.Sink) anyerror!void {
        const f: *Frames = @ptrCast(@alignCast(ctx));
        f.calls += 1;
        f.active += 1;
        defer f.active -= 1;
        f.max_active = @max(f.max_active, f.active);
        f.max_indices = @max(f.max_indices, indices.len);
        var pixels: [53 * 21 * 3]u8 = undefined;
        for (indices, 0..) |index, k| {
            if (f.fail_after == k) return error.RecordedDecodeFailure;
            for (&pixels, 0..) |*v, i| v.* = @intCast((index * 17 + i) % 256);
            if (f.cancel_after == k) f.cancelled = true;
            try sink.take(k, &pixels, 53, 21, 53 * 3);
        }
    }
    fn source(f: *Frames, frames: u64) video.Source {
        return .{ .identity = 99, .frames = frames, .rate_num = 2, .rate_den = 1, .width = 53, .height = 21, .ctx = f, .decode = decode };
    }
    fn cancel(ctx: ?*anyopaque) bool {
        const f: *Frames = @ptrCast(@alignCast(ctx.?));
        return f.cancelled;
    }
};
fn pairImage(pair: video.Pair) Image {
    const d = pair.deferred.?;
    return .{ .pixels = pair.pixels, .deferred = .{ .ctx = d.ctx, .prepare = d.prepare, .checked = d.checked }, .gh = pair.gh, .gw = pair.gw, .at = 0, .tokens = pair.tokens, .hash = pair.hash };
}

test "media lifecycle: native lazy video groups exactly match eager pixels including an odd final frame" {
    var plan_arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer plan_arena.deinit();
    const a = plan_arena.allocator();
    var frames: Frames = .{};
    var src = frames.source(5);
    src.rate_num = 1;
    const lazy = try video.prepareLazy(a, src, config, 16384, 128);
    try t.expectEqual(@as(usize, 1), frames.calls);
    try t.expectEqual(@as(usize, 3), lazy.groups.len);
    const eager = try video.prepare(a, src, config, 16384, 128);
    frames = .{};
    for (lazy.groups, eager.groups) |l, e| {
        try t.expectEqual(@as(usize, 0), l.pixels.len);
        try t.expectEqual(e.tokens, l.tokens);
        try t.expectEqual(e.gh, l.gh);
        try t.expectEqual(e.gw, l.gw);
        var scratch: std.heap.ArenaAllocator = .init(t.allocator);
        defer scratch.deinit();
        const pixels = try pairImage(l).materialize(scratch.allocator());
        try t.expectEqualSlices(f32, e.pixels, pixels);
    }
    try t.expectEqualSlices(f64, eager.times, lazy.times);
    try t.expectEqual(@as(usize, 0), frames.calls);
    try t.expectEqual(@as(usize, 0), frames.max_indices);
    try t.expectEqual(@as(usize, 0), frames.max_active);
    try t.expectEqual(@as(usize, 0), frames.active);
    try t.expect(lazy.groups[0].hash != lazy.groups[1].hash);
    const replay = try video.prepareLazy(a, src, config, 16384, 128);
    for (lazy.groups, replay.groups) |l, rr| try t.expectEqual(l.hash, rr.hash);
    var different = src;
    different.identity = 100;
    try t.expect(lazy.groups[0].hash != (try video.prepareLazy(a, different, config, 16384, 128)).groups[0].hash);
}

test "media lifecycle: 128-frame video decodes once before admission; active pairs support cancellation retry and deadline" {
    var plan_arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer plan_arena.deinit();
    var frames: Frames = .{};
    const lazy = try video.prepareLazy(plan_arena.allocator(), frames.source(128), config, 16384, 128);
    try t.expectEqual(@as(usize, 64), lazy.groups.len);
    var cpu: Counted = .{};
    var max_payload: usize = 0;
    for (lazy.groups) |pair| {
        var scratch: std.heap.ArenaAllocator = .init(cpu.allocator());
        {
            const pixels = try pairImage(pair).materialize(scratch.allocator());
            max_payload = @max(max_payload, pixels.len * 4);
        }
        scratch.deinit();
        try t.expectEqual(@as(usize, 0), cpu.live);
    }
    try t.expectEqual(@as(usize, 1), frames.calls);
    try t.expectEqual(@as(usize, 128), frames.max_indices);
    try t.expect(cpu.peak < 4 * max_payload);
    std.debug.print("media receipt: video groups={d} decoder inputs peak={d} CPU arena peak={d} group payload={d}\n", .{ lazy.groups.len, frames.max_indices, cpu.peak, max_payload });
    // Decoder failures are preparation failures, even when only the last selected frame is invalid.
    frames = .{ .fail_after = 127 };
    try t.expectError(error.RecordedDecodeFailure, video.prepareLazy(plan_arena.allocator(), frames.source(128), config, 16384, 128));
    try t.expectEqual(@as(usize, 1), frames.calls);
    try t.expectEqual(@as(usize, 0), frames.active);
    for (0..2) |mode| {
        var scratch: std.heap.ArenaAllocator = .init(cpu.allocator());
        frames = .{ .cancelled = mode == 0 };
        var img = pairImage(lazy.groups[0]);
        if (mode == 1) img.deadline = .{ .io = t.io, .at = .{ .nanoseconds = 0 } };
        try t.expectError(if (mode == 0) error.Cancelled else error.MediaDeadline, img.materializeChecked(scratch.allocator(), .{ .ptr = &frames, .check = Frames.cancel }));
        try t.expectEqual(@as(usize, 0), frames.calls);
        scratch.deinit();
        try t.expectEqual(@as(usize, 0), cpu.live);
    }
    frames = .{};
    var scratch: std.heap.ArenaAllocator = .init(cpu.allocator());
    _ = try pairImage(lazy.groups[0]).materialize(scratch.allocator());
    try t.expectEqual(@as(usize, 0), frames.calls);
    scratch.deinit();
    try t.expectEqual(@as(usize, 0), cpu.live);
}

test "media lifecycle: four streams retain all 50 images and four clips without per-stream CUDA allocations" {
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, t.allocator, &map);
    defer cleanup(&v);
    var sources: [4]Source = .{ .{ .id = 1 }, .{ .id = 2 }, .{ .id = 3 }, .{ .id = 4 } };
    var all: [4][54]Image = undefined;
    for (&sources, &all, 0..) |*source, *imgs, slot| {
        for (imgs[0..50], 0..) |*img, i| img.* = source.image(@intCast(i * 2048), 2048);
        for (imgs[50..], 0..) |*img, i| img.* = source.image(@intCast(102400 + i * 8192), 8192);
        try v.begin(@intCast(slot + 1), imgs, 0);
    }
    try t.expectEqual(@as(usize, 0), Fake.encoded);
    for (0..4) |slot| {
        try v.activate(@intCast(slot + 1));
        try t.expectEqual(@as(usize, 54), v.source.images.len);
        try v.windowWith(131072, 135168, null, {}, Fake.encode);
        try t.expectEqual(@as(usize, 4096), v.rowCount());
        try t.expectEqual(@as(u32, @intCast((slot + 1) * 100000 + 8191)), Fake.values[4095]);
        v.finishWindow();
    }
    try t.expectEqual(@as(usize, 1), Fake.driver_allocs);
    try t.expectEqual(@as(usize, 192 << 20), Fake.driver_peak);
    for (0..4) |slot| v.release(@intCast(slot + 1));
    try t.expectEqual(@as(usize, 0), v.source.images.len);
    try t.expectEqual(@as(u32, 0), v.saved.count());
}

fn failingVideoPlan(allocator: std.mem.Allocator) !void {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    var frames: Frames = .{};
    _ = try video.prepareLazy(arena.allocator(), frames.source(128), config, 16384, 128);
    try t.expectEqual(@as(usize, 1), frames.calls);
}
test "media lifecycle: every lazy plan allocation failure releases the request arena before retry" {
    try std.testing.checkAllAllocationFailures(t.allocator, failingVideoPlan, .{});
}

test "media lifecycle: failed activation keeps source ownership and failed release can forget its borrow" {
    var cpu: Counted = .{};
    var d: cuda.Driver = undefined;
    var v: Images = undefined;
    var map: [media.step_rows / 2 + 1]i32 = undefined;
    try setup(&v, &d, cpu.allocator(), &map);
    defer cleanup(&v);
    var source: Source = .{ .id = 1 };
    const imgs = [_]Image{source.image(0, 1)};
    // Begin on slot zero needs no saved map allocation; switching slots must reserve before moving ownership.
    try v.begin(0, &imgs, 0);
    cpu.fail_at = cpu.allocations;
    try t.expectError(error.OutOfMemory, v.activate(1));
    try t.expectEqual(@as(u32, 0), v.slot);
    try t.expectEqual(@as(usize, 1), v.source.images.len);
    cpu.fail_at = std.math.maxInt(usize);
    try v.activate(1);
    v.forget(0);
    try v.activate(0);
    try t.expectEqual(@as(usize, 0), v.source.images.len);
    try v.begin(0, &imgs, 0);
    v.forget(0);
    try t.expectEqual(@as(usize, 0), v.source.images.len);
}
