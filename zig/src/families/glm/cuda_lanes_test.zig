//! The lane backend's slots: chosen before a stream's map entry exists, and none left behind by a failed insertion.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const dflash = @import("dflash.zig");
const Lanes = @import("cuda_lanes.zig").Lanes;
const fw = @import("cuda_forward.zig");
const solo = @import("cuda_prompt_solo.zig");
const prompt_cuts = @import("cuda_prompt_cuts.zig");

test "a lane's slot is chosen before its map entry exists" {
    var storage: [8192]u8 = @splat(0);
    var fixed = std.heap.FixedBufferAllocator.init(&storage);
    var e: engine.Engine = undefined;
    e.gpa = fixed.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const Probe = struct {
        var bad = false;
        fn choose(l: *const Lanes) u32 {
            bad = l.lanes.count() != 0;
            return 0;
        }
    };
    _ = try self.claimUsing(&streams[0], Probe.choose);
    try std.testing.expect(!Probe.bad);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[0])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
    _ = self.lanes.remove(&streams[0]);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&streams[2])).slot);
    try std.testing.expectEqual(@as(u32, 1), (try self.claim(&streams[1])).slot);
}

test "a failed lane insertion leaves no entry, and a retry takes slot zero" {
    var fail = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var e: engine.Engine = undefined;
    e.gpa = fail.allocator();
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var stream: lanes.Stream = undefined;
    try std.testing.expectError(error.OutOfMemory, self.claim(&stream));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    fail.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u32, 0), (try self.claim(&stream)).slot);
}

test "a prompt pass goes solo only while no other stream is past its prompt pass" {
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    defer self.lanes.deinit(e.gpa);
    var streams: [3]lanes.Stream = undefined;
    const first = try self.claim(&streams[0]); // a stream's own lane, fresh as its first chunk starts
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
    first.fill = fw.prefill_rows;
    (try self.claim(&streams[1])).fill = 2 * fw.prefill_rows; // another prompt pass, partway
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
    _ = try self.claim(&streams[2]); // past its prompt pass: decoding
    try std.testing.expect(solo.othersDecode(&self.lanes, &streams[0]) and solo.othersDecode(&self.lanes, &streams[1]));
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[2])); // the other two still fill
    _ = self.lanes.remove(&streams[2]); // released
    try std.testing.expect(!solo.othersDecode(&self.lanes, &streams[0]));
}

test "full GLM-5.3 declares chunk_invariant, so --decode-share's cap reaches its prompt steps; GLM-5.3-Flash does not" {
    var e: engine.Engine = undefined;
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    try std.testing.expect(!self.backend().chunk_invariant); // the lanes' own: GLM-5.3-Flash steps whole cold chunks
    try std.testing.expect(!prompt_cuts.declared(self.backend(), false).chunk_invariant);
    try std.testing.expect(prompt_cuts.declared(self.backend(), true).chunk_invariant); // cuda_native passes Config.dsa
}

test "glm request starts: a kept state off the new request's cold cuts, or over other chunks below, passes from 0" {
    const m = fw.prefill_rows;
    var e: engine.Engine = undefined;
    e.gpa = std.testing.allocator;
    e.snaps = .empty;
    defer e.snaps.deinit(e.gpa);
    try e.snaps.put(e.gpa, 7, undefined); // every rank holds state 7; resumeFrom reads no more of it
    var self: Lanes = .{ .x = .{ .e = &e, .link = undefined } };
    const Case = struct { kept: []const u32, cuts: []const u32, why: ?anyerror, id: u64 = 7, at: u32 = 2 * m };
    const cases = [_]Case{
        .{ .kept = &.{}, .cuts = &.{m + 300}, .why = error.SnapshotOutOfStep }, // 2m lies inside a chunk now
        .{ .kept = &.{300}, .cuts = &.{}, .why = error.SnapshotOutOfStep, .at = 300 }, // a chat's start, a grid pass
        .{ .kept = &.{300}, .cuts = &.{ 300, 2 * m }, .why = null, .at = 300 }, // the same start: it resumes
        .{ .kept = &.{}, .cuts = &.{ 300, 2 * m }, .why = error.SnapshotOutOfStep }, // a grid state, a start below
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 700, 2 * m }, .why = error.SnapshotOutOfStep }, // other starts below
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{}, .why = error.SnapshotOutOfStep }, // a chat's state, a grid pass
        .{ .kept = &.{}, .cuts = &.{ 2 * m, 2 * m + 500 }, .why = null }, // the starts from 2m on chunk below it alike
        .{ .kept = &.{m}, .cuts = &.{}, .why = null }, // a start on the grid chunks as the grid does
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 300, 2 * m, 3 * m }, .why = null },
        .{ .kept = &.{ 300, 2 * m }, .cuts = &.{ 300, 2 * m }, .why = error.SnapshotStale, .id = 8 }, // no rank holds 8
    };
    for (cases) |c| {
        var k: engine.Kept = .{ .id = c.id, .at = c.at, .below = prompt_cuts.below(c.kept, c.at) };
        var s: lanes.Stream = undefined;
        s.chunks = c.cuts;
        s.reuse = .{ .saved = &k, .at = c.at };
        s.reuse_failed = false;
        s.reuse_error = null;
        s.cached = 0;
        const from = self.resumeFrom(&s, 4 * m);
        try std.testing.expectEqual(c.why, s.reuse_error);
        try std.testing.expectEqual(c.why != null, s.reuse_failed);
        try std.testing.expectEqual(@as(usize, if (c.why == null) c.at else 0), if (from) |r| r.at else 0);
        try std.testing.expectEqual(if (c.why == null) c.at else 0, s.cached); // refused: the pass starts at 0
    }
}

test "multi_prefill and opening roll back announced slots across ranks when begin or cuts fails" {
    const a = std.testing.allocator;
    const Wire = struct {
        var sent: [8]follow.Header = undefined;
        var steps: usize = 0;
        fn reset() void {
            steps = 0;
        }
        fn put(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            const w: *align(1) const follow.Words = @ptrCast(params.?[2].?);
            if (steps < sent.len) @memcpy(std.mem.asBytes(&sent[steps]), std.mem.sliceAsBytes(w.w[0..16]));
            steps += 1;
            return 0;
        }
        fn memset8(_: u64, _: u8, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn memset32(_: u64, _: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn htod(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return 0;
        }
        fn broadcast(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            return 0;
        }
    };
    Wire.reset();
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Wire.put;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Wire.put);
    d.api.cuMemsetD8Async = Wire.memset8;
    d.api.cuMemsetD32Async = Wire.memset32;
    d.api.cuMemcpyHtoD_v2 = Wire.htod;
    d.api.cuStreamSynchronize = Wire.sync;
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Wire.broadcast;
    var cur_bits = [_]u1{0};
    var rings_buf = [_]u64{ 0, 0 };
    const Seq = @import("cuda_engine_types.zig").Seq;
    var seqs = [2]Seq{ .{ .st = undefined }, .{ .st = undefined } };
    for (&seqs) |*q| {
        q.st.arena.buf = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true };
        q.st.conv = 0;
        q.st.rec = 0;
        q.st.kda_layers = 0;
        q.st.conv_c = 0;
        q.st.rec_layer = 0;
        q.st.cur = &cur_bits;
        q.st.caches = &.{};
        q.st.rings = &rings_buf;
        q.st.ring_row = 256;
        q.st.head = false;
        q.st.pos_dev = 0;
        q.st.mtp_pos_dev = 0;
        q.st.capacity = 4096;
        q.cuts = .empty;
    }
    defer for (&seqs) |*q| q.cuts.deinit(a);
    var e: engine.Engine = undefined;
    e.gpa = a;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.m.glue.put_words = .{ .d = &d, .handle = null };
    e.m.tiled = true;
    e.capacity = 4096;
    e.pages = null;
    e.images = null;
    e.local = &.{};
    e.keep_seqs = true;
    e.dropped = .empty;
    e.seqs = .empty;
    defer e.seqs.deinit(a);
    try e.seqs.appendSlice(a, &.{ &seqs[0], &seqs[1] });
    e.cur = &seqs[0];
    e.slot = 0;
    var fail = std.testing.FailingAllocator.init(a, .{ .fail_index = 1 });
    var dr: dflash.Drafter = undefined;
    dr.d = &d;
    dr.gpa = fail.allocator();
    dr.slots = .empty;
    defer dr.slots.deinit(dr.gpa);
    try dr.slots.ensureTotalCapacityPrecise(dr.gpa, 1);
    dr.slots.appendAssumeCapacity(.{ .ring = .{ .d = &d, .ptr = 0x2000, .len = 16, .borrowed = true } });
    e.drafter = &dr;
    var buf: [64 + 64 * 4]u8 align(8) = undefined;
    var ids: [64]u32 = undefined;
    var link: follow.Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = e.stream, .buf = .{ .d = &d, .ptr = @intFromPtr(&buf), .len = buf.len, .borrowed = true }, .ids = &ids, .gpa = a, .wait = false };
    var self: Lanes = .{ .x = .{ .e = &e, .link = &link }, .whole = false, .multi_prefill = true };
    defer self.deinit();
    const prompt = [_]u32{ 10, 20, 30, 40 };
    var s0 = try lanes.Stream.init(a, .{ .id = "s0", .prompt = &prompt, .max_new = 4, .drafts = false });
    defer s0.deinit(a);
    var s1 = try lanes.Stream.init(a, .{ .id = "s1", .prompt = &prompt, .max_new = 4, .drafts = false });
    defer s1.deinit(a);
    // Slot 1 fails locally in e.begin (DFlash slot grow OOM) before send: slot 1 stays off the wire, slot 0 releases.
    const be = self.backend();
    try std.testing.expectError(error.OutOfMemory, be.vtable.prefill_many.?(be.ptr, &.{ &s0, &s1 }));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    try std.testing.expectEqual(@as(usize, 2), Wire.steps);
    try std.testing.expectEqual(@backingInt(follow.Kind.begin), Wire.sent[0].kind);
    try std.testing.expectEqual(@as(u32, 0), Wire.sent[0].slot);
    try std.testing.expectEqual(@backingInt(follow.Kind.release), Wire.sent[1].kind);
    try std.testing.expectEqual(@as(u32, 0), Wire.sent[1].slot);
    // 2) In opening, if cuts() fails after beginOn succeeded, followers still receive release on that slot.
    Wire.reset();
    e.drafter = null;
    var bad_cuts = try lanes.Stream.init(a, .{ .id = "bad-cuts", .prompt = &prompt, .max_new = 4, .drafts = false, .chunks = &.{0} });
    defer bad_cuts.deinit(a);
    try std.testing.expectError(error.BadCuts, self.prefillStep(&bad_cuts, 0));
    try std.testing.expectEqual(@as(usize, 0), self.lanes.count());
    try std.testing.expectEqual(@as(usize, 2), Wire.steps);
    try std.testing.expectEqual(@backingInt(follow.Kind.begin), Wire.sent[0].kind);
    try std.testing.expectEqual(@as(u32, 0), Wire.sent[0].slot);
    try std.testing.expectEqual(@backingInt(follow.Kind.release), Wire.sent[1].kind);
    try std.testing.expectEqual(@as(u32, 0), Wire.sent[1].slot);
    // Subsequent slot 0 reuse after rollback announces cleanly.
    Wire.reset();
    try self.x.bind(0);
    try self.x.beginOn(null, &.{}, 0);
    try std.testing.expectEqual(@as(usize, 1), Wire.steps);
    try std.testing.expectEqual(@backingInt(follow.Kind.begin), Wire.sent[0].kind);
    try std.testing.expectEqual(@as(u32, 0), Wire.sent[0].slot);
}
