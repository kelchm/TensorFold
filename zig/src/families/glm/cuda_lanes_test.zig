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

test "draftBlock folds per-stream keeps into Kind.block, applies on follower, and validates before send" {
    const a = std.testing.allocator;
    const be = lanes.backend;
    const Seq = @import("cuda_engine_types.zig").Seq;
    const proposals = @import("dflash_proposals.zig");
    const h_put: cuda.abi.Function = @ptrFromInt(1);
    const h_conv: cuda.abi.Function = @ptrFromInt(2);
    const h_replay: cuda.abi.Function = @ptrFromInt(3);
    const Wire = struct {
        var sent: [4]follow.Header = undefined;
        var payload: [4][48]u32 = undefined;
        var steps: usize = 0;
        var conv_calls: usize = 0;
        var fail_conv_at: usize = 0;
        var memset_calls: usize = 0;
        var fail_memset_at: usize = 0;
        fn reset() void {
            steps = 0;
            conv_calls = 0;
            fail_conv_at = 0;
            memset_calls = 0;
            fail_memset_at = 0;
        }
        fn launch(func: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            if (func == h_put) {
                const w: *align(1) const follow.Words = @ptrCast(params.?[2].?);
                if (steps < sent.len) {
                    @memcpy(std.mem.asBytes(&sent[steps]), std.mem.sliceAsBytes(w.w[0..16]));
                    const n = @min(@as(usize, sent[steps].n), payload[steps].len);
                    @memcpy(payload[steps][0..n], w.w[16..][0..n]);
                }
                steps += 1;
                return 0;
            }
            if (func == h_conv) {
                conv_calls += 1;
                if (fail_conv_at != 0 and conv_calls == fail_conv_at) return cuda.abi.error_not_ready;
            }
            return 0;
        }
        fn memset32(_: u64, _: c_uint, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            memset_calls += 1;
            if (fail_memset_at != 0 and memset_calls == fail_memset_at) return cuda.abi.error_not_ready;
            return 0;
        }
        fn hostAlloc(_: *?*anyopaque, _: usize, _: c_uint) callconv(.c) cuda.abi.Result {
            return cuda.abi.error_not_ready;
        }
        fn broadcast(_: cuda.abi.DevicePtr, _: cuda.abi.DevicePtr, _: usize, _: cuda.nccl.DataType, _: c_int, _: cuda.nccl.Comm, _: cuda.abi.Stream) callconv(.c) cuda.nccl.Result {
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Wire.launch;
    d.api.cuLaunchKernelEx = @import("cuda_pdl.zig").forwardEx(Wire.launch);
    d.api.cuMemsetD32Async = Wire.memset32;
    d.api.cuMemHostAlloc = Wire.hostAlloc;
    var lib: cuda.nccl.Library = .{ .lib = undefined, .api = undefined };
    lib.api.ncclBroadcast = Wire.broadcast;
    var cur_bits = [3][2]u1{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
    var fcur_bits = [3][2]u1{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
    var seqs = [3]Seq{ .{ .st = undefined }, .{ .st = undefined }, .{ .st = undefined } };
    var fseqs = [3]Seq{ .{ .st = undefined }, .{ .st = undefined }, .{ .st = undefined } };
    for (&seqs, &cur_bits, &fseqs, &fcur_bits) |*q, *cb, *fq, *fcb| {
        for ([_]*Seq{ q, fq }, [_]*[2]u1{ cb, fcb }) |s, bits| {
            s.st.arena.buf = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true };
            s.st.kda_layers = 2;
            s.st.cur = bits;
            s.st.conv = 0x1000;
            s.st.proj = 0x2000;
            s.st.proj_layer = 64;
            s.st.proj_row = 16;
            s.st.conv_c = 256;
            s.st.heads = 1;
            s.st.rec = 0x3000;
            s.st.rec_layer = 64;
            s.st.save_k = 0x4000;
            s.st.save_v = 0x5000;
            s.st.save_g = 0x6000;
            s.st.save_b = 0x7000;
            s.st.pos_dev = 0;
            s.st.prompt = false;
            s.st.pos = 0;
            s.st.pending = 0;
        }
    }
    const makeDrafter = struct {
        fn init(drv: *const cuda.Driver, slots_buf: *[3]dflash.Slot) dflash.Drafter {
            var dr: dflash.Drafter = undefined;
            dr.d = drv;
            dr.gpa = a;
            dr.prof = null;
            dr.verifies = 1;
            dr.lost = 0;
            dr.calls = 0;
            dr.log = 0;
            dr.ids = 0;
            dr.slots = .initBuffer(slots_buf);
            for (0..3) |_| dr.slots.appendAssumeCapacity(.{ .ring = .{ .d = drv, .ptr = 0x2000, .len = 16, .borrowed = true }, .v_step = 0 });
            return dr;
        }
    }.init;
    var dr_slots: [3]dflash.Slot = undefined;
    var fdr_slots: [3]dflash.Slot = undefined;
    var dr = makeDrafter(&d, &dr_slots);
    var fdr = makeDrafter(&d, &fdr_slots);
    var e: engine.Engine = undefined;
    var fe: engine.Engine = undefined;
    for ([_]*engine.Engine{ &e, &fe }, [_]*[3]Seq{ &seqs, &fseqs }, [_]*dflash.Drafter{ &dr, &fdr }) |eng, sq, drafter| {
        eng.gpa = a;
        eng.d = &d;
        eng.c.vocab = dflash.VOCAB;
        eng.stream = .{ .d = &d, .handle = null };
        eng.bd.prefill = false;
        eng.m.index = null;
        eng.m.tiled = false;
        eng.m.packs = null;
        eng.m.dsel = false;
        eng.m.dscore = null;
        eng.m.mla = null;
        eng.m.tfq3 = null;
        eng.m.kda.pdl = false;
        eng.m.kda.replay_layers = .{ .d = &d, .handle = h_replay };
        eng.m.glue.put_words = .{ .d = &d, .handle = h_put };
        eng.m.glue.conv_shift = .{ .d = &d, .handle = h_conv };
        eng.m.priced_windows = true;
        eng.local = &.{};
        eng.images = null;
        eng.group = null;
        eng.lib = &lib;
        eng.comm = null;
        eng.collective = undefined;
        eng.settings = .{};
        eng.pair = null;
        eng.l2pf = null;
        eng.dropped = .empty;
        eng.seqs = .empty;
        try eng.seqs.appendSlice(a, &.{ &sq[0], &sq[1], &sq[2] });
        eng.cur = &sq[0];
        eng.slot = 0;
        eng.drafter = drafter;
    }
    defer e.seqs.deinit(a);
    defer fe.seqs.deinit(a);
    var buf: [64 + 64 * 4]u8 align(8) = undefined;
    var ids: [64]u32 = undefined;
    var link: follow.Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = e.stream, .buf = .{ .d = &d, .ptr = @intFromPtr(&buf), .len = buf.len, .borrowed = true }, .ids = &ids, .gpa = a, .wait = false };
    var self: Lanes = .{ .x = .{ .e = &e, .link = &link } };
    defer self.deinit();
    const prompt = [_]u32{ 10, 20, 30, 40 };
    var s0 = try lanes.Stream.init(a, .{ .id = "s0", .prompt = &prompt, .max_new = 8, .drafts = true });
    defer s0.deinit(a);
    var s1 = try lanes.Stream.init(a, .{ .id = "s1", .prompt = &prompt, .max_new = 8, .drafts = true });
    defer s1.deinit(a);
    var s2 = try lanes.Stream.init(a, .{ .id = "s2", .prompt = &prompt, .max_new = 8, .drafts = true });
    defer s2.deinit(a);
    var stray = try lanes.Stream.init(a, .{ .id = "stray", .prompt = &prompt, .max_new = 8, .drafts = true });
    defer stray.deinit(a);
    try self.lanes.put(a, &s0, .{ .slot = 0, .pending_rows = 4 });
    try self.lanes.put(a, &s1, .{ .slot = 1, .pending_rows = null });
    try self.lanes.put(a, &s2, .{ .slot = 2, .pending_rows = 5 });
    const rows3 = [_]u32{ 0, 1, 2 };
    const rows2 = [_]u32{ 0, 1 };
    const rows4 = [_]u32{ 0, 1, 2, 3 };
    const follow_ok = [_]u32{ 11, 22, 33, 44 };
    const follow_oov = [_]u32{ 11, 22, dflash.VOCAB, dflash.VOCAB };
    const backend = self.backend();
    // 1) Batched 3-stream draftBlock (depth=0) sends ONE Kind.block with commit_mask=0b101, ZERO Kind.keep.
    seqs[0].st.pos = 10;
    seqs[0].st.pending = 4;
    seqs[1].st.pos = 20;
    seqs[1].st.pending = 0;
    seqs[2].st.pos = 30;
    seqs[2].st.pending = 5;
    fseqs[0].st.pos = 10;
    fseqs[0].st.pending = 4;
    fseqs[1].st.pos = 20;
    fseqs[1].st.pending = 0;
    fseqs[2].st.pos = 30;
    fseqs[2].st.pending = 5;
    const reqs_zero = [_]be.DraftRequest{
        .{ .stream = &s0, .follow = follow_ok[0..3], .rows = &rows3, .start = 10, .position = 14, .depth = 0 },
        .{ .stream = &s1, .follow = follow_ok[0..2], .rows = &rows2, .start = 20, .position = 21, .depth = 0 },
        .{ .stream = &s2, .follow = follow_ok[0..4], .rows = &rows4, .start = 30, .position = 35, .depth = 0 },
    };
    Wire.reset();
    try backend.vtable.draft(backend.ptr, &reqs_zero);
    try std.testing.expectEqual(@as(usize, 1), Wire.steps);
    try std.testing.expectEqual(@backingInt(follow.Kind.block), Wire.sent[0].kind);
    try std.testing.expectEqual(@as(u32, 3), Wire.sent[0].depth);
    try std.testing.expectEqual(@as(u32, 0b101), Wire.sent[0].flags >> 16);
    try std.testing.expectEqual(@as(u32, 0), Wire.sent[0].flags & 0xffff);
    try std.testing.expectEqual(@as(usize, 13), seqs[0].st.pos);
    try std.testing.expectEqual(@as(usize, 0), seqs[0].st.pending);
    try std.testing.expectEqual([2]u1{ 1, 1 }, cur_bits[0]);
    try std.testing.expectEqual(@as(usize, 20), seqs[1].st.pos);
    try std.testing.expectEqual([2]u1{ 0, 0 }, cur_bits[1]);
    try std.testing.expectEqual(@as(usize, 34), seqs[2].st.pos);
    try std.testing.expectEqual(@as(usize, 0), seqs[2].st.pending);
    try std.testing.expectEqual([2]u1{ 1, 1 }, cur_bits[2]);
    try std.testing.expect(self.lanes.get(&s0).?.pending_rows == null);
    try std.testing.expect(self.lanes.get(&s2).?.pending_rows == null);
    // 2) Follower applies the exact wire packet: commits only slots 0 and 2 with q.n before draftBlock.
    try proposals.applyBlock(&fe, Wire.sent[0].depth, Wire.sent[0].flags, Wire.payload[0][0..Wire.sent[0].n]);
    try std.testing.expectEqual(@as(usize, 13), fseqs[0].st.pos);
    try std.testing.expectEqual(@as(usize, 0), fseqs[0].st.pending);
    try std.testing.expectEqual([2]u1{ 1, 1 }, fcur_bits[0]);
    try std.testing.expectEqual(@as(usize, 20), fseqs[1].st.pos);
    try std.testing.expectEqual(@as(usize, 34), fseqs[2].st.pos);
    try std.testing.expectEqual(@as(usize, 0), fseqs[2].st.pending);
    try std.testing.expectEqual([2]u1{ 1, 1 }, fcur_bits[2]);
    try std.testing.expectError(error.BadStep, proposals.applyBlock(&fe, 3, @as(u32, 1) << 19, Wire.payload[0][0..9]));
    try std.testing.expectError(error.SlotTwice, proposals.applyBlock(&fe, 2, 0, &.{ 1, 0, 11, 1, 0, 22 }));
    fseqs[0].st.pending = 4;
    try std.testing.expectError(error.BadTokenId, proposals.applyBlock(&fe, 2, (@as(u32, 1) << 16) | 0b10, &.{ 0, 3, 11, 1, 0, dflash.VOCAB }));
    try std.testing.expectEqual(@as(usize, 4), fseqs[0].st.pending);
    try std.testing.expectEqual(@as(usize, 13), fseqs[0].st.pos);
    // 3) Mixed propose_mask and commit_mask (0b101) pack cleanly; uploadIds failure syncs committed pending_rows.
    seqs[0].st.pending = 4;
    seqs[2].st.pending = 5;
    self.lanes.getPtr(&s0).?.pending_rows = 4;
    self.lanes.getPtr(&s2).?.pending_rows = 5;
    const reqs_propose = [_]be.DraftRequest{
        .{ .stream = &s0, .follow = follow_ok[0..3], .rows = &rows3, .start = 13, .position = 17, .depth = 7 },
        .{ .stream = &s1, .follow = follow_ok[0..2], .rows = &rows2, .start = 20, .position = 21, .depth = 0 },
        .{ .stream = &s2, .follow = follow_ok[0..4], .rows = &rows4, .start = 34, .position = 39, .depth = 7 },
    };
    Wire.reset();
    try std.testing.expectError(error.NotReady, backend.vtable.draft(backend.ptr, &reqs_propose));
    try std.testing.expectEqual(@as(usize, 1), Wire.steps);
    try std.testing.expectEqual(@backingInt(follow.Kind.block), Wire.sent[0].kind);
    try std.testing.expectEqual(@as(u32, 0b101), Wire.sent[0].flags >> 16);
    try std.testing.expectEqual(@as(u32, 0b101), Wire.sent[0].flags & 0xffff);
    try std.testing.expect(self.lanes.get(&s0).?.pending_rows == null);
    try std.testing.expect(self.lanes.get(&s2).?.pending_rows == null);
    // 4) Preflight errors fail BEFORE any wire send (even before Pass-2 rowless bind), leaving state untouched.
    seqs[0].st.pos = 10;
    seqs[0].st.pending = 4;
    seqs[1].st.pos = 20;
    seqs[1].st.pending = 2;
    self.lanes.getPtr(&s0).?.pending_rows = 4;
    self.lanes.getPtr(&s1).?.pending_rows = 2;
    const valid0: be.DraftRequest = .{ .stream = &s0, .follow = follow_ok[0..3], .rows = &rows3, .start = 10, .position = 14, .depth = 7 };
    const rowless0: be.DraftRequest = .{ .stream = &s0, .follow = &.{}, .rows = null, .first = .{ .value = 11 }, .start = 10, .position = 15, .depth = 7 };
    const TreeShape = @typeInfo(@typeInfo(@FieldType(be.DraftRequest, "lanes")).optional.child).pointer.child;
    const bad_tree_shape: TreeShape = undefined;
    const bad_rows = [_]u32{ 0, 2 };
    const failures = [_]struct { first: be.DraftRequest, req: be.DraftRequest, err: anyerror }{
        .{ .first = valid0, .req = .{ .stream = &s1, .follow = follow_ok[0..2], .rows = &rows2, .start = 20, .position = 99, .depth = 7 }, .err = error.PositionMismatch },
        .{ .first = valid0, .req = .{ .stream = &s1, .follow = follow_ok[0..2], .rows = &rows2, .start = 20, .position = 23, .depth = 7, .lanes = &bad_tree_shape }, .err = error.TreesNotBuilt },
        .{ .first = valid0, .req = .{ .stream = &s1, .follow = follow_ok[0..2], .rows = &bad_rows, .start = 20, .position = 23, .depth = 7 }, .err = error.TreesNotBuilt },
        .{ .first = valid0, .req = .{ .stream = &stray, .follow = follow_ok[0..2], .rows = &rows2, .start = 0, .position = 1, .depth = 7 }, .err = error.UnknownStream },
        .{ .first = valid0, .req = .{ .stream = &s1, .follow = follow_ok[0..3], .rows = &rows3, .start = 20, .position = 24, .depth = 7 }, .err = error.BadCommit },
        .{ .first = rowless0, .req = .{ .stream = &s1, .follow = follow_oov[0..3], .rows = &rows3, .start = 20, .position = 24, .depth = 7 }, .err = error.BadTokenId },
        .{ .first = rowless0, .req = .{ .stream = &s0, .follow = follow_ok[0..3], .rows = &rows3, .start = 10, .position = 14, .depth = 7 }, .err = error.SlotTwice },
    };
    for (failures) |case| {
        Wire.reset();
        try std.testing.expectError(case.err, backend.vtable.draft(backend.ptr, &.{ case.first, case.req }));
        try std.testing.expectEqual(@as(usize, 0), Wire.steps);
        try std.testing.expectEqual(@as(?usize, 4), self.lanes.get(&s0).?.pending_rows);
        try std.testing.expectEqual(@as(?usize, 2), self.lanes.get(&s1).?.pending_rows);
        try std.testing.expectEqual(@as(usize, 10), seqs[0].st.pos);
        try std.testing.expectEqual(@as(usize, 4), seqs[0].st.pending);
        try std.testing.expectEqual(@as(usize, 20), seqs[1].st.pos);
        try std.testing.expectEqual(@as(usize, 2), seqs[1].st.pending);
    }
    // Duplicate stream with pending_rows == null also returns SlotTwice with zero wire sends.
    seqs[1].st.pending = 0;
    self.lanes.getPtr(&s1).?.pending_rows = null;
    const dup_clean: be.DraftRequest = .{ .stream = &s1, .follow = follow_ok[0..2], .rows = &rows2, .start = 20, .position = 21, .depth = 7 };
    Wire.reset();
    try std.testing.expectError(error.SlotTwice, backend.vtable.draft(backend.ptr, &.{ dup_clean, dup_clean }));
    try std.testing.expectEqual(@as(usize, 0), Wire.steps);
    // Direct Leader.blockKeeping preflight rejects BadCommit, SlotTwice (commit_mask=0), and OOV active anchor.
    Wire.reset();
    var bad_commit = [_]dflash.Req{.{ .slot = 0, .pos = 0, .n = 5, .anchor = 11 }};
    try std.testing.expectError(error.BadCommit, self.x.blockKeeping(&bad_commit, 1));
    var bad_dup = [_]dflash.Req{ .{ .slot = 1, .pos = 0, .n = 0, .anchor = 11 }, .{ .slot = 1, .pos = 0, .n = 0, .anchor = 22 } };
    try std.testing.expectError(error.SlotTwice, self.x.blockKeeping(&bad_dup, 0));
    var bad_oov = [_]dflash.Req{.{ .slot = 1, .pos = 0, .n = 0, .anchor = dflash.VOCAB, .propose = true }};
    try std.testing.expectError(error.BadTokenId, self.x.blockKeeping(&bad_oov, 0));
    try std.testing.expectEqual(@as(usize, 0), Wire.steps);
    // 5) Forward.commit / State.setPos leave cur, pos, pending, and pending_rows intact on convShift or setPos failure.
    for ([_][2]usize{ .{ 2, 0 }, .{ 0, 2 } }) |fail_at| {
        cur_bits = .{ .{ 0, 0 }, .{ 0, 0 }, .{ 0, 0 } };
        seqs[0].st.pos = 10;
        seqs[0].st.pending = 4;
        seqs[1].st.pos = 20;
        seqs[1].st.pending = 0;
        seqs[2].st.pos = 30;
        seqs[2].st.pending = 5;
        self.lanes.getPtr(&s0).?.pending_rows = 4;
        self.lanes.getPtr(&s1).?.pending_rows = null;
        self.lanes.getPtr(&s2).?.pending_rows = 5;
        Wire.reset();
        Wire.fail_conv_at = fail_at[0];
        Wire.fail_memset_at = fail_at[1];
        try std.testing.expectError(error.NotReady, backend.vtable.draft(backend.ptr, &reqs_zero));
        try std.testing.expectEqual(@as(usize, 13), seqs[0].st.pos);
        try std.testing.expectEqual(@as(usize, 0), seqs[0].st.pending);
        try std.testing.expectEqual([2]u1{ 1, 1 }, cur_bits[0]);
        try std.testing.expect(self.lanes.get(&s0).?.pending_rows == null);
        try std.testing.expectEqual(@as(usize, 30), seqs[2].st.pos);
        try std.testing.expectEqual(@as(usize, 5), seqs[2].st.pending);
        try std.testing.expectEqual([2]u1{ 0, 0 }, cur_bits[2]);
        try std.testing.expectEqual(@as(?usize, 5), self.lanes.get(&s2).?.pending_rows);
    }
}
