//! TF_GLM_REQUEST_STARTS=marks: mark-driven cuts, wide/paired solo on 262K chats, resume, rewind, and cancel.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");
const glm_env = @import("cuda_env.zig");
const glm_lanes = @import("cuda_lanes.zig");
const prompt_cuts = @import("cuda_prompt_cuts.zig");
const prompt_pair = @import("cuda_prompt_pair.zig");
const prompt_solo = @import("cuda_prompt_solo.zig");
const share_mod = @import("cuda_share.zig");
const Seq = @import("cuda_engine_types.zig").Seq;
const t = std.testing;

const Wire = struct {
    var headers: [8]follow.Header = undefined;
    var cuts_buf: [64]u32 = undefined;
    var cuts_len: usize = 0;
    var steps: usize = 0;

    fn reset() void {
        steps = 0;
        cuts_len = 0;
    }

    fn put(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
        const w: *align(1) const follow.Words = @ptrCast(params.?[2].?);
        var h: follow.Header = undefined;
        @memcpy(std.mem.asBytes(&h), std.mem.sliceAsBytes(w.w[0..16]));
        if (steps < headers.len) headers[steps] = h;
        if (h.kind == @backingInt(follow.Kind.cuts)) {
            cuts_len = @min(h.n, cuts_buf.len);
            @memcpy(cuts_buf[0..cuts_len], w.w[16..][0..cuts_len]);
        }
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

test "glm request starts: markCuts deduplicates, sorts, and guarantees coldCut at from for every position" {
    const a = t.allocator;
    const empty = try prompt_cuts.markCuts(a, 0, &.{}, 4096);
    defer a.free(empty);
    try t.expectEqualSlices(u32, &.{}, empty);

    const cold = try prompt_cuts.markCuts(a, 0, &.{ 262_307, 0, 3999, 262_307, 262_309, 300_000 }, 262_309);
    defer a.free(cold);
    try t.expectEqualSlices(u32, &.{ 3999, 262_307 }, cold);

    const resumed = try prompt_cuts.markCuts(a, 262_307, &.{ 3999, 262_307, 262_853, 262_855 }, 262_855);
    defer a.free(resumed);
    try t.expectEqualSlices(u32, &.{ 262_307, 262_853 }, resumed);
    try t.expect(prompt_cuts.coldCut(262_855, resumed, 262_307));
    try t.expect(prompt_cuts.coldCut(262_855, resumed, 262_853));

    const short_turn = try prompt_cuts.markCuts(a, 262_853, &.{262_886}, 262_888);
    defer a.free(short_turn);
    try t.expectEqualSlices(u32, &.{ 262_853, 262_886 }, short_turn);
    try t.expect(prompt_cuts.coldCut(262_888, short_turn, 262_853));
    try t.expectEqual(@as(usize, 262_886), prompt_cuts.nextCut(262_888, short_turn, 262_853));

    // Arbitrary off-grid resume points always satisfy coldCut when markCuts leads with from.
    for ([_]u32{ 1, 33, 512, 2047, 2048, 2049, 3999, 111_949, 262_307 }) |from| {
        const len: usize = @as(usize, from) + 500;
        const cuts = try prompt_cuts.markCuts(a, from, &.{from + 123}, len);
        defer a.free(cuts);
        try t.expect(prompt_cuts.coldCut(len, cuts, from));
        try t.expect(prompt_cuts.coldCut(len, cuts, from + 123));
    }

    try t.expect(!prompt_cuts.usableCut(&.{}, 0));
    try t.expect(prompt_cuts.usableCut(&.{}, fw.prefill_rows));
    try t.expect(!prompt_cuts.usableCut(&.{}, 500));
    try t.expect(prompt_cuts.usableCut(&.{ 500, 262_307 }, 262_307));
    try t.expect(!prompt_cuts.usableCut(&.{ 500, 262_307 }, 2048));
}

test "glm request starts: marks mode sends only from+marks over Kind.cuts, runs 262K paired-solo, and resumes multi-turn + rewind" {
    const a = t.allocator;
    const C = fw.prefill_rows;
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
    var seq: Seq = .{ .st = undefined };
    seq.st.arena.buf = .{ .d = &d, .ptr = 0, .len = 0, .borrowed = true };
    seq.st.conv = 0;
    seq.st.rec = 0;
    seq.st.kda_layers = 0;
    seq.st.conv_c = 0;
    seq.st.rec_layer = 0;
    seq.st.cur = &cur_bits;
    seq.st.caches = &.{};
    seq.st.rings = &rings_buf;
    seq.st.ring_row = 256;
    seq.st.ring_rows = 4 * C;
    seq.st.pool = 4;
    seq.st.base = 0;
    seq.st.head = false;
    seq.st.pos_dev = 0;
    seq.st.mtp_pos_dev = 0;
    seq.st.capacity = 300_000;
    seq.cuts = .empty;
    defer seq.deinitMeta(a);

    var e: engine.Engine = undefined;
    e.gpa = a;
    e.d = &d;
    e.stream = .{ .d = &d, .handle = null };
    e.m.glue.put_words = .{ .d = &d, .handle = null };
    e.m.tiled = true;
    e.capacity = 300_000;
    e.pages = null;
    e.images = null;
    e.drafter = null;
    e.local = &.{};
    e.keep_seqs = true;
    e.prompt_logprobs = null;
    e.prompt_profs = @splat(null);
    e.settings = .{};
    e.pair = .{ .bp = undefined, .pieces = undefined, .pos = undefined, .rows = 2 * C };
    e.bp.rows = 2 * C;
    e.pieces = undefined;
    e.pieces.count = 3;
    e.share.world = 2;
    e.dropped = .empty;
    e.seqs = .empty;
    defer e.seqs.deinit(a);
    try e.seqs.append(a, &seq);
    e.snaps = .empty;
    defer e.snaps.deinit(a);
    try e.snaps.put(a, 11, undefined);
    try e.snaps.put(a, 12, undefined);
    try e.snaps.put(a, 13, undefined);
    e.cur = &seq;
    e.slot = 0;

    var buf: [64 + 64 * 4]u8 align(8) = undefined;
    var link_ids: [64]u32 = undefined;
    var link: follow.Link = .{ .d = &d, .lib = &lib, .comm = null, .stream = e.stream, .buf = .{ .d = &d, .ptr = @intFromPtr(&buf), .len = buf.len, .borrowed = true }, .ids = &link_ids, .gpa = a, .wait = false };
    var self: glm_lanes.Lanes = .{ .x = .{ .e = &e, .link = &link }, .request_starts = .marks };
    defer self.deinit();

    // Build a 262,309-token Turn 1 with ~400 unmarked message starts in s.chunks and only 1 mark at 262,307.
    const prompt1 = try a.alloc(u32, 262_309);
    defer a.free(prompt1);
    @memset(prompt1, 7);
    var chunks1: std.ArrayList(u32) = .empty;
    defer chunks1.deinit(a);
    var p: u32 = 650;
    while (p < 262_000) : (p += 650) try chunks1.append(a, p);
    try chunks1.append(a, 262_307);

    var s1 = try lanes.Stream.init(a, .{ .id = "t1", .prompt = prompt1, .max_new = 64, .drafts = false, .chunks = chunks1.items, .reuse = .{ .marks = &.{262_307} } });
    defer s1.deinit(a);
    try t.expectEqual(@as(?engine.Resume, null), try self.opening(&s1));
    // Engine receives ONLY [262_307] over Kind.cuts, NOT the ~400 unmarked message starts!
    try t.expectEqualSlices(u32, &.{262_307}, seq.cuts.items);
    try t.expectEqualSlices(u32, &.{262_307}, Wire.cuts_buf[0..Wire.cuts_len]);

    // Walk full 262,309-token prefill: 32 paired-solo (4096+4096) steps to 262,144, then 163 to mark, then 2.
    const Mark = struct {
        fn call(_: *anyopaque, _: u32) void {}
    };
    var ctx: u8 = 0;
    const m1: engine.Marks = .{ .at = &.{262_307}, .ctx = &ctx, .call = Mark.call };
    seq.st.base = 0;
    var at: usize = 0;
    var paired_steps: usize = 0;
    var saw_mark = false;
    while (at < prompt1.len) {
        if (at == 262_307) saw_mark = true;
        const left = prompt1.len - at;
        const paired = prompt_pair.pairedAt(&e, at, prompt1.len, null, m1);
        const solo = !paired and prompt_solo.at(&e, at, prompt1.len, null, m1);
        const wide: usize = if (paired) 4 * C else if (solo) 2 * C else 0;
        const n = prompt_cuts.leaderRows(prompt1.len, seq.cuts.items, at, wide);
        const got = try prompt_cuts.stepRows(prompt1.len, seq.cuts.items, at, @min(left, n + 1), if (paired) 4 * C else 2 * C);
        try t.expectEqual(@min(left, n), got);
        if (at + got > seq.st.base + seq.st.ring_rows) seq.st.base = at / seq.st.pool * seq.st.pool;
        if (paired) paired_steps += 1;
        at += got;
    }
    try t.expectEqual(@as(usize, 32), paired_steps); // 32 * 8192 = 262,144!
    try t.expect(saw_mark);

    // Turn 2 (262,855 tokens): resumes 262,307 even though k.below (0) != below(chunks2, 262,307).
    const prompt2 = try a.alloc(u32, 262_855);
    defer a.free(prompt2);
    @memset(prompt2, 7);
    var chunks2: std.ArrayList(u32) = .empty;
    defer chunks2.deinit(a);
    try chunks2.appendSlice(a, chunks1.items);
    try chunks2.append(a, 262_853);
    var k1: engine.Kept = .{ .id = 11, .at = 262_307, .below = prompt_cuts.below(&.{262_307}, 262_307) };
    try t.expect(k1.below != prompt_cuts.below(chunks2.items, 262_307));
    var s2 = try lanes.Stream.init(a, .{ .id = "t2", .prompt = prompt2, .max_new = 64, .drafts = false, .chunks = chunks2.items, .reuse = .{ .saved = &k1, .at = 262_307, .marks = &.{262_853} } });
    defer s2.deinit(a);
    // In .all ("1") mode, mismatched historical below is still rejected; in .marks mode it resumes cleanly.
    self.request_starts = .all;
    try t.expectEqual(@as(?engine.Resume, null), self.resumeFrom(&s2, prompt2.len));
    try t.expectEqual(@as(?anyerror, error.SnapshotOutOfStep), s2.reuse_error);
    s2.reuse_failed = false;
    s2.reuse_error = null;
    self.request_starts = .marks;
    _ = self.lanes.remove(&s1);
    Wire.reset();
    const r2 = (try self.opening(&s2)) orelse return error.ExpectedResume;
    try t.expectEqual(@as(usize, 262_307), r2.at);
    try t.expectEqual(@as(u32, 262_307), s2.cached);
    try t.expectEqualSlices(u32, &.{ 262_307, 262_853 }, seq.cuts.items);
    try t.expectEqualSlices(u32, &.{ 262_307, 262_853 }, Wire.cuts_buf[0..Wire.cuts_len]);
    try t.expect(prompt_cuts.coldCut(prompt2.len, seq.cuts.items, r2.at));
    try t.expectEqual(@as(usize, 262_853 - 262_307), try prompt_cuts.stepRows(prompt2.len, seq.cuts.items, r2.at, 547, 4 * C));

    // Turn 3 (short turn < 256 tokens apart: 262,888 tokens, resumes 262,853 with k2.below != 0, marks 262,886).
    const prompt3 = try a.alloc(u32, 262_888);
    defer a.free(prompt3);
    @memset(prompt3, 7);
    var chunks3: std.ArrayList(u32) = .empty;
    defer chunks3.deinit(a);
    try chunks3.appendSlice(a, chunks2.items);
    try chunks3.append(a, 262_886);
    var k2: engine.Kept = .{ .id = 12, .at = 262_853, .below = prompt_cuts.below(&.{ 262_307, 262_853 }, 262_853) };
    try t.expect(k2.below != 0);
    var s3 = try lanes.Stream.init(a, .{ .id = "t3", .prompt = prompt3, .max_new = 64, .drafts = false, .chunks = chunks3.items, .reuse = .{ .saved = &k2, .at = 262_853, .marks = &.{262_886} } });
    defer s3.deinit(a);
    _ = self.lanes.remove(&s2);
    Wire.reset();
    const r3 = (try self.opening(&s3)) orelse return error.ExpectedResume;
    try t.expectEqual(@as(usize, 262_853), r3.at);
    try t.expectEqualSlices(u32, &.{ 262_853, 262_886 }, seq.cuts.items);
    try t.expectEqual(@as(usize, 33), try prompt_cuts.stepRows(prompt3.len, seq.cuts.items, r3.at, 34, 4 * C));

    // Rewind turn: resumes earlier rewind mark 260,000 cleanly and cuts at new mark 260,400.
    var chunks_rw: std.ArrayList(u32) = .empty;
    defer chunks_rw.deinit(a);
    for (chunks1.items) |c| if (c < 260_000) try chunks_rw.append(a, c);
    try chunks_rw.appendSlice(a, &.{ 260_000, 260_400 });
    var k_rw: engine.Kept = .{ .id = 13, .at = 260_000, .below = 0 };
    var s_rw = try lanes.Stream.init(a, .{ .id = "rw", .prompt = prompt3[0..260_402], .max_new = 64, .drafts = false, .chunks = chunks_rw.items, .reuse = .{ .saved = &k_rw, .at = 260_000, .marks = &.{260_400} } });
    defer s_rw.deinit(a);
    _ = self.lanes.remove(&s3);
    const r_rw = (try self.opening(&s_rw)) orelse return error.ExpectedResume;
    try t.expectEqual(@as(usize, 260_000), r_rw.at);
    try t.expectEqualSlices(u32, &.{ 260_000, 260_400 }, seq.cuts.items);

    // Unusable position not in s.chunks is rejected in .marks mode too.
    var k_bad: engine.Kept = .{ .id = 13, .at = 12_345, .below = 0 };
    var s_bad = try lanes.Stream.init(a, .{ .id = "bad", .prompt = prompt3[0..20_000], .max_new = 64, .drafts = false, .chunks = &.{ 650, 260_000 }, .reuse = .{ .saved = &k_bad, .at = 12_345 } });
    defer s_bad.deinit(a);
    try t.expectEqual(@as(?engine.Resume, null), self.resumeFrom(&s_bad, 20_000));
    try t.expectEqual(@as(?anyerror, error.SnapshotOutOfStep), s_bad.reuse_error);
}

test "glm request starts: cancelled prefill in marks mode skips unusable mid-chunk positions and saves usable ones" {
    const a = t.allocator;
    const Hook = struct {
        saved: std.ArrayList(u32) = .empty,
        fn at(ptr: *anyopaque, _: *lanes.Stream, pos: u32) void {
            const h: *@This() = @ptrCast(@alignCast(ptr));
            h.saved.append(t.allocator, pos) catch unreachable;
        }
    };
    var hook: Hook = .{};
    defer hook.saved.deinit(a);
    var e: engine.Engine = undefined;
    e.pages = null;
    var self: glm_lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined }, .request_starts = .marks };
    const prompt = try a.alloc(u32, 20_000);
    defer a.free(prompt);
    @memset(prompt, 1);
    var s = try lanes.Stream.init(a, .{ .id = "cancel", .prompt = prompt, .max_new = 16, .drafts = false, .chunks = &.{ 650, 4096, 16_500 }, .reuse = .{ .marks = &.{16_500}, .hook = .{ .ptr = &hook, .at = Hook.at } } });
    defer s.deinit(a);
    var keeps: share_mod.Keeps = undefined;
    const marks = self.marksOf(&s, &keeps).?;
    const interrupted = marks.interrupted.?;
    // 8192 is a GPU cold cut under e.cur.cuts=[16_500], but NOT in s.chunks: marks mode skips saving it.
    interrupted(marks.ctx, 8192);
    try t.expectEqualSlices(u32, &.{}, hook.saved.items);
    // 4096 and 16_500 ARE in s.chunks: marks mode saves them.
    interrupted(marks.ctx, 4096);
    interrupted(marks.ctx, 16_500);
    try t.expectEqualSlices(u32, &.{ 4096, 16_500 }, hook.saved.items);
}

test "glm request starts: executedBelow accumulates multi-hop cut history, matches belowExecuted, and rejects bad below" {
    const a = t.allocator;
    // Cold Turn 1 at 1,027 (< 2048): executedBelow is empty, below is 0.
    const ex_short = try prompt_cuts.executedBelow(a, &.{}, 0, &.{1027}, 1027);
    defer a.free(ex_short);
    try t.expectEqualSlices(u32, &.{}, ex_short);
    try t.expectEqual(@as(u64, 0), prompt_cuts.below(ex_short, 1027));
    try t.expectEqual(prompt_cuts.below(ex_short, 1027), prompt_cuts.belowExecuted(&.{}, 0, &.{1027}, 1027));

    // Cold Turn 1 at 4,100 -> Turn 2 at 4,650 -> Turn 3 at 4,900 -> Turn 4 long suffix at 9,000.
    const ex1 = try prompt_cuts.executedBelow(a, &.{}, 0, &.{4100}, 4100);
    defer a.free(ex1);
    try t.expectEqualSlices(u32, &.{ 2048, 4096 }, ex1);
    const b1 = prompt_cuts.below(ex1, 4100);
    try t.expectEqual(@as(u64, 0), b1);
    try t.expectEqual(b1, prompt_cuts.belowExecuted(&.{}, 0, &.{4100}, 4100));

    const ex2 = try prompt_cuts.executedBelow(a, ex1, 4100, &.{ 4100, 4650 }, 4650);
    defer a.free(ex2);
    try t.expectEqualSlices(u32, &.{ 2048, 4096, 4100 }, ex2);
    const b2 = prompt_cuts.below(ex2, 4650);
    try t.expect(b2 != 0);
    try t.expectEqual(b2, prompt_cuts.belowExecuted(ex1, 4100, &.{ 4100, 4650 }, 4650));

    const ex3 = try prompt_cuts.executedBelow(a, ex2, 4650, &.{ 4650, 4900 }, 4900);
    defer a.free(ex3);
    try t.expectEqualSlices(u32, &.{ 2048, 4096, 4100, 4650 }, ex3);
    const b3 = prompt_cuts.below(ex3, 4900);
    try t.expect(b3 != 0 and b3 != b2);
    try t.expectEqual(b3, prompt_cuts.belowExecuted(ex2, 4650, &.{ 4650, 4900 }, 4900));

    const ex4 = try prompt_cuts.executedBelow(a, ex3, 4900, &.{ 4900, 9000 }, 9000);
    defer a.free(ex4);
    try t.expectEqualSlices(u32, &.{ 2048, 4096, 4100, 4650, 4900, 6948, 8996 }, ex4);
    try t.expectEqual(prompt_cuts.below(ex4, 9000), prompt_cuts.belowExecuted(ex3, 4900, &.{ 4900, 9000 }, 9000));

    // Re-saving right at at == resumed_at preserves inherited cuts in (0, at) without dropping or duplicating at.
    const ex_same = try prompt_cuts.executedBelow(a, &.{300}, 1027, &.{ 1027, 2500 }, 1027);
    defer a.free(ex_same);
    try t.expectEqualSlices(u32, &.{300}, ex_same);
    const b_same = prompt_cuts.below(&.{300}, 1027);
    try t.expect(b_same != 0);
    try t.expectEqual(b_same, prompt_cuts.below(ex_same, 1027));
    try t.expectEqual(b_same, prompt_cuts.belowExecuted(&.{300}, 1027, &.{ 1027, 2500 }, 1027));
    const ex_same2 = try prompt_cuts.executedBelow(a, ex2, 4650, &.{ 4650, 4900 }, 4650);
    defer a.free(ex_same2);
    try t.expectEqualSlices(u32, ex2, ex_same2);
    try t.expectEqual(b2, prompt_cuts.belowExecuted(ex2, 4650, &.{ 4650, 4900 }, 4650));
    var seq: Seq = .{ .st = undefined };
    defer seq.deinitMeta(a);
    const k_same: engine.Kept = .{ .id = 1, .at = 1027, .below = b_same, .executed = @constCast(&[_]u32{300}), .has_executed = true };
    try seq.setExecuted(a, &k_same);
    try seq.cuts.appendSlice(a, &.{ 1027, 2500 });
    const ex_seq = try prompt_cuts.executedBelow(a, seq.resumed_starts.items, seq.resumed_at, seq.cuts.items, 1027);
    defer a.free(ex_seq);
    try t.expectEqualSlices(u32, &.{300}, ex_seq);
    try t.expectEqual(b_same, prompt_cuts.belowExecuted(seq.resumed_starts.items, seq.resumed_at, seq.cuts.items, 1027));

    // resumeFrom in marks mode accepts matching below and rejects corrupted below when has_executed is true.
    var e: engine.Engine = undefined;
    e.gpa = a;
    e.snaps = .empty;
    defer e.snaps.deinit(a);
    try e.snaps.put(a, 77, undefined);
    var self: glm_lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined }, .request_starts = .marks };
    var k: engine.Kept = .{ .id = 77, .at = 4900, .below = b3, .executed = ex3, .has_executed = true };
    const prompt = try a.alloc(u32, 5000);
    defer a.free(prompt);
    @memset(prompt, 3);
    var s = try lanes.Stream.init(a, .{ .id = "hop", .prompt = prompt, .max_new = 16, .drafts = false, .chunks = &.{ 300, 4100, 4650, 4900 }, .reuse = .{ .saved = &k, .at = 4900 } });
    defer s.deinit(a);
    const ok = self.resumeFrom(&s, prompt.len) orelse return error.ExpectedResume;
    try t.expectEqual(@as(usize, 4900), ok.at);

    k.below = b3 ^ 2; // corrupted cut digest must be rejected
    s.reuse_failed = false;
    s.reuse_error = null;
    s.cached = 0;
    try t.expectEqual(@as(?engine.Resume, null), self.resumeFrom(&s, prompt.len));
    try t.expectEqual(@as(?anyerror, error.SnapshotOutOfStep), s.reuse_error);
}
