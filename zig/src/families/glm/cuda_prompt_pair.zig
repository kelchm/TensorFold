//! A prompt step of two chunks in one pass (TF_GLM_PROMPT_PAIR): each chunk's exchanges under the other's sites.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const snapshots = @import("cuda_snapshots.zig");
const engine = @import("cuda_engine.zig");
const dflash = @import("dflash.zig");
const Engine = engine.Engine;

/// The fewest rows a paired step's second chunk takes: a shorter one's sites end before the first's exchanges do.
pub const min_rows = fw.prefill_rows / 4;

/// A paired step's second chunk: twins of the prompt buffers, pieces on their side stream, and its first position.
pub const Pair = struct {
    bp: fw.Buffers,
    pieces: fw.Pieces,
    pos: cuda.DeviceBuffer, // an i32: where the second chunk's view of the sequence starts
    steps: u64 = 0, // paired steps so far (glm-pair-check counts them)

    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, bp: *const fw.Buffers, pieces: *const fw.Pieces, capacity: usize) !Pair {
        var b = try fw.Buffers.initTwin(d, w, bp, capacity);
        errdefer b.deinit();
        var p = try fw.Pieces.initTwin(d, pieces);
        errdefer p.deinit();
        return .{ .bp = b, .pieces = p, .pos = try cuda.DeviceBuffer.alloc(d, 4) };
    }

    pub fn deinit(p: *Pair) void {
        p.pos.free();
        p.pieces.deinit();
        p.bp.deinit();
        p.* = undefined;
    }
};

/// Whether an engine makes the pair's twin: TF_GLM_PROMPT_PAIR on, no solo passes, and ranks to hide exchanges from.
pub fn wanted(on: bool, solo: bool, world: usize) bool {
    return on and !solo and world > 1;
}

/// Whether a step at `start` of a `len`-id prompt takes the next chunk too: min_rows or more there, and no mark at it.
pub fn takesTwo(start: usize, len: usize, world: usize, marks: []const u32) bool {
    const next = start + fw.prefill_rows;
    if (next >= len or len - next < @max(min_rows, world)) return false;
    return std.mem.indexOfScalar(u32, marks, @intCast(next)) == null;
}

/// Whether a pair at `start` (b of `rb` rows) rebases the index rings where chunks one at a time do: at a, or not.
pub fn ringsAgree(base: usize, ring: usize, start: usize, rb: usize) bool {
    const ra = fw.prefill_rows;
    return start + ra > base + ring or start + ra + rb <= base + ring;
}

/// Where a paired step's chunks tap for DFlash2: only a chunk reaching the prompt's last window, as prefillChunk taps.
pub const TapPlan = struct {
    a_kept: usize, // a's rows in the last window (0: a taps none), tapped into taps_prefill
    b: bool, // b taps: into taps_prefill when a taps none, else into the context input past a's rows (pairTaps)
};

/// The pair at `start` (b of `rb` rows) of a `len`-id prompt: b's rows follow a's in one window, MAX_CTX rows at most.
pub fn tapPlan(start: usize, rb: usize, len: usize) TapPlan {
    const keep_from = len -| @as(usize, @intCast(dflash.WIN)); // dflash.promptRun's first retained row
    const end = start + fw.prefill_rows;
    return .{ .a_kept = end -| keep_from, .b = end + rb > keep_from };
}

/// The pieces a forward over `b` exchanges in: the engine's, or the pair's for its twin; none for other buffers.
pub fn piecesOf(e: *const Engine, b: *const fw.Buffers) ?*const fw.Pieces {
    if (b == &e.bp) return &e.pieces;
    const p = if (e.pair) |*p| p else return null;
    return if (b == &p.bp) &p.pieces else null;
}

/// Rank 0's choice for the step at `start` (the others take the ids it sends); not with prompt rows or prof.
pub fn pairedAt(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    if (e.pair == null or !e.settings.pair or e.prompt_logprobs != null or e.prompt_profs[3] != null) return false;
    if (e.cur.cuts.items.len > 0) return false; // a paired step would step over a request start
    if (e.pieces.count == 1 and !e.settings.split) return false; // run takes such a chunk whole
    if (!takesTwo(start, len, e.share.world, if (marks) |m| m.at else &.{})) return false;
    // a resumed step restores its kept state, and the rings' base, after this choice
    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    return ringsAgree(base, e.cur.st.ring_rows, start, @min(len - start - fw.prefill_rows, fw.prefill_rows));
}

/// Engine.prefillChunk of two chunks (`part`: both, then the next id): one pass, then each chunk's MTP rows and commit.
pub fn prefill(e: *Engine, part: []const u32, start: usize, len: usize, cancel: ?engine.Cancel, marks: ?engine.Marks, split: bool) !?u32 {
    const ra: usize = fw.prefill_rows;
    if (len <= start + ra) return error.BadChunk;
    const rb: usize = @min(len - start - ra, fw.prefill_rows);
    const pair = if (e.pair) |*p| p else return error.BadChunk;
    if (rb < e.share.world or part.len != @min(len - start, ra + rb + 1) or e.prompt_logprobs != null) return error.BadChunk;
    if (!ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, rb)) return error.BadChunk; // pairedAt's rule, every rank
    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    e.gate(cancel) catch |err| {
        if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
        return err;
    };
    var one: [1]fw.Rank = undefined;
    var two: [1]fw.Rank = undefined;
    var fa = e.view(&e.bp, &one);
    var fb = e.view(&pair.bp, &two);
    fa.prof = null;
    fb.prof = null;
    fa.index_split = split; // the step's choice (TF_GLM_INDEX_SPLIT) for both chunks
    fb.index_split = split;
    const tp = tapPlan(start, rb, len); // DFlash2 taps where two chunks one at a time do
    if (tp.a_kept == 0) fa.taps = null;
    fb.taps = if (e.drafter) |dr| (if (!tp.b) null else if (tp.a_kept > 0) dr.pairTaps(tp.a_kept) else dr.taps(true)) else null;
    const st = &e.cur.st;
    try st.ready(e.stream, ra + rb); // one rebase for both chunks, at a as a chunk at a time, before b's view copies it
    var sb = st.*;
    sb.pos_dev = pair.pos.ptr;
    try sb.setPos(e.stream, st.pos + ra);
    two[0].st = &sb;
    st.pending = ra;
    st.prompt = true;
    const ids: []const i32 = @ptrCast(part);
    try cuda.copy_source.enqueue(e.d, e.bp.ids, std.mem.sliceAsBytes(ids[0..ra]), e.stream);
    try cuda.copy_source.enqueue(e.d, pair.bp.ids, std.mem.sliceAsBytes(ids[ra .. ra + rb]), e.stream);
    try fw.Forward.pairBody(&fa, &fb, ra, rb);
    pair.steps += 1;
    e.index_splits += @intFromBool(split);
    if (e.drafter) |dr| { // a's rows, then b's where they were tapped: the runs two chunks one at a time ingest
        try dr.prompt(e, e.slot, start, ra, len);
        if (tp.a_kept > 0) try dr.promptIn(e, e.slot, start + ra, rb, len, tp.a_kept) else try dr.prompt(e, e.slot, start + ra, rb, len);
    }
    try mtpRows(e, &fa, ids[1 .. ra + 1], e.bp.fnormed);
    try fa.commit(ra);
    st.pending = rb;
    st.prompt = true;
    var first: ?u32 = null;
    if (start + ra + rb == len) {
        var t: [1]u32 = undefined;
        try e.draw(pair.bp.logits, 1, len, 0, &t);
        first = t[0];
        try e.holdPrompt(pair.bp.fnormed + (rb - 1) * e.c.hidden * 2);
    }
    try mtpRows(e, &fa, ids[ra + 1 ..], pair.bp.fnormed); // the MTP head on a's buffers, over b's rows
    try fa.commit(rb);
    return first;
}

/// decode._absorb_rows of a chunk: the MTP head over its next ids `nxt` on its final-norm rows `hidden`.
fn mtpRows(e: *Engine, f: *fw.Forward, nxt: []const i32, hidden: u64) !void {
    if (!e.mtp or nxt.len == 0) return;
    try f.mtp(nxt, &.{hidden});
    try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
}

test "a prompt step takes the next chunk when it holds min_rows or more and no kept state starts there" {
    const C = fw.prefill_rows;
    try std.testing.expect(takesTwo(0, 2 * C, 3, &.{}));
    try std.testing.expect(takesTwo(0, C + min_rows, 3, &.{}));
    try std.testing.expect(!takesTwo(0, C + min_rows - 1, 3, &.{})); // a shorter tail goes alone
    try std.testing.expect(!takesTwo(0, C, 3, &.{}));
    try std.testing.expect(!takesTwo(C, 2 * C, 3, &.{})); // the prompt's last chunk
    try std.testing.expect(takesTwo(2 * C, 5 * C, 3, &.{}));
    try std.testing.expect(!takesTwo(0, 3 * C, 3, &.{C})); // a state kept at the next chunk's start: that chunk starts a step
    try std.testing.expect(takesTwo(0, 3 * C, 3, &.{ 2 * C, 3 * C })); // marks past the pair
}

test "a pair rebases the index rings where a chunk at a time would, so a ring based at chunk 0 pairs from chunk 2" {
    const C = fw.prefill_rows;
    try std.testing.expect(ringsAgree(0, 2 * C, 0, C)); // neither rebases
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C)); // a chunk at a time rebases at b's start, a pair at a's
    try std.testing.expect(!ringsAgree(0, 2 * C, C, 1));
    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, C)); // both at a's start
    try std.testing.expect(ringsAgree(C, 2 * C, C, C));
}

test "only ranks that exchange make the twin: TP=1 has no exchanges to hide" {
    try std.testing.expect(!wanted(true, false, 1));
    try std.testing.expect(wanted(true, false, 2));
    try std.testing.expect(!wanted(false, false, 3));
}

test "a server with solo passes holds no twin: they take the pair's steps" {
    try std.testing.expect(!wanted(true, true, 2));
    try std.testing.expect(!wanted(true, true, 3));
}

test "a pair taps for DFlash2 where two chunks one at a time do, b's rows after a's in one window" {
    const C = fw.prefill_rows;
    const W: usize = @intCast(dflash.WIN);
    for ([_]usize{ 0, C, 3 * C }) |start| for ([_]usize{ min_rows, C - 1, C }) |tail| for ([_]usize{ 0, 1, 300, C - 1, C, 2 * C }) |after| {
        const len = start + C + tail + after;
        const rb = @min(len - start - C, C);
        const p = tapPlan(start, rb, len);
        const ra_run = try dflash.promptRun(0, 0, start, C, len, 0, C);
        const rb_run = try dflash.promptRun(0, 0, start + C, rb, len, 0, C);
        try std.testing.expectEqual(if (ra_run) |r| r.n else 0, p.a_kept); // a chunk alone keeps these rows
        try std.testing.expectEqual(rb_run != null, p.b);
        if (p.a_kept == 0) continue;
        // b's rows sit after a's in the context input: all of b in the window, both within one ingest's rows
        try std.testing.expectEqual(@as(i64, @intCast(start + C)), rb_run.?.pos0);
        try std.testing.expectEqual(rb, rb_run.?.n);
        try std.testing.expect(p.a_kept + rb <= dflash.MAX_CTX and p.a_kept + rb <= W);
    };
}
