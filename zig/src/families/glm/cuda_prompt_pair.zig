//! A prompt step of two chunks in one pass (TF_GLM_PROMPT_PAIR): each chunk's exchanges under the other's sites.
const std = @import("std");
const cuda = @import("cuda");
const fw = @import("cuda_forward.zig");
const weights = @import("cuda_weights.zig");
const snapshots = @import("cuda_snapshots.zig");
const engine = @import("cuda_engine.zig");
const dflash = @import("dflash.zig");
const prompt_solo = @import("cuda_prompt_solo.zig");
const Engine = engine.Engine;

/// A paired step's fewest second-chunk rows beside a first of `rows`: a shorter one ends before a's exchanges do.
pub fn minRows(rows: usize) usize {
    return rows / 4;
}

/// minRows of the grid's chunks.
pub const min_rows = minRows(fw.prefill_rows);

/// A paired step's second chunk: twins of the prompt buffers, pieces on their side stream, and its first position.
pub const Pair = struct {
    bp: fw.Buffers,
    pieces: fw.Pieces,
    pos: cuda.DeviceBuffer, // an i32: where the second chunk's view of the sequence starts
    rows: usize, // each chunk's rows at most: the prompt buffers', a solo pass's with TF_GLM_PROMPT_PAIR=solo
    steps: u64 = 0, // paired steps so far (glm-pair-check counts them)

    pub fn init(d: *const cuda.Driver, w: *const weights.Weights, bp: *const fw.Buffers, pieces: *const fw.Pieces, capacity: usize) !Pair {
        var b = try fw.Buffers.initTwin(d, w, bp, capacity);
        errdefer b.deinit();
        var p = try fw.Pieces.initTwin(d, pieces);
        errdefer p.deinit();
        return .{ .bp = b, .pieces = p, .pos = try cuda.DeviceBuffer.alloc(d, 4), .rows = bp.rows };
    }

    pub fn deinit(p: *Pair) void {
        p.pos.free();
        p.pieces.deinit();
        p.bp.deinit();
        p.* = undefined;
    }
};

/// Whether an engine makes the twin: pairing on, no solo passes unless it pairs them, and ranks to hide exchanges from.
pub fn wanted(on: bool, solo: bool, with_solo: bool, world: usize) bool {
    return on and (!solo or with_solo) and world > 1;
}

/// Whether a step at `start` takes a second chunk past a first of `rows`: enough rows, no mark at a grid point inside.
pub fn takesTwo(start: usize, len: usize, world: usize, marks: []const u32, rows: usize) bool {
    const next = start + rows;
    if (next >= len or len - next < @max(minRows(rows), world)) return false;
    var at = start + fw.prefill_rows; // a kept state starts a step at any grid point, inside a solo-sized chunk too
    while (at < @min(len, next + rows)) : (at += fw.prefill_rows) {
        if (std.mem.indexOfScalar(u32, marks, @intCast(at)) != null) return false;
    }
    return true;
}

/// Whether a pair of `ra` and `rb` rows rebases the index rings where grid chunks one at a time do (as one solo pass).
pub fn ringsAgree(base: usize, ring: usize, start: usize, ra: usize, rb: usize) bool {
    return prompt_solo.ringsAgree(base, ring, start, ra + rb);
}

/// Where a paired step's chunks tap for DFlash2: only a chunk reaching the prompt's last window, as prefillChunk taps.
pub const TapPlan = struct {
    a_kept: usize, // a's rows in the last window (0: a taps none), tapped into taps_prefill
    b: bool, // b taps: into taps_prefill when a taps none, else into the context input past a's rows (pairTaps)
};

/// The pair at `start` (a of `ra` rows, b of `rb`) of a `len`-id prompt: b's rows follow a's in one window.
pub fn tapPlan(start: usize, ra: usize, rb: usize, len: usize) TapPlan {
    const keep_from = len -| @as(usize, @intCast(dflash.WIN)); // dflash.promptRun's first retained row
    const end = start + ra;
    return .{ .a_kept = @min(ra, end -| keep_from), .b = end + rb > keep_from };
}

/// The pieces a forward over `b` exchanges in: the engine's, or the pair's for its twin; none for other buffers.
pub fn piecesOf(e: *const Engine, b: *const fw.Buffers) ?*const fw.Pieces {
    if (b == &e.bp) return &e.pieces;
    const p = if (e.pair) |*p| p else return null;
    return if (b == &p.bp) &p.pieces else null;
}

/// Rank 0's choice for the step at `start` (the others take the ids it sends); not with prompt rows or prof.
pub fn pairedAt(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    const ra = if (e.pair) |p| p.rows else return false;
    if (!e.settings.pair or e.prompt_logprobs != null or e.prompt_profs[3] != null) return false;
    if (e.cur.cuts.items.len > 0) return false; // a paired step would step over a request start
    if (e.pieces.count == 1 and !e.settings.split) return false; // run takes such a chunk whole
    if (ra > fw.prefill_rows) if (e.images) |v| if (v.rowCount() > 0) return false; // as prompt_solo.at
    if (!takesTwo(start, len, e.share.world, if (marks) |m| m.at else &.{}, ra)) return false;
    // a resumed step restores its kept state, and the rings' base, after this choice
    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    return ringsAgree(base, e.cur.st.ring_rows, start, ra, @min(len - start - ra, ra));
}

/// Engine.prefillChunk of two chunks (`part`: both, then the next id): one pass, then each chunk's MTP rows and commit.
pub fn prefill(e: *Engine, part: []const u32, start: usize, len: usize, cancel: ?engine.Cancel, marks: ?engine.Marks, split: bool) !?u32 {
    const pair = if (e.pair) |*p| p else return error.BadChunk;
    const ra: usize = pair.rows;
    if (len <= start + ra) return error.BadChunk;
    const rb: usize = @min(len - start - ra, ra);
    if (rb < e.share.world or part.len != @min(len - start, ra + rb + 1) or e.prompt_logprobs != null) return error.BadChunk;
    if (!ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, ra, rb)) return error.BadChunk; // pairedAt's rule, every rank
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
    const tp = tapPlan(start, ra, rb, len); // DFlash2 taps where two chunks one at a time do
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
    if (e.mtp) try prompt_solo.mtpRowsAt(e, &fa, ids[0 .. ra + 1], e.bp.fnormed); // a grid chunk's rows at a time, as unpaired
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
    if (e.mtp) try prompt_solo.mtpRowsAt(e, &fa, ids[ra..], pair.bp.fnormed); // the MTP head on a's buffers, over b's rows
    try fa.commit(rb);
    return first;
}

test "a prompt step takes the next chunk when it holds min_rows or more and no kept state starts there" {
    const C = fw.prefill_rows;
    try std.testing.expect(takesTwo(0, 2 * C, 3, &.{}, C));
    try std.testing.expect(takesTwo(0, C + min_rows, 3, &.{}, C));
    try std.testing.expect(!takesTwo(0, C + min_rows - 1, 3, &.{}, C)); // a shorter tail goes alone
    try std.testing.expect(!takesTwo(0, C, 3, &.{}, C));
    try std.testing.expect(!takesTwo(C, 2 * C, 3, &.{}, C)); // the prompt's last chunk
    try std.testing.expect(takesTwo(2 * C, 5 * C, 3, &.{}, C));
    try std.testing.expect(!takesTwo(0, 3 * C, 3, &.{C}, C)); // a state kept at the next chunk's start: that chunk starts a step
    try std.testing.expect(takesTwo(0, 3 * C, 3, &.{ 2 * C, 3 * C }, C)); // marks past the pair
}

test "a solo pass pairs (TF_GLM_PROMPT_PAIR=solo): a second solo chunk of a quarter's rows or more, and no mark at any grid point inside" {
    const C = fw.prefill_rows;
    const S = 2 * C;
    try std.testing.expectEqual(@as(usize, C / 2), minRows(S));
    try std.testing.expect(takesTwo(0, 2 * S, 2, &.{}, S));
    try std.testing.expect(takesTwo(0, S + minRows(S), 2, &.{}, S));
    try std.testing.expect(!takesTwo(0, S + minRows(S) - 1, 2, &.{}, S)); // the tail goes alone, after a solo chunk
    try std.testing.expect(!takesTwo(0, S, 2, &.{}, S));
    try std.testing.expect(takesTwo(2 * C, 9 * C, 2, &.{}, S)); // pairs start on the grid, not on a solo chunk's
    for ([_]u32{ C, 2 * C, 3 * C }) |mark| try std.testing.expect(!takesTwo(0, 5 * C, 2, &.{mark}, S));
    try std.testing.expect(!takesTwo(0, 3 * C + 600, 2, &.{3 * C}, S)); // in the second chunk's rows
    try std.testing.expect(takesTwo(0, 5 * C, 2, &.{ 4 * C, 5 * C }, S)); // at the next step's start and past it
    try std.testing.expect(takesTwo(C, 6 * C, 2, &.{C}, S)); // the step's own start
}

test "a pair rebases the index rings where a chunk at a time would, so a ring based at chunk 0 pairs from chunk 2" {
    const C = fw.prefill_rows;
    try std.testing.expect(ringsAgree(0, 2 * C, 0, C, C)); // neither rebases
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C, C)); // a chunk at a time rebases at b's start, a pair at a's
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C, 1));
    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, C, C)); // both at a's start
    try std.testing.expect(ringsAgree(C, 2 * C, C, C, C));
}

test "a solo pass pairs where its grid chunks rebase: a pair of two solo chunks needs a ring of both" {
    const C = fw.prefill_rows;
    for ([_]usize{ C, 2 * C }) |ra| for ([_]usize{ 2 * ra, 4 * ra }) |ring| for (0..12) |chunk| for ([_]usize{ minRows(ra), ra - 1, ra }) |rb| {
        const start = chunk * C;
        var base: usize = 0; // State.ready of grid chunks one at a time, from a ring based at 0
        var at = start;
        while (at < start + ra + rb) : (at += C) {
            if (at + @min(C, start + ra + rb - at) > base + ring) base = at;
        }
        const pair_base: usize = if (start + ra + rb > ring) start else 0; // the pair's one State.ready at a
        try std.testing.expectEqual(base == pair_base, ringsAgree(0, ring, start, ra, rb));
    };
    try std.testing.expect(!ringsAgree(0, 2 * C, 0, 2 * C, C)); // two solo chunks overrun a ring of one
    try std.testing.expect(ringsAgree(0, 4 * C, 4 * C, 2 * C, 2 * C)); // the second pair rebases at its start
    try std.testing.expect(!ringsAgree(0, 4 * C, 2 * C, 2 * C, 2 * C)); // grid chunks would rebase at 4C, inside
}

test "only ranks that exchange make the twin: TP=1 has no exchanges to hide" {
    try std.testing.expect(!wanted(true, false, false, 1));
    try std.testing.expect(wanted(true, false, false, 2));
    try std.testing.expect(!wanted(false, false, false, 3));
    try std.testing.expect(!wanted(true, true, true, 1));
}

test "a server with solo passes holds no twin, unless TF_GLM_PROMPT_PAIR=solo pairs them" {
    try std.testing.expect(!wanted(true, true, false, 2));
    try std.testing.expect(!wanted(true, true, false, 3));
    try std.testing.expect(wanted(true, true, true, 2));
    try std.testing.expect(wanted(true, true, true, 3));
}

test "a pair taps for DFlash2 where two chunks one at a time do, b's rows after a's in one window" {
    const C = fw.prefill_rows;
    const W: usize = @intCast(dflash.WIN);
    for ([_]usize{ C, 2 * C }) |ra| for ([_]usize{ 0, C, 3 * C }) |start| for ([_]usize{ minRows(ra), ra - 1, ra }) |tail| for ([_]usize{ 0, 1, 300, C - 1, C, 2 * C }) |after| {
        const len = start + ra + tail + after;
        const rb = @min(len - start - ra, ra);
        const p = tapPlan(start, ra, rb, len);
        const ra_run = try dflash.promptRun(0, 0, start, ra, len, 0, ra); // taps_prefill holds a chunk (dflash.tapRows)
        const rb_run = try dflash.promptRun(0, 0, start + ra, rb, len, 0, ra);
        try std.testing.expectEqual(if (ra_run) |r| r.n else 0, p.a_kept); // a chunk alone keeps these rows
        try std.testing.expectEqual(rb_run != null, p.b);
        if (p.a_kept == 0) continue;
        // b's rows sit after a's in the context input: all of b in the window, both within one ingest's rows
        try std.testing.expectEqual(@as(i64, @intCast(start + ra)), rb_run.?.pos0);
        try std.testing.expectEqual(rb, rb_run.?.n);
        try std.testing.expect(p.a_kept + rb <= dflash.MAX_CTX and p.a_kept + rb <= W);
    };
}
