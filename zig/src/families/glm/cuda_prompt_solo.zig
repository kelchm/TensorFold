//! A solo prompt pass (TF_GLM_CHUNK_SOLO): whole grid chunks in one pass, while no other stream decodes.
const std = @import("std");
const fw = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const Engine = engine.Engine;

/// Whether a stream besides `s` is past its prompt pass (a lane with no fill): a solo pass would hold up its rounds.
pub fn othersDecode(lanes: anytype, s: anytype) bool {
    return @import("lanes").solo_chunk.othersDecode(lanes, s, "fill");
}

/// Whether a pass at `start` of a `len`-id prompt can reach past the next grid point: two rows, and no mark at it.
pub fn takesTwo(start: usize, len: usize, marks: []const u32) bool {
    return takesChunks(start, len, 2 * fw.prefill_rows, marks);
}

/// Every grid boundary inside a solo pass must be unmarked; the last row may end the prompt off the grid.
pub fn takesChunks(start: usize, len: usize, width: usize, marks: []const u32) bool {
    if (start >= len or start % fw.prefill_rows != 0 or width <= fw.prefill_rows) return false;
    if (len - start <= fw.prefill_rows + 1) return false; // one row: a chunk's part as sent
    const end = start + rowsAt(start, len, width);
    var next = start + fw.prefill_rows;
    while (next < end) : (next += fw.prefill_rows) {
        if (std.mem.indexOfScalar(u32, marks, @intCast(next)) != null) return false;
    }
    return true;
}

/// Whether a pass at `start` to `rows` rebases the index rings where chunks one at a time do: at its start, or not.
pub fn ringsAgree(base: usize, ring: usize, start: usize, rows: usize) bool {
    if (start < base or rows > ring) return false;
    return start + fw.prefill_rows > base + ring or start + rows <= base + ring;
}

/// The rows a solo pass takes at `start` of a `len`-id prompt.
pub fn rowsAt(start: usize, len: usize, width: usize) usize {
    return @min(len - start, width);
}

/// Rank 0's choice for the step at `start`; not beside what reads a chunk's rows, nor with request starts (cut there).
pub fn at(e: *const Engine, start: usize, len: usize, from: ?engine.Resume, marks: ?engine.Marks) bool {
    if (e.bp.rows <= fw.prefill_rows or e.prompt_logprobs != null or e.prompt_profs[3] != null or e.cur.cuts.items.len > 0) return false;
    if (e.images) |v| if (v.rowCount() > 0) return false;
    if (!takesChunks(start, len, e.bp.rows, if (marks) |m| m.at else &.{})) return false;
    // a resumed step restores its kept state, and the rings' base, after this choice
    const base = if (from) |r| (e.snaps.get(r.id) orelse return false).bases[0] else e.cur.st.base;
    return ringsAgree(base, e.cur.st.ring_rows, start, rowsAt(start, len, e.bp.rows));
}

/// Whether a pass of `rows` past a chunk may run: `at`'s rule on every rank, its kept state restored.
pub fn allowed(e: *const Engine, start: usize, rows: usize) bool {
    if (e.bp.rows < rows or rows <= fw.prefill_rows or start % fw.prefill_rows != 0 or e.prompt_logprobs != null or e.cur.cuts.items.len > 0) return false;
    return ringsAgree(e.cur.st.base, e.cur.st.ring_rows, start, rows);
}

/// decode._absorb_rows of a step, a chunk's rows at a time: the head's ring holds a window more than a pass.
pub fn mtpRows(e: *Engine, f: *fw.Forward, ids: []const i32) !void {
    return mtpRowsAt(e, f, ids, e.bp.fnormed);
}

/// mtpRows of a step whose final-norm rows are at `hidden` (a paired step's second chunk: the twin's).
pub fn mtpRowsAt(e: *Engine, f: *fw.Forward, ids: []const i32, hidden: u64) !void {
    const mp = f.prof;
    defer f.prof = mp;
    f.prof = null; // the head's pass as one section
    var done: usize = 0;
    while (done + 1 < ids.len) {
        const nxt = ids[done + 1 .. @min(ids.len, done + 1 + fw.prefill_rows)];
        try f.mtp(nxt, &.{hidden + done * e.c.hidden * 2});
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
        done += nxt.len;
    }
}

test "a solo pass: rows past the next grid point, and no mark at it" {
    const C = fw.prefill_rows;
    try std.testing.expect(takesTwo(0, C + 2, &.{}));
    try std.testing.expect(!takesTwo(0, C + 1, &.{})); // its part would read as a chunk and the next id
    try std.testing.expect(!takesTwo(0, C, &.{})); // one chunk to the end
    try std.testing.expect(!takesTwo(C, 2 * C, &.{}));
    try std.testing.expect(takesTwo(2 * C, 5 * C, &.{}));
    try std.testing.expect(!takesTwo(0, 3 * C, &.{C})); // a state kept at the next chunk: that chunk starts a step
    try std.testing.expect(takesTwo(0, 3 * C, &.{ 2 * C, 3 * C })); // marks past the pass
}

test "a solo pass rebases the rings where chunks one at a time do" {
    const C = fw.prefill_rows;
    try std.testing.expect(ringsAgree(0, 2 * C, 0, 2 * C)); // neither rebases
    try std.testing.expect(!ringsAgree(0, 2 * C, C, 2 * C)); // chunks rebase at the second, a pass at its first
    try std.testing.expect(!ringsAgree(0, 2 * C, C, C + 1));
    try std.testing.expect(ringsAgree(0, 2 * C, 2 * C, 2 * C)); // both at the pass's start
    try std.testing.expect(ringsAgree(C, 2 * C, C, 2 * C));
}

test "a solo pass fits a whole ring once it rebases, and its buffers only when asked" {
    const state = @import("cuda_state.zig");
    for ([_]usize{ 2048, 4096, 8192 }) |width| {
        try std.testing.expectEqual(@max(state.index_ring, width), state.ringRows(1 << 20, width));
    }
}
