//! Transactional, opt-in text prefill. A prepare-only header precedes admission and any forward collective.
const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const undo = @import("cuda_slice_undo.zig");
const snapshots = @import("cuda_snapshots.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const df = @import("dflash.zig");
const max_layers = @import("config.zig").max_layers;
const Engine = engine.Engine;
pub const Workspace = @import("cuda_slice_workspace.zig").Workspace;

pub fn startup(e: *Engine, enabled: bool) !void {
    if (!enabled) return;
    if (e.slice_workspace != null) return;
    e.slice_workspace = try Workspace.create(e);
}

pub const Partial = struct {
    slot: u32,
    start: usize,
    len: usize,
    j: usize = 0,
    rows: usize,
    part: []u32,
    split: bool,
    undo: undo.Undo,
    events: [2]cuda.Event,
    costs: [max_layers + 1]f64 = @splat(0),
    calibrated: bool,
    measured: usize = 0,
    admitted: bool = false,
    logprobs: ?u8,
};

pub fn resident(e: *const Engine) usize {
    return if (e.slice_workspace) |w| w.resident() else 0;
}

pub fn hi(e: *Engine, rows: usize, lo: usize, budget: f64) !usize {
    if (lo >= e.w.layers.len or !std.math.isFinite(budget) or budget <= 0) return error.BadSlice;
    if (e.slice_costs.readyRows(rows)) return @min(e.w.layers.len, e.slice_costs.stop(rows, lo, budget));
    return lo + 1;
}

/// No forward and no control collective: the leader does this before announcing preparation to its peers.
pub fn prepare(e: *Engine, part: []const u32, start: usize, len: usize, from: ?engine.Resume) !void {
    if (e.partial != null) return error.SliceBusy;
    if (e.w.layers.len == 0 or e.w.layers.len > max_layers) return error.SliceUnsupportedThin;
    if (e.images) |v| if (v.source.images.len != 0) return error.SliceMediaWholeChunk;
    if (start >= len or start % engine.prompt_rows != 0 or !Engine.promptFits(len, e.cur.st.capacity)) return error.BadChunk;
    const rows = @min(len - start, engine.prompt_rows);
    if (part.len != @min(rows + 1, len - start)) return error.BadChunk;
    if (from != null and try e.prefillStart(len, from) != start) return error.BadChunk;
    if (e.cur.st.pos != start) return error.BadChunk;
    const workspace = e.slice_workspace orelse return error.SlicingNotStarted;
    var saved = try undo.Undo.capture(e);
    errdefer saved.close(e.gpa);
    const ids = workspace.part[0..part.len];
    @memcpy(ids, part);
    const p: Partial = .{
        .slot = e.slot,
        .start = start,
        .len = len,
        .rows = rows,
        .part = ids,
        .split = e.index_split,
        .undo = saved,
        .events = workspace.events,
        .calibrated = e.slice_costs.readyRows(rows),
        .logprobs = if (e.prompt_logprobs) |w| w.k else null,
    };
    e.partial = p;
    e.index_split = false;
}

/// A failed follower still joins this agreement; a prepare header never authorizes forward work.
pub fn admit(e: *Engine, failed: ?anyerror) !void {
    if (e.partial) |p| if (p.admitted) undo.fatal(error.SliceProtocol);
    if (failed) |err| undo.admissionError(err);
    const bad = e.shared(if (failed != null) 1 else 0) catch |err| undo.fatal(err);
    if (bad != 0) {
        if (e.partial) |p| dispose(e, p.slot);
        return failed orelse error.PeerSliceAdmission;
    }
    e.partial.?.admitted = true;
}

fn dispose(e: *Engine, slot: u32) void {
    if (e.partial) |*p| if (p.slot == slot) {
        p.undo.close(e.gpa);
        e.partial = null;
    };
}

/// All ranks call rollback before interrupted marks, slot release, reset, or another forward on the owner.
pub fn rollback(e: *Engine, slot: u32) bool {
    const p = if (e.partial) |*v| v else return false;
    if (p.slot != slot) return false;
    p.undo.restore(e) catch |err| undo.fatal(err);
    if (p.admitted) {
        const bad = e.shared(0) catch |err| undo.fatal(err);
        if (bad != 0) undo.fatal(error.PeerSliceRollback);
    }
    dispose(e, slot);
    return true;
}

/// The same row geometry and kernel order as a whole chunk, with ownership retained between layer ranges.
pub fn run(e: *Engine, part: []const u32, start: usize, len: usize, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks, end: usize) !?u32 {
    // The direct Engine API remains useful for a single-rank device check; TP must use the prepare protocol.
    if (e.partial == null) {
        if (e.share.world != 1) return error.SliceNotAdmitted;
        prepare(e, part, start, len, from) catch |err| {
            undo.admissionError(err);
            return err;
        };
        try admit(e, null);
    }
    const p = &e.partial.?;
    const invalid = !p.admitted or p.slot != e.slot or p.start != start or p.len != len or end <= p.j or end > e.w.layers.len or
        p.logprobs != (if (e.prompt_logprobs) |w| w.k else @as(?u8, null));
    if ((e.shared(@floatFromInt(@intFromBool(invalid))) catch |err| undo.fatal(err)) != 0) {
        _ = rollback(e, p.slot);
        return error.BadSlice;
    }
    if (p.j == 0) if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    // Gate every slice, so cancellation does not wait for completion of the rest of the chunk.
    e.gate(cancel) catch |err| {
        if (err != error.Cancelled) undo.fatal(err);
        _ = rollback(e, e.slot);
        snapshots.interrupted(e, marks, @intCast(start)) catch |why| undo.fatal(why);
        return err;
    };
    var one: [1]fw.Rank = undefined;
    var f = e.view(&e.bp, &one);
    f.index_split = p.split;
    f.prof = null; // chunk-long profiling intervals would include another stream's decode work
    f.images = null;
    if (start + p.rows <= len -| @as(usize, @intCast(df.WIN))) f.taps = null;
    p.events[0].record(e.stream) catch |err| undo.fatal(err);
    const ids: []const i32 = @ptrCast(p.part[0..p.rows]);
    f.runRange(ids, p.j, end) catch |err| undo.fatal(err);
    p.events[1].record(e.stream) catch |err| undo.fatal(err);
    e.stream.synchronize() catch |err| undo.fatal(err);
    const ms = p.events[0].elapsedMs(p.events[1]) catch |err| undo.fatal(err);
    if (!std.math.isFinite(ms) or ms < 0) undo.fatal(error.BadSliceTiming);
    if (!p.calibrated and end == p.j + 1) {
        p.costs[p.j] = ms;
        p.measured += 1;
    }
    if (p.calibrated) e.slice_costs.observe(p.rows, p.j, end, ms);
    p.j = end;
    if (end < e.w.layers.len) return null;
    p.events[0].record(e.stream) catch |err| undo.fatal(err);
    const first = tail(e, &f, p) catch |err| undo.fatal(err);
    p.events[1].record(e.stream) catch |err| undo.fatal(err);
    e.stream.synchronize() catch |err| undo.fatal(err);
    const tail_ms = p.events[0].elapsedMs(p.events[1]) catch |err| undo.fatal(err);
    if (!std.math.isFinite(tail_ms) or tail_ms < 0) undo.fatal(error.BadSliceTiming);
    if (!p.calibrated) {
        p.costs[end] = tail_ms;
        // Direct callers may choose wider cold ranges; only complete single-layer measurements train a band.
        if (p.measured == end) e.slice_costs.calibrate(p.rows, p.costs[0 .. end + 1]) catch {};
    } else e.slice_costs.observeTail(p.rows, tail_ms);
    if ((e.shared(0) catch |err| undo.fatal(err)) != 0) undo.fatal(error.PeerSliceCommit);
    e.index_splits += @intFromBool(p.split);
    dispose(e, e.slot);
    return first;
}

fn tail(e: *Engine, f: *fw.Forward, p: *const Partial) !?u32 {
    if (e.prompt_logprobs) |want| if (p.part.len > 1) try prompt_lp.run(e, want, p.part[1..]);
    if (e.drafter) |dr| try dr.prompt(e, e.slot, p.start, p.rows, p.len);
    if (e.drafter) |dr| if (p.start + p.rows == p.len) try dr.ringDigest(e, e.slot, p.len);
    const last = p.start + p.rows == p.len;
    var first: ?u32 = null;
    if (last) {
        var token: [1]u32 = undefined;
        try e.draw(e.bp.logits, 1, p.len, 0, &token);
        first = token[0];
        try e.holdPrompt(e.bp.fnormed + (p.rows - 1) * e.c.hidden * 2);
    }
    try f.commit(p.rows);
    return first;
}
