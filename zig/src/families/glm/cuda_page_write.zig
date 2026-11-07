//! Native page writes: bounded single-rank COW, collective refusal until TP remapping exists.
//! Calls run on the engine/pool owner, outside forward collectives. Checkpoint readers retain their old ids.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const state = @import("cuda_state.zig");

/// A fixed bound even for restore and disjoint MTP ranges; larger protected writes refuse before allocation.
pub const max_copies = 20;
pub const Range = struct {
    start: usize,
    end: usize, // exclusive logical rows; pooled keys occupy the same 512-row page as their four input rows

    pub fn rows(start: usize, count: usize) !Range {
        return .{ .start = start, .end = try std.math.add(usize, start, count) };
    }
};

/// Every TP rank votes, including a rank with no protected pages or with a local validation failure.
/// TP does no allocation/copy/table update here: the leader alone owns Pool references today. Refuse the write
/// on every rank before any model kernel/collective, retaining the accepted checkpoint and the live owner.
pub fn guard(e: anytype, ranges: []const Range) !void {
    return at(e, e.slot, ranges);
}

/// Batched forwards keep their already-bound segments; guarding a slot must not reactivate media or allocate State.
pub fn at(e: anytype, slot: u32, ranges: []const Range) !void {
    if (e.share.world == 1) return local(e, slot, ranges, true);
    var failed: bool = false;
    local(e, slot, ranges, false) catch {
        failed = true;
    };
    const bad = e.shared(if (failed) 1 else 0) catch |err| fatal(err);
    if (bad != 0) return error.PageWriteDeferred;
}

/// A local failure computing the write range joins the SAME agreement as peers with a valid range.
pub fn refuse(e: anytype, why: anyerror) anyerror!void {
    if (e.share.world == 1) return why;
    _ = e.shared(1) catch |err| fatal(err);
    return error.PageWriteDeferred;
}

/// Prepare-only sliced prefill uses this before its existing admission vote, not before a new collective.
/// `copy=false` also checks rollback destinations without changing the undo capsule's canonical ids.
pub fn local(e: anytype, slot: u32, ranges: []const Range, copy: bool) !void {
    if (slot >= e.seqs.items.len) return error.PageWriteOwner;
    const st = &(e.seqs.items[slot] orelse return error.PageWriteOwner).st;
    if (st.table == 0 or (st.caches.len == 0 and !st.head)) return; // unpaged or no physical KV rows
    const p = e.pages orelse return error.PageWriteOwner;
    if (!p.paged(slot) or !p.used[slot]) return error.PageWriteOwner;
    if (p.leader != (e.share.rank == 0)) return error.PageWriteOwner;
    const live = &p.leases[slot];
    if (st.pages.ptr != live.ids.items.ptr or st.pages.len != live.ids.items.len) return error.PageWriteOwner;
    const page = p.layout.pageBytes();
    if (page == 0 or page != p.pool.page_bytes) return error.PageWriteOwner;
    const arena = p.arena orelse return error.PageWriteOwner;
    if (arena.len / page < p.pool.total()) return error.PageWriteOwner;
    if (st.table < st.arena.buf.ptr or st.table - st.arena.buf.ptr > st.arena.buf.len or
        live.ids.items.len > (st.arena.buf.len - @as(usize, @intCast(st.table - st.arena.buf.ptr))) / @sizeOf(u32)) return error.PageWriteOwner;
    for (st.caches) |c| if (c.pt != st.table or c.ps != page or c.lc < arena.ptr or c.lc - arena.ptr >= page) return error.PageWriteOwner;
    if (st.head) {
        const c = st.mtp_caches;
        if (c.pt != st.table or c.ps != page or c.lc < arena.ptr or c.lc - arena.ptr >= page) return error.PageWriteOwner;
    }
    var indices: [max_copies]usize = undefined;
    var n: usize = 0;
    for (ranges) |r| {
        if (r.end < r.start or r.end > st.pages.len * state.page_rows) return error.PastCapacity;
        if (r.start == r.end) continue;
        const end = (r.end - 1) / state.page_rows + 1;
        for (r.start / state.page_rows..end) |i| {
            const id = live.ids.items[i];
            if (id >= p.pool.total()) return error.PageWriteOwner;
            // Followers map the leader's table, not an independently allocated pool. They cannot infer exclusivity.
            if (!p.leader) continue;
            if (p.pool.leased[id] == 0 or p.pool.refs[id] < p.pool.leased[id]) return error.PageWriteOwner;
            if (p.pool.pins[id] > p.pool.refs[id] - p.pool.leased[id]) return error.PageWriteOwner;
            if (p.pool.refs[id] == 1) continue;
            if (std.mem.indexOfScalar(usize, indices[0..n], i) != null) continue;
            if (!copy) return error.PageWriteDeferred;
            if (n == indices.len) return error.PageWriteTooWide;
            indices[n] = i;
            n += 1;
        }
    }
    if (n == 0) return;
    if (!p.leader or e.share.world != 1) return error.PageWriteDeferred;
    // COW takes EXTRA, currently unpromised pages. Neither this lease's growth promise nor any other lease's
    // (including contiguous run reservations) is consumed or re-promised. No second device/cache arena exists.
    // Do not ask Pool.promise to invoke an optional store reclaimer; deferral retains every idle/checkpoint owner.
    if (p.pool.available() < n) return error.LaneMemory;
    var replacement: kv.Lease = .{};
    defer p.pool.finish(&replacement);
    try replacement.ids.ensureTotalCapacityPrecise(p.pool.gpa, n); // at most max_copies * 4 transient host bytes
    try p.pool.promise(&replacement, n);
    try p.pool.cover(&replacement, n, null);
    // No table or canonical owner changes until every physical page, including FP8 scales and MTP, has drained.
    errdefer cuda.completion.drain(e.d, e.stream, "GLM page COW copy did not drain");
    for (indices[0..n], replacement.ids.items) |i, id| {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(arena.ptr + @as(u64, id) * page, arena.ptr + @as(u64, live.ids.items[i]) * page, @intCast(page), e.stream.handle), "GLM page COW");
    }
    try e.stream.synchronize();
    for (indices[0..n], replacement.ids.items, 0..) |i, id, attempted| {
        put(e, st.table, i, id) catch |err| {
            // A failed upload can have written bytes. Repair ALL attempted entries before freeing replacements.
            // If even the repair cannot be proved, stop the process/peers with both owners still retained.
            for (indices[0 .. attempted + 1]) |old_i| put(e, st.table, old_i, live.ids.items[old_i]) catch |why| fatal(why);
            e.stream.synchronize() catch |why| fatal(why);
            return err;
        };
    }
    // Each reference is still held by exactly one of these leases. Swap owners only, then finish the temporary
    // lease's OLD pages. Pins and other branches keep them alive; st.pages aliases this canonical table in place.
    for (indices[0..n], replacement.ids.items) |i, *id| std.mem.swap(u32, &live.ids.items[i], id);
}

fn put(e: anytype, table: u64, i: usize, id: u32) !void {
    try e.d.check(e.d.api.cuMemcpyHtoD_v2(table + i * @sizeOf(u32), &id, @sizeOf(u32)), "GLM page COW table");
}

pub fn fatal(err: anyerror) noreturn {
    cuda.tp_watch.fail(@errorName(err));
    std.log.err("GLM page write stopped this rank and its peers: {s}", .{@errorName(err)});
    std.c._exit(1);
}
