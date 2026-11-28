//! Prompt-sized snapshots on each rank; the existing chunk grid and the live sequence's layout do not change.
const std = @import("std");
const cuda = @import("cuda");
const types = @import("cuda_engine_types.zig");
pub const pages = @import("cuda_snapshot_pages.zig");
pub const layout = @import("cuda_snapshot_layout.zig");

pub const Setup = struct { ptr: *anyopaque, run: *const fn (*anyopaque, u64) anyerror!void };

pub fn prepare(e: anytype, budget: u64) !void {
    if (e.local.len > 0 and budget > 0) return error.SharesKeepNoStates; // a state copy holds one share's caches
    if (e.snap_pages != null or e.snaps.count() != 0) return error.SnapshotCacheMade;
    const count = std.math.cast(u32, budget / pages.page) orelse return error.SnapshotBudget;
    try e.snaps.ensureTotalCapacity(e.gpa, count);
    try e.dropped.ensureTotalCapacity(e.gpa, @as(usize, count) + 1);
    e.snap_pages = try pages.Pool.init(e.gpa, e.d, budget, try pages.onHost());
}

/// A failed allocation still joins the startup agreement, so a peer never waits for a missing participant.
pub fn agree(e: anytype, budget: u64) !void {
    var failed: ?anyerror = null;
    e.prepareSnapshots(budget) catch |err| {
        failed = err;
    };
    if (try e.shared(if (failed != null) 1 else 0) > 0) return failed orelse error.PeerSnapshotCache;
}

const draft = @import("dflash_snapshot.zig");

fn draftEnabled(e: anytype) bool {
    const T = @typeInfo(@TypeOf(e)).pointer.child;
    if (!@hasField(T, "settings") or !@hasField(T, "drafter")) return false;
    return e.settings.dflash_snap_context and !e.share_pages and e.drafter != null;
}

fn currentDraft(e: anytype, at: usize) draft.Saved {
    if (!draftEnabled(e)) return .{};
    const dr = e.drafter.?;
    if (e.slot >= dr.slots.items.len) return .{};
    const sl = &dr.slots.items[e.slot];
    const ring = sl.ring orelse return .{};
    if (ring.ptr == 0 or ring.len != draft.ring_bytes) return .{};
    return .{ .end = at, .count = sl.prompt_history.rows(at) };
}

fn currentBelow(e: anytype, at: usize) u64 {
    const C = @TypeOf(e.cur);
    const S = if (@typeInfo(C) == .pointer) @typeInfo(C).pointer.child else C;
    if (!@hasField(S, "cuts")) return 0;
    const cuts = @import("cuda_prompt_cuts.zig");
    if (@hasField(S, "track_executed") and e.cur.track_executed)
        return cuts.belowExecuted(e.cur.resumed_starts.items, e.cur.resumed_at, e.cur.cuts.items, @intCast(at));
    return cuts.below(e.cur.cuts.items, @intCast(at));
}

fn targetLen(e: anytype, at: usize) ?usize {
    const head: ?usize = if (e.mtp) @min(e.capacity, std.math.add(usize, at, @import("cuda_state.zig").window_rows) catch return null) else null;
    // a paged sequence holds its request's rows: its prefix is counted to the capacity, in a window's bytes
    return layout.Plan.size(&e.cur.st, &e.c, e.capacity, at, head, heldPages(e, at)) catch null;
}

/// Target-only rounded pages for `at`, excluding any RAM-only DFlash2 draft tail (used by disk restores).
pub fn targetBytes(e: anytype, at: usize) u64 {
    const n = targetLen(e, at) orelse return std.math.maxInt(u64);
    return pages.rounded(n) catch std.math.maxInt(u64);
}

pub fn bytes(e: anytype, at: usize) u64 {
    const n = targetLen(e, at) orelse return std.math.maxInt(u64);
    // Reservation and promise callers can be planning another stream. Only keepAs knows its valid rows.
    const rows = if (draftEnabled(e)) @min(at, draft.cap) else 0;
    return pages.rounded(std.math.add(usize, n, draft.bytes(rows)) catch return std.math.maxInt(u64)) catch std.math.maxInt(u64);
}

pub fn keep(e: anytype) !u64 {
    if (e.next_snap > std.math.maxInt(u32)) return error.SnapshotIdLimit;
    const id = e.next_snap;
    try keepAs(e, id);
    e.next_snap += 1;
    e.announce = id;
    return id;
}

pub fn keepAs(e: anytype, id: u64) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    if (e.cur.st.pending != 0 or e.cur.st.mtp_len > e.cur.st.pos + @import("cuda_state.zig").window_rows)
        return error.NotAtMark;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    const profiled = e.prompt_profs[0] != null;
    const started: i96 = if (profiled) std.Io.Clock.awake.now(e.io).toNanoseconds() else 0;
    var plan = try layout.Plan.init(&e.cur.st, &e.c, e.cur.st.pos, if (e.mtp) e.cur.st.mtp_len else null, heldPages(e, e.cur.st.pos));
    const target_len = plan.len;
    const saved = currentDraft(e, e.cur.st.pos);
    if (saved.count != 0) try draft.pack(&plan, e.drafter.?.slots.items[e.slot].ring.?, saved.end, saved.count);
    var lease = try pool.take(plan.len);
    errdefer {
        cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot copy did not drain");
        pool.give(&lease);
    }
    const cur = try e.gpa.dupe(u1, e.cur.st.cur);
    errdefer e.gpa.free(cur);
    try pool.move(e.stream, &lease, &plan, true);
    try e.stream.synchronize();
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = target_len, .packed_len = plan.len, .draft = saved, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur, .bases = .{ e.cur.st.base, e.cur.st.mtp_base }, .shared = plan.shared, .below = currentBelow(e, e.cur.st.pos) });
    if (saved.count != 0) std.log.info("glm draft snapshot keep rank={d} slot={d} id={d} pos={d} rows={d} bytes={d}", .{ e.share.rank, e.slot, id, saved.end, saved.count, draft.bytes(saved.count) });
    if (profiled) {
        const wall_ms: f64 = @as(f64, @floatFromInt(std.Io.Clock.awake.now(e.io).toNanoseconds() - started)) / 1e6;
        std.log.info("glm prof snapshot rank={d} slot={d} id={d} pos={d} bytes={d} parts={d} wall_ms={d:.4}", .{ e.share.rank, e.slot, id, e.cur.st.pos, plan.len, plan.count, wall_ms });
    }
}

/// A disk read's state: `at`'s pages leased under rank 0's id on each rank, refused by `restore` until `seal`.
pub fn blankAs(e: anytype, id: u64, at: usize) !void {
    if (e.snaps.contains(id)) return error.StateKept;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    var lease = try pool.take(targetBytes(e, at));
    errdefer pool.give(&lease);
    const cur = try e.gpa.alloc(u1, e.cur.st.cur.len);
    @memset(cur, 0);
    e.snaps.putAssumeCapacityNoClobber(id, .{ .lease = lease, .len = 0, .pos = at, .mtp_len = 0, .cur = cur, .bases = .{ 0, 0 } });
}

/// Whether an index ring of `rows` rows based at `base` holds every row up to `len`, as the image's ring bytes do.
fn ringHolds(base: usize, len: usize, rows: usize) bool {
    return base <= len and len - base <= rows;
}

/// A blank state made whole from its image's words (its bytes are in or follow): a restore takes it from here.
pub fn seal(e: anytype, id: u64, mtp_len: usize, bases: [2]usize, cur: u64, below: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.len != 0) return error.StateKept;
    if (s.cur.len > 64 or (s.cur.len < 64 and cur >> @intCast(s.cur.len) != 0)) return error.DiskWords;
    const head = if (e.mtp) @min(e.capacity, s.pos + @import("cuda_state.zig").window_rows) else 0;
    if (mtp_len > head) return error.DiskWords;
    const rows = e.cur.st.ring_rows;
    if (!ringHolds(bases[0], s.pos, rows) or !ringHolds(bases[1], mtp_len, rows)) return error.DiskWords;
    const n = try layout.Plan.size(&e.cur.st, &e.c, e.capacity, s.pos, if (e.mtp) mtp_len else null, s.shared);
    if (n == 0 or n > s.lease.ids.items.len * pages.page) return error.SnapshotSize;
    for (s.cur, 0..) |*b, i| b.* = @intCast(cur >> @intCast(i) & 1);
    s.len = n;
    s.packed_len = n;
    s.mtp_len = mtp_len;
    s.bases = bases;
    s.below = below;
}

pub fn restore(e: anytype, id: u64) !void {
    const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
    if (s.len == 0) return error.UnsealedState;
    const pool = if (e.snap_pages) |*p| p else return error.NoSnapshotCache;
    // the kept state's own parity picks the half a live-half keep carries; the slot's dead half stays as it is
    var plan = try layout.Plan.initFor(&e.cur.st, &e.c, s.pos, if (e.mtp) s.mtp_len else null, s.shared, s.cur);
    if (plan.len != s.len) return error.SnapshotSize;
    // Publish the draft floor only after every transfer finishes; a missing tail leaves a cold drafter.
    var restored: usize = 0;
    if (@hasField(@typeInfo(@TypeOf(e)).pointer.child, "drafter")) {
        if (e.drafter) |dr| {
            try dr.restart(e.slot, s.pos);
            if (draftEnabled(e) and s.draft.valid(s.pos)) {
                const ring = dr.slots.items[e.slot].ring.?;
                const lease_bytes = s.lease.ids.items.len * pages.page;
                if (ring.ptr != 0 and ring.len == draft.ring_bytes and s.len <= lease_bytes and s.packed_len >= s.len and s.packed_len <= lease_bytes and draft.bytes(s.draft.count) == s.packed_len - s.len) {
                    try draft.pack(&plan, ring, s.pos, s.draft.count);
                    restored = s.draft.count;
                }
            }
        }
    }
    errdefer cuda.completion.drain(e.d, e.stream, "a failed GLM snapshot restore did not drain");
    try pool.move(e.stream, &s.lease, &plan, false);
    try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur, s.bases);
    e.cur.in_head = 0;
    try e.stream.synchronize();
    if (@hasField(@typeInfo(@TypeOf(e)).pointer.child, "drafter")) {
        if (draftEnabled(e)) std.log.info("glm draft snapshot restore rank={d} slot={d} id={d} pos={d} rows={d}", .{ e.share.rank, e.slot, id, s.pos, restored });
        if (restored != 0) {
            const sl = &e.drafter.?.slots.items[e.slot];
            sl.from = @intCast(s.pos - restored);
            sl.prompt_history = .{ .lo = s.pos - restored, .end = s.pos };
        }
    }
}

/// The pages below `at` the shared prompt store holds (TF_KV_SHARE): a kept state's copy leaves them out.
fn heldPages(e: anytype, at: usize) usize {
    return if (e.share_pages) at / @import("cuda_state.zig").page_rows else 0;
}

pub fn holds(e: anytype, id: u64) bool {
    return e.snaps.contains(id);
}

pub fn drop(e: anytype, id: u64) void {
    var kv = e.snaps.fetchRemove(id) orelse return;
    free(e, &kv.value);
    if (e.share.rank == 0 and e.announce == id) {
        e.announce = 0; // the store failed to publish this keep, so no follower has it yet
        return;
    }
    if (e.share.rank == 0 and e.share.world > 1) e.dropped.appendAssumeCapacity(@intCast(id));
}

pub fn free(e: anytype, s: *types.Snap) void {
    e.snap_pages.?.give(&s.lease);
    e.gpa.free(s.cur);
}

/// Odd words release an evicted id before an even word announces its replacement; an idle gate still sends one word.
pub fn gate(e: anytype) !void {
    const leader = e.share.rank == 0;
    var dropped: usize = 0;
    while (true) {
        const word: u64 = if (!leader) 0 else if (dropped < e.dropped.items.len)
            2 * @as(u64, e.dropped.items[dropped]) + 1
        else
            2 * e.announce;
        const shared = try e.shared(@floatFromInt(word));
        if (!std.math.isFinite(shared) or shared < 0 or shared > 2 * @as(f64, std.math.maxInt(u32)) + 1 or @floor(shared) != shared)
            return error.BadSnapshotControl;
        const got: u64 = @intFromFloat(shared);
        if (got & 1 != 0) {
            if (leader) dropped += 1 else e.dropState(got / 2);
            continue;
        }
        if (leader) {
            e.dropped.clearRetainingCapacity();
            e.announce = 0;
        } else if (got > 0) try e.keepAs(got / 2);
        return;
    }
}

/// After all ranks agree to cancel, publish the last completed state before any rank releases or rebinds its slot.
pub fn interrupted(e: anytype, marks: ?types.Marks, at: u32) !void {
    if (marks) |m| if (m.interrupted) |save_prefix| save_prefix(m.ctx, at);
    try gate(e);
}
