//! One engine's undo capsule; completed paged rows stay in their live owner's pages.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("cuda_checkpoint.zig");
const Workspace = @import("cuda_slice_workspace.zig").Workspace;
const layout = @import("cuda_snapshot_layout.zig");
const state = @import("cuda_state.zig");
const types = @import("cuda_engine_types.zig");
const df = @import("dflash.zig");
const max_layers = @import("config.zig").max_layers;
const max_parts = @typeInfo(@FieldType(layout.Plan, "parts")).array.len;

pub const Undo = struct {
    plan: layout.Plan,
    immutable: [max_parts]usize,
    frozen: checkpoint.Frozen,
    ids: []u32,
    seq: *types.Seq,
    pos: usize,
    mtp_len: usize,
    bases: [2]usize,
    parity: [max_layers]u1,
    prompt: bool,
    in_head: usize,
    rows_at: u64,
    draft: ?df.Slot,
    slot: u32,

    /// Every page stays owned by this slot until rollback/commit; release restores before returning its lease.
    pub fn capture(e: anytype) !Undo {
        const st = &e.cur.st;
        const workspace = e.slice_workspace orelse return error.SlicingNotStarted;
        if (workspace.leased) return error.SliceBusy;
        if (st.pending != 0) return error.NotAtMark;
        if (st.cur.len > max_layers or st.pos > st.capacity or st.ring_rows > state.index_ring) return error.SnapshotShape;
        for (st.caches) |x| if (x.pt != st.table) return error.SliceNeedsPagedState;
        var plan = try layout.Plan.init(st, &e.c, st.pos, null);
        // The full fixed rings, including unused rows, are bounded by index_ring, never by the prefix length.
        for (plan.parts[0..plan.count]) |*part| {
            for (st.rings[0 .. 2 * st.caches.len]) |ring| if (part.live == ring and part.span == 0) {
                const n = st.ring_rows * st.ring_row;
                plan.len += n - part.len;
                part.len = n;
            };
        }
        // At position zero Plan omits both rings. Include them explicitly before any layer can write one.
        if (st.pos == 0) for (st.rings[0 .. 2 * st.caches.len]) |ring| try plan.outside(ring, st.ring_rows * st.ring_row);
        var draft: ?df.Slot = null;
        if (e.drafter) |dr| {
            if (e.slot >= dr.slots.items.len) return error.SliceOwnerChanged;
            draft = dr.slots.items[e.slot];
            const ring = draft.?.ring orelse return error.SliceOwnerChanged;
            if (ring.ptr == 0 or ring.len != df.ring_bytes) return error.SliceOwnerChanged;
            try plan.outside(ring.ptr, df.ring_bytes);
        }
        var immutable: [max_parts]usize = @splat(0);
        var bytes: usize = 0;
        const completed = st.pos / state.page_rows;
        for (plan.parts[0..plan.count], 0..) |p, i| {
            if (p.span != 0) immutable[i] = @min(p.len, try std.math.mul(usize, completed, p.span));
            bytes = try std.math.add(usize, bytes, p.len - immutable[i]);
        }
        if (bytes > workspace.buffer.len or st.pages.len > workspace.ids.len) return error.SliceCapsuleUnbounded;
        const ids = workspace.ids[0..st.pages.len];
        @memcpy(ids, st.pages);
        plan.ids = ids;
        var mutable: checkpoint.Mutable = .{ .plan = &plan, .immutable = &immutable };
        errdefer e.stream.synchronize() catch |err| fatal(err);
        while (try mutable.next(2 << 20)) |chunk| {
            e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(workspace.buffer.ptr + chunk.offset, chunk.location.device, chunk.len, e.stream.handle), "slice capture copy") catch |err| fatal(err);
        }
        e.stream.synchronize() catch |err| fatal(err);
        if (mutable.offset != bytes) return error.IncompleteFreeze;
        workspace.leased = true;
        const frozen: checkpoint.Frozen = .{ .storage = .{ .device = workspace.buffer.ptr }, .len = bytes, .ctx = workspace, .release = Workspace.release };
        var result: Undo = .{
            .plan = plan,
            .immutable = immutable,
            .frozen = frozen,
            .ids = ids,
            .seq = e.cur,
            .pos = st.pos,
            .mtp_len = st.mtp_len,
            .bases = .{ st.base, st.mtp_base },
            .parity = @splat(0),
            .prompt = st.prompt,
            .in_head = e.cur.in_head,
            .rows_at = e.cur.rows_at,
            .draft = draft,
            .slot = e.slot,
        };
        @memcpy(result.parity[0..st.cur.len], st.cur);
        return result;
    }

    pub fn resident(u: *const Undo) usize {
        return u.frozen.len; // copied payload; the engine accounts its persistent workspace once
    }

    /// Copy only the captured mutable ranges; validate the retained owner before using any live address.
    pub fn restore(u: *Undo, e: anytype) !void {
        const st = &u.seq.st;
        if (!std.mem.eql(u32, st.pages, u.ids)) return error.SliceOwnerChanged;
        // Check every external owner before the first copy, including a nullable/replaced DFlash ring.
        if (u.draft) |saved| {
            const dr = e.drafter orelse return error.SliceOwnerChanged;
            if (u.slot >= dr.slots.items.len) return error.SliceOwnerChanged;
            const ring = dr.slots.items[u.slot].ring orelse return error.SliceOwnerChanged;
            const before = saved.ring orelse return error.SliceOwnerChanged;
            if (ring.ptr != before.ptr or ring.len != before.len) return error.SliceOwnerChanged;
        }
        var mutable: checkpoint.Mutable = .{ .plan = &u.plan, .immutable = &u.immutable };
        while (try mutable.next(2 << 20)) |chunk| {
            try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(chunk.location.device, u.frozen.storage.device + chunk.offset, chunk.len, e.stream.handle), "slice undo copy");
        }
        try st.restoreHost(e.stream, u.pos, u.mtp_len, u.parity[0..st.cur.len], u.bases);
        try e.stream.synchronize();
        st.prompt = u.prompt;
        u.seq.in_head = u.in_head;
        u.seq.rows_at = u.rows_at;
        if (u.draft) |saved| {
            e.drafter.?.slots.items[u.slot] = saved;
        }
    }

    pub fn close(u: *Undo, gpa: std.mem.Allocator) void {
        u.frozen.release(u.frozen.ctx);
        _ = gpa; // borrowed storage is returned to the engine, never freed per chunk
        u.* = undefined;
    }
};

/// A CUDA/collective failure has unknown completion: never let a request retry on that process.
pub fn fatal(err: anyerror) noreturn {
    cuda.tp_watch.fail(@errorName(err));
    std.log.err("sliced prefill stopped this rank and its peers: {s}", .{@errorName(err)});
    std.c._exit(1);
}

pub fn admissionError(err: anyerror) void {
    switch (err) {
        error.CudaFailed, error.CudaContextLost, error.NotReady, error.NotFound, error.NcclFailed => fatal(err),
        else => {},
    }
}
