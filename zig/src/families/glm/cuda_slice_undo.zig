//! One engine's undo capsule; completed paged rows stay in their live owner's pages.
const std = @import("std");
const cuda = @import("cuda");
const checkpoint = @import("cuda_checkpoint.zig");
const freezer = @import("cuda_checkpoint_freeze.zig");
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
        if (st.table == 0) return error.SliceNeedsPagedState;
        if (e.settings.layers != 0) return error.SliceUnsupportedThin;
        if (e.mtp or st.head) return error.SliceUnsupportedMtp;
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
            const ptr = try dr.ringPtr(e.slot);
            draft = dr.slots.items[e.slot];
            try plan.outside(ptr, df.ring_bytes);
        }
        var immutable: [max_parts]usize = @splat(0);
        var bytes: usize = 0;
        const completed = st.pos / state.page_rows;
        for (plan.parts[0..plan.count], 0..) |p, i| {
            if (p.span != 0) immutable[i] = @min(p.len, try std.math.mul(usize, completed, p.span));
            bytes = try std.math.add(usize, bytes, p.len - immutable[i]);
        }
        // The Plan contains only fixed state, ring windows and at most a page tail plus pooled lookahead.
        var bound: usize = @intCast(st.table - st.arena.buf.ptr);
        for (st.caches) |x| bound = try std.math.add(usize, bound, @intCast(2 * x.ps));
        if (draft != null) bound = try std.math.add(usize, bound, df.ring_bytes);
        if (bytes > bound) return error.SliceCapsuleUnbounded;
        const ids = try e.gpa.dupe(u32, st.pages);
        errdefer e.gpa.free(ids);
        plan.ids = ids;
        var mutable: checkpoint.Mutable = .{ .plan = &plan, .immutable = &immutable };
        var device: freezer.Device = .{ .gpa = e.gpa, .d = e.d, .stream = e.stream, .release_failure = fatal };
        const f = device.adapter();
        const frozen = try f.freeze(f.ctx, &mutable, bytes);
        errdefer frozen.release(frozen.ctx);
        if (mutable.offset != bytes or frozen.len != bytes) return error.IncompleteFreeze;
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
        return u.frozen.len + u.ids.len * @sizeOf(u32) + freezer.Device.descriptor_bytes;
    }

    /// Copy only the captured mutable ranges; validate the retained owner before using any live address.
    pub fn restore(u: *Undo, e: anytype) !void {
        const st = &u.seq.st;
        if (!std.mem.eql(u32, st.pages, u.ids)) return error.SliceOwnerChanged;
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
            const dr = e.drafter orelse return error.SliceOwnerChanged;
            if (u.slot >= dr.slots.items.len or dr.slots.items[u.slot].ring.?.ptr != saved.ring.?.ptr) return error.SliceOwnerChanged;
            dr.slots.items[u.slot] = saved;
        }
    }

    pub fn close(u: *Undo, gpa: std.mem.Allocator) void {
        u.frozen.release(u.frozen.ctx);
        gpa.free(u.ids);
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
