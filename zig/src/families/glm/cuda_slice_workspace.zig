//! One reusable device capsule and timing event pair, allocated before live-pool planning.
const std = @import("std");
const cuda = @import("cuda");
const state = @import("cuda_state.zig");
const sparse = @import("cuda_sparse.zig");
const df = @import("dflash.zig");
const fatal = @import("cuda_slice_undo.zig").fatal;

pub const Workspace = struct {
    buffer: cuda.DeviceBuffer,
    events: [2]cuda.Event,
    ids: []u32,
    part: [state.prefill_rows + 1]u32 = undefined,
    leased: bool = false,

    /// Validate the supported profile once, before any serving request or pool allocation.
    pub fn validate(e: anytype) !void {
        if (e.mtp or e.cur.st.head) return error.SliceUnsupportedMtp;
        if (e.settings.layers != 0 or e.w.layers.len == 0 or e.w.layers.len != e.c.layers)
            return error.SliceUnsupportedThin;
        if (!e.settings.pages or e.cur.st.table == 0) return error.SliceNeedsPagedState;
        for (e.cur.st.caches) |x| if (x.pt != e.cur.st.table) return error.SliceNeedsPagedState;
    }

    /// Maximum copied payload: full fixed state/rings and a page tail plus two pooled lookahead rows per cache.
    pub fn capacity(e: anytype) !usize {
        const st = &e.cur.st;
        if (st.table < st.mtp_pos_dev or st.rings[0] < st.arena.buf.ptr) return error.SnapshotShape;
        var bytes: usize = @intCast(st.rings[0] - st.arena.buf.ptr + st.table - st.mtp_pos_dev);
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, 2 * st.caches.len, st.ring_rows * st.ring_row));
        for (st.caches) |x| {
            const row: usize = e.c.kv_lora * @as(usize, if (x.ls == 0) 2 else 1) + @as(usize, if (x.ls == 0) 0 else 4);
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, state.page_rows - 1, row));
            bytes = try std.math.add(usize, bytes, (state.page_rows / sparse.pool + 1) * @as(usize, e.c.i_dim) * 2);
        }
        if (e.drafter != null) bytes = try std.math.add(usize, bytes, df.ring_bytes);
        return bytes;
    }

    pub fn create(e: anytype) !*Workspace {
        try validate(e);
        const bytes = try capacity(e);
        const w = try e.gpa.create(Workspace);
        errdefer e.gpa.destroy(w);
        const ids = try e.gpa.alloc(u32, @import("cuda_pages.zig").tableLen(e.capacity));
        errdefer e.gpa.free(ids);
        var buf = try cuda.DeviceBuffer.alloc(e.d, bytes);
        errdefer buf.freeChecked() catch |err| fatal(err);
        var first = try cuda.Event.init(e.d, true);
        errdefer closeEvent(&first);
        var last = try cuda.Event.init(e.d, true);
        errdefer closeEvent(&last);
        // Materialize the capsule before unified-memory room is sampled; this is not a request-time reservation.
        buf.fill8(0, e.stream.handle) catch |err| fatal(err);
        e.stream.synchronize() catch |err| fatal(err);
        w.* = .{ .buffer = buf, .events = .{ first, last }, .ids = ids };
        return w;
    }

    pub fn resident(w: *const Workspace) usize {
        return w.buffer.len + w.ids.len * @sizeOf(u32) + @sizeOf(Workspace);
    }

    pub fn release(ptr: *anyopaque) void {
        const w: *Workspace = @ptrCast(@alignCast(ptr));
        if (!w.leased) fatal(error.SliceLease);
        w.leased = false;
    }

    pub fn destroy(w: *Workspace, gpa: std.mem.Allocator) void {
        if (w.leased) fatal(error.SliceLease);
        for (&w.events) |*ev| closeEvent(ev);
        w.buffer.freeChecked() catch |err| fatal(err);
        gpa.free(w.ids);
        gpa.destroy(w);
    }
};

fn closeEvent(ev: *cuda.Event) void {
    ev.d.check(ev.d.api.cuEventDestroy_v2(ev.handle), "slice event destroy") catch |err| fatal(err);
}
