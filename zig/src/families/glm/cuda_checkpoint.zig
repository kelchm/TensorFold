//! GLM's actual snapshot Plan as a bounded immutable source. No server path is enabled by this module.
const std = @import("std");
const core = @import("core");
const cap = core.checkpoint_capture;
const layout = @import("cuda_snapshot_layout.zig");
const state = @import("cuda_state.zig");
const Config = @import("config.zig").Config;
const max_parts = @typeInfo(@FieldType(layout.Plan, "parts")).array.len;
const max_layers = @import("config.zig").max_layers;

pub const Draft = union(enum) {
    none,
    dflash2: struct { live: u64, from: i64, through: usize },
};

pub const Frozen = struct {
    storage: cap.Location,
    len: usize,
    ctx: *anyopaque,
    release: *const fn (*anyopaque) void,
};

/// Owns frozen bytes on success; drains partial work on failure. The caller holds the sequence paused until return.
pub const Freezer = struct {
    ctx: *anyopaque,
    freeze: *const fn (*anyopaque, *Mutable, usize) anyerror!Frozen,
};

/// Only mutable ranges, packed into the capsule in Plan order. No array proportional to the prefix is built.
pub const Mutable = struct {
    plan: *const layout.Plan,
    immutable: *const [max_parts]usize,
    part: usize = 0,
    done: usize = 0,
    offset: usize = 0,

    pub fn next(m: *Mutable, limit: usize) !?cap.Chunk {
        if (limit == 0) return error.InvalidChunk;
        while (m.part < m.plan.count) {
            const p = m.plan.parts[m.part];
            m.done = @max(m.done, m.immutable[m.part]);
            if (m.done == p.len) {
                m.part += 1;
                m.done = 0;
                continue;
            }
            const piece = m.plan.piece(p, m.done);
            const n = @min(piece.len, limit);
            const chunk: cap.Chunk = .{ .location = .{ .device = piece.live }, .offset = m.offset, .len = n };
            m.done += n;
            m.offset += n;
            return chunk;
        }
        return null;
    }
};

/// Move-only descriptor; its host metadata and ids never borrow the sequence's mutable tables.
pub const Capture = struct {
    pages: cap.Lease,
    frozen: Frozen,
    plan: layout.Plan,
    immutable: [max_parts]usize,
    capsule_offsets: [max_parts]usize,
    pos: usize,
    mtp_len: usize,
    bases: [2]usize,
    cur: [max_layers]u1,
    cur_len: usize,
    ring_from: ?i64,

    /// All nonpaged state, tails and pooled lookahead require an explicit freezer, even at a page boundary.
    pub fn create(budget: *cap.Budget, live: *const core.kv_pool.Lease, st: *const state.State, c: *const Config, draft: Draft, freezer: ?Freezer) !Capture {
        if (st.pending != 0 or st.mtp_len > st.pos +| state.window_rows) return error.NotAtMark;
        if (st.table == 0) return error.UnsupportedUnpagedCapture;
        // A lagging head must not turn the bounded tail capsule into a copy of the target prefix.
        if (st.head and st.pos / state.page_rows > st.mtp_len / state.page_rows + 1) return error.UnsupportedHeadLag;
        for (st.caches) |x| if (x.pt != st.table) return error.UnsupportedUnpagedCapture;
        if (st.head and st.mtp_caches.pt != st.table) return error.UnsupportedUnpagedCapture;
        if (st.cur.len > max_layers or !std.mem.eql(u32, st.pages, live.ids.items)) return error.SnapshotShape;
        var plan = try layout.Plan.init(st, c, st.pos, if (st.head) st.mtp_len else null);
        const ring_from: ?i64 = switch (draft) {
            .none => null,
            .dflash2 => |d| blk: {
                if (d.from < 0 or d.through != st.pos or @as(u64, @intCast(d.from)) >= st.pos or
                    st.pos - @as(usize, @intCast(d.from)) > @import("dflash.zig").WIN) return error.InvalidDraftContext;
                try plan.outside(d.live, @import("dflash.zig").ring_bytes);
                break :blk d.from;
            },
        };
        // A physical page includes the head cache too. Tentative head rows cannot be leased as immutable.
        const completed = @min(st.pos, if (st.head) st.mtp_len else st.pos) / state.page_rows;
        var immutable: [max_parts]usize = @splat(0);
        var offsets: [max_parts]usize = @splat(0);
        var bytes: usize = 0;
        for (plan.parts[0..plan.count], 0..) |p, i| {
            if (p.span != 0) {
                if (p.page != budget.pool.page_bytes or p.span > p.page) return error.SnapshotShape;
                immutable[i] = @min(p.len, try std.math.mul(usize, completed, p.span));
            }
            offsets[i] = bytes;
            bytes = try std.math.add(usize, bytes, p.len - immutable[i]);
        }
        const f = freezer orelse return error.UnsupportedMutableCapture;
        var pages = try budget.acquire(live, completed, bytes);
        errdefer pages.close() catch unreachable;
        var mutable: Mutable = .{ .plan = &plan, .immutable = &immutable };
        const frozen = try f.freeze(f.ctx, &mutable, bytes);
        errdefer frozen.release(frozen.ctx);
        if (frozen.len != bytes or mutable.offset != bytes) return error.IncompleteFreeze;
        switch (frozen.storage) {
            .host => |h| if (h.len != bytes) return error.IncompleteFreeze,
            .device => |p| if (p == 0 or p > std.math.maxInt(u64) - bytes) return error.IncompleteFreeze,
        }
        // Only immutable pieces use Plan.piece after capture. The tail ids are no longer borrowed.
        plan.ids = pages.ids;
        var result: Capture = .{
            .pages = pages,
            .frozen = frozen,
            .plan = plan,
            .immutable = immutable,
            .capsule_offsets = offsets,
            .pos = st.pos,
            .mtp_len = st.mtp_len,
            .bases = .{ st.base, st.mtp_base },
            .cur = @splat(0),
            .cur_len = st.cur.len,
            .ring_from = ring_from,
        };
        @memcpy(result.cur[0..st.cur.len], st.cur);
        return result;
    }

    pub fn close(s: *Capture) !void {
        try s.pages.close();
        s.frozen.release(s.frozen.ctx);
    }
};

/// A streaming read cursor. The descriptor's page pins and capsule survive every borrowed ring.
pub const Cursor = struct {
    capture: *Capture,
    part: usize = 0,
    done: usize = 0,
    offset: usize = 0,

    pub fn source(c: *Cursor) cap.Source {
        return .{ .owner = &c.capture.pages, .ctx = c, .len = c.capture.plan.len, .next = next };
    }

    fn next(ptr: *anyopaque, limit: usize) !?cap.Chunk {
        const c: *Cursor = @ptrCast(@alignCast(ptr));
        const s = c.capture;
        if (s.pages.closed) return error.Closed;
        if (limit == 0) return error.InvalidChunk;
        while (c.part < s.plan.count) {
            const p = s.plan.parts[c.part];
            if (c.done == p.len) {
                c.part += 1;
                c.done = 0;
                continue;
            }
            const imm = s.immutable[c.part];
            var chunk: cap.Chunk = undefined;
            if (c.done < imm) {
                const piece = s.plan.piece(p, c.done);
                chunk = .{ .location = .{ .device = piece.live }, .offset = c.offset, .len = @min(limit, piece.len, imm - c.done) };
            } else {
                const n = @min(limit, p.len - c.done);
                chunk = .{ .location = try s.frozen.storage.slice(s.capsule_offsets[c.part] + c.done - imm, n), .offset = c.offset, .len = n };
            }
            c.done += chunk.len;
            c.offset += chunk.len;
            return chunk;
        }
        return null;
    }
};
