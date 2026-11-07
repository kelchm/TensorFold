//! Bounded checkpoint ownership, separate from RAM-cache admission. Calls run on the pool's owning thread.
const std = @import("std");
const kv = @import("kv_pool.zig");

pub const Limits = struct { descriptors: usize, unique_pages: usize, pages_per_descriptor: usize, capsule_bytes: usize };

/// One admission domain per pool. Capsule capacity is reserved before any freezer allocates or copies bytes.
pub const Budget = struct {
    pool: *kv.Pool,
    limits: Limits,
    active: usize = 0,
    capsule_used: usize = 0,

    /// The family proves these leading pages immutable at a committed mark, including every cache in each page.
    pub fn acquire(b: *Budget, live: *const kv.Lease, completed: usize, capsule_bytes: usize) !Lease {
        if (completed > live.ids.items.len) return error.InvalidPages;
        if (b.active >= b.limits.descriptors or completed > b.limits.pages_per_descriptor or
            capsule_bytes > b.limits.capsule_bytes - b.capsule_used) return error.CheckpointBackpressure;
        const ids = live.ids.items[0..completed];
        var added: usize = 0;
        for (ids, 0..) |id, i| {
            if (id >= b.pool.total() or b.pool.leased[id] == 0) return error.InvalidPages;
            if (b.pool.refs[id] == std.math.maxInt(u32)) return error.HolderLimit;
            // A lease maps a physical page once. Reject malformed tables before changing any owner.
            if (std.mem.indexOfScalar(u32, ids[0..i], id) != null) return error.InvalidPages;
            added += @intFromBool(b.pool.pins[id] == 0);
        }
        if (added > b.limits.unique_pages -| b.pool.npinned) return error.CheckpointBackpressure;
        const owned = try b.pool.gpa.dupe(u32, ids);
        for (owned) |id| b.pool.pinCheckpoint(id);
        b.active += 1;
        b.capsule_used += capsule_bytes;
        return .{ .budget = b, .ids = owned, .capsule_bytes = capsule_bytes };
    }
};

/// Move-only owner. A stream borrows it until all submitted DMA and I/O have actually completed or drained.
pub const Lease = struct {
    budget: *Budget,
    ids: []const u32,
    capsule_bytes: usize,
    readers: usize = 0,
    closed: bool = false,

    pub fn borrow(l: *Lease) !void {
        if (l.closed) return error.Closed;
        l.readers = try std.math.add(usize, l.readers, 1);
    }

    pub fn unborrow(l: *Lease) void {
        std.debug.assert(l.readers > 0);
        l.readers -= 1;
    }

    pub fn close(l: *Lease) !void {
        if (l.closed) return error.Closed;
        if (l.readers != 0) return error.CheckpointBusy;
        const b = l.budget;
        for (l.ids) |id| b.pool.unpinCheckpoint(id);
        b.pool.gpa.free(l.ids);
        l.ids = &.{};
        b.active -= 1;
        b.capsule_used -= l.capsule_bytes;
        l.closed = true;
    }
};

/// Immutable backing storage, either device pages or an independently owned frozen capsule.
pub const Location = union(enum) {
    device: u64,
    host: []const u8,

    pub fn slice(s: Location, off: usize, len: usize) !Location {
        return switch (s) {
            .device => |p| .{ .device = try std.math.add(u64, p, off) },
            .host => |b| if (off > b.len or len > b.len - off) error.InvalidChunk else .{ .host = b[off..][0..len] },
        };
    }
};

pub const Chunk = struct { location: Location, offset: u64, len: usize };

/// A cursor over one owned descriptor. No full-prefix buffer or per-chunk allocation is required.
pub const Source = struct {
    owner: *Lease,
    ctx: *anyopaque,
    len: u64,
    next: *const fn (*anyopaque, usize) anyerror!?Chunk,
};

test {
    _ = @import("checkpoint_capture_test.zig");
}
