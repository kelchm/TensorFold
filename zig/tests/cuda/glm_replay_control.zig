//! Host replay accounting, pool reservations and per-forward path checks.
const std = @import("std");

pub const slots = 4;
pub const page_rows = 512;
pub const Part = struct { stream_id: u32, position: usize, tape_offset: usize, offered: usize, kept: usize };
pub const Forward = struct { forward_id: u64, parts: []const Part };
pub const Stream = struct { prompt: []const u32, tape: []const u32, owner: u8 };
pub const Cursor = struct {
    positions: [slots]usize,
    offsets: [slots]usize = @splat(0),
    next: u64 = 0,
    pending: bool = false,

    pub fn init(streams: [slots]Stream) Cursor {
        var c: Cursor = .{ .positions = undefined };
        for (streams, 0..) |s, i| c.positions[i] = s.prompt.len;
        return c;
    }

    pub fn check(c: *const Cursor, f: Forward, streams: [slots]Stream) !void {
        if (c.pending) return error.IngestPending;
        if (f.forward_id != c.next) return error.ForwardOrder;
        if (f.parts.len == 0 or f.parts.len > slots) return error.PartCount;
        var seen: [slots]bool = @splat(false);
        var rows: usize = 0;
        for (f.parts) |p| {
            if (p.stream_id >= slots) return error.StreamId;
            const i = p.stream_id;
            if (seen[i]) return error.DuplicateStream;
            seen[i] = true;
            if (p.offered == 0 or p.offered > 16 or p.kept == 0 or p.kept > p.offered) return error.RowCount;
            rows += p.offered;
            if (rows > 32) return error.SharedWidth;
            if (p.position != c.positions[i] or p.tape_offset != c.offsets[i]) return error.StalePosition;
            const s = streams[i];
            if (p.tape_offset > s.tape.len or p.offered > s.tape.len - p.tape_offset) return error.ShortTape;
            if (p.kept >= s.tape.len - p.tape_offset) return error.MissingAnchor;
        }
    }

    pub fn submitted(c: *Cursor, f: Forward, streams: [slots]Stream) !void {
        try c.check(f, streams);
        c.pending = true;
    }

    pub fn ingested(c: *Cursor, f: Forward) !void {
        if (!c.pending or f.forward_id != c.next) return error.DuplicateIngest;
        for (f.parts) |p| {
            c.positions[p.stream_id] += p.kept;
            c.offsets[p.stream_id] += p.kept;
        }
        c.next += 1;
        c.pending = false;
    }
};

pub const Budget = struct {
    half_pages: usize,
    logical_pages: usize,
    page_argument: usize,
    reservation: [slots]usize,
    owner_pages: [2]usize,
    twin_headroom: usize,
};

pub fn budget(streams: [slots]Stream, forwards: []const Forward, capacity: usize, sharded: bool, half: usize) !Budget {
    if (capacity < 2 or capacity > std.math.maxInt(u32)) return error.Capacity;
    var c = Cursor.init(streams);
    var peak = c.positions;
    var b: Budget = .{ .half_pages = half, .logical_pages = 0, .page_argument = 0, .reservation = undefined, .owner_pages = @splat(0), .twin_headroom = 0 };
    if (half > (std.math.maxInt(usize) - 4) / 2) return error.PoolOverflow;
    b.logical_pages = 2 * half;
    b.page_argument = b.logical_pages + @as(usize, if (sharded) 4 else 0);
    if (half < pages(@max(131072, capacity / 4))) return error.PartFloor;
    for (streams) |s| {
        if (s.owner > 1 or s.prompt.len == 0) return error.StreamConfig;
        if (s.prompt.len > capacity) return error.Capacity;
        b.twin_headroom = @max(b.twin_headroom, pages(s.prompt.len));
    }
    for (forwards) |f| {
        try c.submitted(f, streams);
        for (f.parts) |p| {
            if (p.offered > capacity -| p.position) return error.Capacity;
            peak[p.stream_id] = @max(peak[p.stream_id], p.position + p.offered);
        }
        try c.ingested(f);
    }
    for (streams, 0..) |s, i| {
        const reserved = pages(peak[i] + 16);
        if (reserved > pages(capacity)) return error.Capacity;
        b.reservation[i] = reserved;
        b.owner_pages[s.owner] += reserved;
    }
    for (b.owner_pages) |used| if (used > half or b.twin_headroom > half - used) return error.PoolHeadroom;
    return b;
}

pub fn pages(n: usize) usize {
    return (n + page_rows - 1) / page_rows;
}

pub const Counters = struct { owner_layers: u64 = 0, owner_rows: u64 = 0, owner_bytes: u64 = 0, calls: u64 = 0, lost: u64 = 0, carries: u64 = 0, twins: u64 = 0 };
pub const Geometry = struct { layers: u64, heads: u64, latent: u64 };

pub fn checkCounters(before: Counters, after: Counters, rows: u64, sharded: bool, g: Geometry) !void {
    if (after.calls != before.calls) return error.ProposerCalled;
    if (after.lost != before.lost) return error.ContextLost;
    if (after.carries != before.carries or after.twins != 0) return error.PrimaryState;
    if (g.layers == 0 or g.heads == 0 or g.latent == 0) return error.Geometry;
    const layers = if (sharded) g.layers else 0;
    const owner_rows = layers * rows;
    const bytes = owner_rows * g.heads * g.latent * 2;
    if (after.owner_layers -| before.owner_layers != layers or after.owner_rows -| before.owner_rows != owner_rows or after.owner_bytes -| before.owner_bytes != bytes) return error.OwnerCounters;
}
