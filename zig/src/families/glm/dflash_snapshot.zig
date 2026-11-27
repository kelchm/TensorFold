//! Bounded 640-row suffix of the ten modulo-WIN BF16 DFlash2 context arrays already projected by prompt ingest.
const std = @import("std");
pub const window: usize = 2048;
pub const bands: usize = 10;
pub const stride: usize = 1024 * 2;
pub const cap: usize = 640;
pub const ring_bytes: usize = bands * window * stride;

pub const History = struct {
    lo: usize = 0,
    end: usize = 0,

    pub fn append(h: *History, start: usize, n: usize) void {
        if (n == 0) return;
        const end = start + n;
        h.lo = @max(if (h.end == start) h.lo else start, end -| window);
        h.end = end;
    }

    pub fn rows(h: History, at: usize) usize {
        if (h.end != at or h.lo > at) return 0;
        return @min(cap, at - h.lo);
    }
};

pub const Saved = struct {
    end: usize = 0,
    count: usize = 0,

    pub fn valid(s: Saved, at: usize) bool {
        return s.count > 0 and s.count <= cap and s.count <= at and s.end == at;
    }
};

pub fn bytes(n: usize) usize {
    std.debug.assert(n <= cap);
    return n * bands * stride;
}

/// Append raw, checked device ranges to a snapshot plan after all target-state parts.
pub fn pack(plan: anytype, ring: anytype, at: usize, count: usize) !void {
    if (count == 0) return;
    if (count > cap or at < count or ring.ptr == 0 or ring.len != ring_bytes) return error.BadDraftRing;
    const slot = (at - count) % window;
    const first = @min(count, window - slot);
    for (0..bands) |band| {
        const base = band * window * stride;
        try span(plan, ring, base + slot * stride, first * stride);
        if (count > first) try span(plan, ring, base, (count - first) * stride);
    }
}

fn span(plan: anytype, ring: anytype, offset: usize, len: usize) !void {
    if (offset > ring.len or len > ring.len - offset or plan.count == plan.parts.len) return error.BadDraftRing;
    const address = try std.math.add(u64, ring.ptr, offset);
    _ = try std.math.add(u64, address, len);
    plan.len = try std.math.add(usize, plan.len, len);
    plan.parts[plan.count] = .{ .live = address, .len = len };
    plan.count += 1;
}
