//! Protect queued resumptions, retain the next endpoint, and trim optional copies to the remaining room.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const images = @import("cache_images.zig");
const others = @import("prompt_cache_others.zig");
const policy = @import("prompt_cache_policy.zig");
pub const Room = struct { bytes: u64 = 0, slots: usize = 0 };
pub const Guard = struct {
    ctx: *anyopaque,
    needed: *const fn (*anyopaque, *const pc.Entry) bool,
    exact: bool = false, // true: the callback already selects each job's deepest match
    promised: Room = .{}, // net room promised to other admitted passes
    promise: ?*Room = null, // this lookup's net promise, owned by its job

    pub fn protects(g: Guard, s: *const pc.Store, e: *const pc.Entry) bool {
        if (!g.needed(g.ctx, e)) return false;
        if (g.exact) return true;
        for (s.by_position.items) |o| {
            if (o.at <= e.at) break;
            if (!o.expired and o.tokens.len >= e.tokens.len and o.image_frame == e.image_frame and
                std.mem.eql(u32, o.tokens[0..e.tokens.len], e.tokens) and
                modes.equal(o.decode_spans, e.decode_spans, e.at) and g.needed(g.ctx, o)) return false;
        }
        return true;
    }
};

fn exists(s: *const pc.Store, prompt: []const u32, at: u32, spans: []const modes.Span) bool {
    const n = @as(usize, at) + s.rules.lookahead;
    for (s.entries.items) |e| {
        if (!e.expired and e.at == at and e.tokens.len == n and n <= prompt.len and
            std.mem.eql(u32, e.tokens, prompt[0..n]) and modes.equal(e.decode_spans, spans, at) and images.matches(s, e))
            return true;
    }
    return false;
}

/// The current restore funds its replacement only when no other borrower or branch will still need it.
fn reclaimable(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, born: u32) Room {
    const e = from orelse return .{};
    if (e.borrowers > 0 or !policy.redundant(s, e, prompt, born)) return .{};
    if (policy.continuedLater(s, e) and !std.mem.eql(u32, prompt, e.last)) return .{};
    if (s.reservation_guard) |g| if (g.protects(s, e)) return .{};
    return .{ .bytes = e.bytes, .slots = 1 };
}

/// Bytes and slots the victim order can eventually free, excluding borrowed, protected and skipped entries.
pub fn evictable(s: *const pc.Store, skip: ?*pc.Entry, prompt: []const u32) Room {
    var out: Room = .{};
    for (s.entries.items) |e| {
        if (e.borrowers > 0 or e == skip) continue;
        if (s.reservation_guard) |g| if (g.protects(s, e)) continue;
        if (s.keep_others and !policy.own(e, skip, prompt)) continue;
        out.bytes += e.bytes;
        out.slots += 1;
    }
    return out;
}

fn future(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, born: u32) Room {
    const credit = reclaimable(s, from, prompt, born);
    const promised: Room = if (s.reservation_guard) |g| g.promised else .{};
    return .{ .bytes = (s.budget -| s.held) +| credit.bytes -| promised.bytes, .slots = (s.most -| s.entries.items.len) +| credit.slots -| promised.slots };
}

/// The primary history mark follows the same warm/lookahead and short-prompt cuts as marksRewind.
pub fn endpointFor(s: *const pc.Store, prompt: []const u32, from: u32, history: u32, starts: []const u32, spans: []const modes.Span) u32 {
    var at = history;
    if (s.rules.warm != .off and history == prompt.len and s.rules.lookahead > 0)
        at = @intCast(prompt.len -| s.rules.lookahead);
    if (!s.rules.planned and prompt.len < s.rules.min_prompt and s.rules.chunk > 0)
        return @import("prompt_cuts.zig").chunkEnd(s.rules.chunk, from, at, prompt.len, starts, spans);
    if (history == 0 and s.rules.tail) at = @intCast(prompt.len -| @max(1, s.rules.lookahead));
    if (!s.rules.planned) return at;
    if (starts.len == 0 and s.rules.grid > 0) return at - at % s.rules.grid;
    return @import("prompt_cuts.zig").floorStart(starts, at);
}

fn priority(marks: []const u32, endpoint: u32) ?u32 {
    if (std.mem.indexOfScalar(u32, marks, endpoint) != null) return endpoint;
    return null;
}

/// A guarded reservation makes room for the endpoint alone; other copies cannot displace a newest state.
pub fn reserve(s: *pc.Store, prompt: []const u32, born: u32, from: ?*pc.Entry, marks: []const u32, spans: []const modes.Span, endpoint: u32) void {
    if (s.reservation_guard != null) {
        const at = priority(marks, endpoint) orelse return;
        if (exists(s, prompt, at, spans)) return;
        const need = s.family.vtable.bytes(s.family.ptr, at);
        if (need > s.budget or s.most == 0) return;
        const credit = reclaimable(s, from, prompt, born);
        const promised = s.reservation_guard.?.promised;
        // Preserve deficits: freeing a state must pay the other jobs' promises before this endpoint can fit.
        var bytes: i128 = @as(i128, s.budget) - s.held + credit.bytes - promised.bytes;
        var slots: i128 = @as(i128, s.most) - s.entries.items.len + credit.slots - promised.slots;
        const can = evictable(s, from, prompt);
        if (bytes + can.bytes < need or slots + can.slots < 1) return;
        while (bytes < need or slots < 1) {
            const native = others.victim(s, from, prompt, born, need, 1) orelse return;
            const missing: u64 = @intCast(@max(0, @as(i128, need) - bytes));
            const i = if (s.eviction == .single_leaf) singleLeafVictim(s, from, prompt, born, missing, native) else native;
            bytes += s.entries.items[i].bytes;
            slots += 1;
            s.remove(i);
            s.counts.evicted += 1;
        }
        return;
    }
    var need: u64 = 0;
    var new: usize = 0;
    for (marks) |at| if (!exists(s, prompt, at, spans)) {
        need += s.family.vtable.bytes(s.family.ptr, at);
        new += 1;
    };
    while (s.held + need > s.budget or s.entries.items.len + @min(new, s.most) > s.most) {
        const i = others.victim(s, from, prompt, born, need, @min(new, s.most)) orelse return;
        s.remove(i);
        s.counts.evicted += 1;
    }
}

/// A byte shortage can evict a reused state and then a second victim. In the LRU fallback,
/// use one eligible, never-resumed leaf when it alone funds the endpoint. Keep native tiers,
/// the restore source, other conversations' restrictions and every queued reservation intact.
fn singleLeafVictim(s: *const pc.Store, from: ?*pc.Entry, prompt: []const u32, born: u32, missing: u64, native: usize) usize {
    const first = s.entries.items[native];
    if (first.bytes >= missing or first.seen.hits == 0 or policy.redundant(s, first, prompt, born)) return native;
    var candidate: ?usize = null;
    for (s.entries.items, 0..) |e, i| {
        if (e == from or e.borrowers != 0 or e.bytes < missing or e.seen.hits != 0) continue;
        if (e.shared or e.branched or e.rewind or e.expired or policy.extended(s, e)) continue;
        if (s.reservation_guard.?.protects(s, e) or (s.keep_others and !policy.own(e, from, prompt))) continue;
        if (candidate == null or e.used < s.entries.items[candidate.?].used) candidate = i;
    }
    return candidate orelse native;
}

/// The endpoint first; interior, rewind and extra shared cuts compete for the room after it, ladder rungs last.
pub fn fitting(s: *pc.Store, a: std.mem.Allocator, prompt: []const u32, marks: []const u32, spans: []const modes.Span, from: ?*pc.Entry, born: u32, endpoint: u32) ![]const u32 {
    if (s.reservation_guard == null) return marks;
    defer a.free(marks);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var room = future(s, from, prompt, born);
    const end = priority(marks, endpoint);
    if (end) |at| {
        const bytes = if (exists(s, prompt, at, spans)) 0 else s.family.vtable.bytes(s.family.ptr, at);
        if (bytes == 0 or (bytes <= room.bytes and room.slots > 0)) {
            try out.append(a, at);
            room.bytes -= bytes;
            if (bytes > 0) room.slots -= 1;
        }
    }
    for ([_]bool{ false, true }) |rungs| for (marks) |at| { // ladder checkpoints take only what every other mark leaves
        if (end == at or @import("prompt_intermediate.zig").is(s.rules, prompt.len, at) != rungs) continue;
        const bytes = if (exists(s, prompt, at, spans)) 0 else s.family.vtable.bytes(s.family.ptr, at);
        if (bytes == 0 or (bytes <= room.bytes and room.slots > 0)) {
            try out.append(a, at);
            room.bytes -= bytes;
            if (bytes > 0) room.slots -= 1;
        }
    };
    const fitted = try out.toOwnedSlice(a);
    s.counts.trimmed += marks.len - fitted.len;
    if (s.reservation_guard.?.promise) |promise| {
        var used: Room = .{};
        for (fitted) |at| if (!exists(s, prompt, at, spans)) {
            used.bytes +|= s.family.vtable.bytes(s.family.ptr, at);
            used.slots +|= 1;
        };
        const credit = reclaimable(s, from, prompt, born);
        promise.* = .{ .bytes = used.bytes -| credit.bytes, .slots = used.slots -| credit.slots };
    }
    std.mem.sort(u32, fitted, {}, std.sort.asc(u32));
    return fitted;
}

test "single leaf: byte deficit, full payment, policy tiers, queue promises and fallback" {
    const t = std.testing;
    var words = [_]u32{ 3, 4, 5, 6 };
    var enough = [_]u32{ 7, 0 };
    const prompt = [_]u32{ 9, 10, 11 };
    const first: pc.Entry = .{ .tokens = words[0..1], .last = words[0..1], .at = 1, .born = 1, .used = 3, .seen = .{ .hits = 5 }, .bytes = 472, .saved = undefined };
    var states: [4]pc.Entry = @splat(first);
    // Both large leaves fit: prefer the least recently used of them.
    for (states[1..], 0..) |*e, i| {
        e.tokens = enough[0..1];
        e.seen.hits = 0;
        e.used = 8 + i;
        e.bytes = 1830;
    }
    states[1].bytes = 1425;
    var guard: ?*const pc.Entry = null;
    const Gate = struct {
        fn needed(ctx: *anyopaque, e: *const pc.Entry) bool {
            const protected: *?*const pc.Entry = @ptrCast(@alignCast(ctx));
            return protected.* == e;
        }
    };
    var s: pc.Store = .{ .gpa = t.allocator, .family = undefined, .rules = .{}, .budget = 8192, .reservation_guard = .{ .ctx = @ptrCast(&guard), .needed = Gate.needed, .exact = true } };
    defer s.entries.deinit(t.allocator);
    for (&states) |*e| try s.entries.append(t.allocator, e);
    try t.expectEqual(@as(usize, 2), singleLeafVictim(&s, null, &prompt, 3, 1426, 0));
    try t.expectEqual(@as(usize, 1), singleLeafVictim(&s, null, &prompt, 3, 1425, 0));
    try t.expectEqual(@as(usize, 0), singleLeafVictim(&s, null, &prompt, 3, 472, 0));
    try t.expectEqual(@as(usize, 0), singleLeafVictim(&s, null, &prompt, 3, 1831, 0));
    try t.expectEqual(@as(usize, 3), singleLeafVictim(&s, &states[2], &prompt, 3, 1426, 0));
    guard = &states[3];
    for (0..7) |i| {
        var e = &states[2];
        const save = e.*;
        switch (i) {
            0 => e.branched = true,
            1 => e.shared = true,
            2 => e.rewind = true,
            3 => e.borrowers = 1,
            4 => e.seen.hits = 1,
            5 => e.expired = true,
            else => try e.ext.append(t.allocator, &states[1]),
        }
        try t.expectEqual(@as(usize, 0), singleLeafVictim(&s, null, &prompt, 3, 1426, 0));
        e.ext.deinit(t.allocator);
        e.* = save;
    }
    s.keep_others = true;
    try t.expectEqual(@as(usize, 0), singleLeafVictim(&s, null, &prompt, 3, 1426, 0));
    s.keep_others = false;
    states[0].seen.hits = 0;
    try t.expectEqual(@as(usize, 0), singleLeafVictim(&s, null, &prompt, 3, 1426, 0));
    states[0] = first;
    try states[0].ext.append(t.allocator, &states[1]);
    defer states[0].ext.deinit(t.allocator);
    try t.expectEqual(@as(usize, 0), singleLeafVictim(&s, null, &prompt, 3, 1426, 0));
}
