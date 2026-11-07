//! Allocated arena capacity, not logical payload sizes or RSS; each allocation belongs to exactly one category.
const std = @import("std");
const gate = @import("prepare_gate.zig");
const Allocator = std.mem.Allocator;
pub const Kind = enum { retained, render, tokenize, media, prompt };

pub const Counter = struct {
    live: std.atomic.Value(usize) = .init(0),
    peak: std.atomic.Value(usize) = .init(0),
    allocated: std.atomic.Value(usize) = .init(0),
    freed: std.atomic.Value(usize) = .init(0),

    fn change(c: *Counter, old: usize, new: usize) void {
        if (new > old) {
            const add = new - old;
            const now = c.live.fetchAdd(add, .monotonic) + add;
            _ = c.peak.fetchMax(now, .monotonic);
            _ = c.allocated.fetchAdd(add, .monotonic);
        } else {
            const sub = old - new;
            std.debug.assert(c.live.fetchSub(sub, .monotonic) >= sub);
            _ = c.freed.fetchAdd(sub, .monotonic);
        }
    }
};

pub const Tracked = struct {
    child: Allocator,
    count: Counter = .{},

    pub fn allocator(c: *Tracked) Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
    fn self(ctx: *anyopaque) *Tracked {
        return @ptrCast(@alignCast(ctx));
    }
    fn alloc(ctx: *anyopaque, len: usize, al: std.mem.Alignment, ra: usize) ?[*]u8 {
        const c = self(ctx);
        const p = c.child.rawAlloc(len, al, ra) orelse return null;
        c.count.change(0, len);
        return p;
    }
    fn resize(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, len: usize, ra: usize) bool {
        const c = self(ctx);
        if (!c.child.rawResize(mem, al, len, ra)) return false;
        c.count.change(mem.len, len);
        return true;
    }
    fn remap(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, len: usize, ra: usize) ?[*]u8 {
        const c = self(ctx);
        const p = c.child.rawRemap(mem, al, len, ra) orelse return null;
        c.count.change(mem.len, len);
        return p;
    }
    fn free(ctx: *anyopaque, mem: []u8, al: std.mem.Alignment, ra: usize) void {
        const c = self(ctx);
        c.child.rawFree(mem, al, ra);
        c.count.change(mem.len, 0);
    }
};

pub const Ledger = struct {
    categories: [5]Tracked,

    pub fn init(a: Allocator) Ledger {
        return .{ .categories = @splat(.{ .child = a }) };
    }
    pub fn of(l: *Ledger, kind: Kind) *Tracked {
        return &l.categories[@backingInt(kind)];
    }
    pub fn render(l: *Ledger, w: *std.Io.Writer) !void {
        inline for (comptime std.meta.fieldNames(Kind), 0..) |kind, i| {
            const c = &l.categories[i].count;
            inline for (.{ "live", "peak", "allocated", "freed" }) |field|
                try w.print("tensorfold_prepare_memory_{s}_bytes{{kind=\"{s}\"}} {d}\n", .{
                    field, kind, @field(c, field).load(.monotonic),
                });
        }
    }
};

/// Only synchronous frontend work uses this scope; async borrowers retain Request explicitly.
pub threadlocal var current: ?*Request = null;

pub fn scratch(kind: Kind) Allocator {
    std.debug.assert(kind != .retained);
    return if (current) |r| r.ledger.of(kind).allocator() else std.heap.page_allocator;
}

pub const Request = struct {
    gpa: Allocator,
    io: std.Io,
    ledger: *Ledger,
    gate: ?*gate.Gate,
    reserve: u64,
    refs: std.atomic.Value(u32) = .init(1),
    arena: std.heap.ArenaAllocator,
    video_reuse: ?*const anyopaque = null, // request-owned prepared clips; warm borrowers retain this Request

    pub fn create(a: Allocator, io: std.Io, ledger: *Ledger, held: ?*gate.Gate, reserve: u64) !*Request {
        const r = try a.create(Request);
        r.* = .{ .gpa = a, .io = io, .ledger = ledger, .gate = held, .reserve = reserve, .arena = .init(ledger.of(.retained).allocator()) };
        return r;
    }
    pub fn retain(r: *Request) void {
        std.debug.assert(r.refs.fetchAdd(1, .monotonic) > 0);
    }
    pub fn release(r: *Request) void {
        if (r.refs.fetchSub(1, .acq_rel) != 1) return;
        r.arena.deinit();
        if (r.gate) |g| g.leave(r.io, r.reserve);
        r.gpa.destroy(r);
    }
};
