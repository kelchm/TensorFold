//! Optional bounded replay events; resolution happens only after all-rank block completion.
const std = @import("std");
const cuda = @import("cuda");

pub const Phase = enum { staging, target, bookkeeping, selection, target_commit, context_commit, end };
pub const phases = 6;
pub const Record = struct { id: u64, ms: [phases]f32 };

pub const Trace = struct {
    gpa: std.mem.Allocator,
    events: []cuda.Event,
    ids: []u64,
    n: usize = 0,
    active: bool = false,
    next: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, forwards: usize) !Trace {
        if (forwards == 0) return error.EmptyReplay;
        const count = try std.math.mul(usize, forwards, phases + 1);
        const ev = try gpa.alloc(cuda.Event, count);
        errdefer gpa.free(ev);
        var made: usize = 0;
        errdefer for (ev[0..made]) |*e| e.deinit();
        for (ev) |*e| {
            e.* = try cuda.Event.init(d, true);
            made += 1;
        }
        return .{ .gpa = gpa, .events = ev, .ids = try gpa.alloc(u64, forwards) };
    }

    pub fn deinit(t: *Trace) void {
        for (t.events) |*e| e.deinit();
        t.gpa.free(t.events);
        t.gpa.free(t.ids);
    }

    pub fn reset(t: *Trace) !void {
        if (t.active) return error.UnfinishedReplay;
        t.n = 0;
    }

    pub fn begin(t: *Trace, s: cuda.Stream, id: u64) !void {
        if (t.active or t.n >= t.ids.len) return error.ReplayTimingCapacity;
        t.active = true;
        t.next = 0;
        t.ids[t.n] = id;
        try t.mark(s, .staging);
    }

    pub fn mark(t: *Trace, s: cuda.Stream, phase: Phase) !void {
        const i = @backingInt(phase);
        if (!t.active or t.next != i) return error.ReplayTimingOrder;
        try t.events[t.n * (phases + 1) + i].record(s);
        t.next += 1;
        if (phase == .end) {
            t.active = false;
            t.n += 1;
        }
    }

    pub fn resolve(t: *const Trace, out: []Record) ![]Record {
        if (t.active or out.len < t.n) return error.UnfinishedReplay;
        for (out[0..t.n], 0..) |*r, j| {
            r.id = t.ids[j];
            const ev = t.events[j * (phases + 1) ..][0 .. phases + 1];
            for (&r.ms, 0..) |*ms, i| ms.* = try cuda.Event.elapsedMs(ev[i], ev[i + 1]);
        }
        return out[0..t.n];
    }
};

pub inline fn mark(t: ?*Trace, s: cuda.Stream, phase: Phase) !void {
    if (t) |v| try v.mark(s, phase);
}

const Fake = struct {
    var created: usize = 0;
    var destroyed: usize = 0;
    var recorded: usize = 0;
    fn create(out: *cuda.abi.Event, _: c_uint) callconv(.c) cuda.abi.Result {
        created += 1;
        out.* = @ptrFromInt(created);
        return 0;
    }
    fn destroy(_: cuda.abi.Event) callconv(.c) cuda.abi.Result {
        destroyed += 1;
        return 0;
    }
    fn record(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        recorded += 1;
        return 0;
    }
};

test "replay null observer submits no events; bounded observer preserves phase order without allocation" {
    var d: cuda.Driver = undefined;
    d.api.cuEventCreate = Fake.create;
    d.api.cuEventDestroy_v2 = Fake.destroy;
    d.api.cuEventRecord = Fake.record;
    Fake.created = 0;
    Fake.destroyed = 0;
    Fake.recorded = 0;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    try mark(null, stream, .target);
    try std.testing.expectEqual(@as(usize, 0), Fake.recorded);
    var trace = try Trace.init(std.testing.allocator, &d, 1);
    try trace.begin(stream, 7);
    try std.testing.expectError(error.ReplayTimingOrder, trace.mark(stream, .selection));
    for ([_]Phase{ .target, .bookkeeping, .selection, .target_commit, .context_commit, .end }) |phase|
        try trace.mark(stream, phase);
    try std.testing.expectEqual(@as(usize, 7), Fake.created);
    try std.testing.expectEqual(@as(usize, 7), Fake.recorded);
    try std.testing.expectError(error.ReplayTimingCapacity, trace.begin(stream, 8));
    trace.deinit();
    try std.testing.expectEqual(Fake.created, Fake.destroyed);
}
