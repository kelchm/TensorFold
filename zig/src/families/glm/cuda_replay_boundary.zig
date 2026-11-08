//! Preallocated developer control boundaries; no serving engine installs this state.
const std = @import("std");
const cuda = @import("cuda");
const audit_mod = @import("cuda_replay_audit.zig");
const timing = @import("cuda_replay_timing.zig");
const Counts = @import("cuda_kv_shard.zig").Counts;
pub const Receipt = extern struct {
    forwards: u64,
    groups: u64,
    rows: u64,
    calls: u64,
    lost: u64,
    owner_layers: u64,
    owner_rows: u64,
    owner_bytes: u64,
    checks: u64 = 0,
    carries: u64,
    twins: u64,
    elapsed_ns: u64 = 0,
    realtime_start_ns: u64 = 0,
    realtime_end_ns: u64 = 0,
};

pub const Boundary = struct {
    audit: audit_mod.Audit,
    trace_store: ?*timing.Trace = null,
    local_timings: []timing.Record,
    all_timings: []timing.Record,
    timings_dev: cuda.DeviceBuffer,
    dev: cuda.DeviceBuffer,
    records_dev: cuda.DeviceBuffer,
    receipts: [2]Receipt = undefined,
    all_records: []audit_mod.Forward,
    started: i96 = 0,
    elapsed_ns: i96 = 0,
    realtime_start_ns: u64 = 0,
    block: bool = false,
    before: Counts = .{},
    prompt_before: Counts = .{},
    prompt_slot: ?u32 = null,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, forwards: usize) !Boundary {
        if (forwards == 0) return error.EmptyReplay;
        var dev = try cuda.DeviceBuffer.alloc(d, 2 * @sizeOf(Receipt));
        errdefer dev.free();
        var records_dev = try cuda.DeviceBuffer.alloc(d, try std.math.mul(usize, 2 * @sizeOf(audit_mod.Forward), forwards));
        errdefer records_dev.free();
        const records = try gpa.alloc(audit_mod.Forward, forwards);
        errdefer gpa.free(records);
        const all_records = try gpa.alloc(audit_mod.Forward, 2 * forwards);
        errdefer gpa.free(all_records);
        const local_timings = try gpa.alloc(timing.Record, forwards);
        errdefer gpa.free(local_timings);
        const all_timings = try gpa.alloc(timing.Record, 2 * forwards);
        errdefer gpa.free(all_timings);
        const timings_dev = try cuda.DeviceBuffer.alloc(d, try std.math.mul(usize, 2 * @sizeOf(timing.Record), forwards));
        return .{ .local_timings = local_timings, .all_timings = all_timings, .timings_dev = timings_dev, .audit = .{ .records = records }, .dev = dev, .records_dev = records_dev, .all_records = all_records };
    }

    pub fn deinit(b: *Boundary, gpa: std.mem.Allocator) void {
        b.timings_dev.free();
        gpa.free(b.local_timings);
        gpa.free(b.all_timings);
        gpa.free(b.audit.records);
        gpa.free(b.all_records);
        b.records_dev.free();
        b.dev.free();
    }

    pub fn apply(b: *Boundary, e: anytype, action: u32, expected: u64) !void {
        if (action == 3 or action == 4) {
            if (b.block or b.audit.pending) return error.UnfinishedReplay;
            b.audit.trace = if (action == 3) b.trace_store orelse return error.ReplayNoTimingStorage else null;
            return;
        }
        if (action == 7) {
            if (b.block or b.audit.pending or expected >= 4 or b.prompt_slot != null) return error.ReplayPreparation;
            b.prompt_before = audit_mod.counts(e);
            b.prompt_slot = @intCast(expected);
            return;
        }
        if (action == 5) {
            if (b.block or b.audit.pending or expected >= 4 or b.prompt_slot != @as(u32, @intCast(expected)))
                return error.ReplayPreparation;
            if (e.settings.kv_shard.on) try audit_mod.checkTwin(b.prompt_before, audit_mod.counts(e));
            try e.giveTwin(@intCast(expected));
            b.prompt_slot = null;
            return;
        }
        if (action == 1) {
            if (b.block or b.audit.pending or b.prompt_slot != null) return error.UnfinishedReplay;
            if (b.audit.trace) |t| try t.reset();
            b.audit.used = 0;
            b.audit.forwards = 0;
            b.audit.groups = 0;
            b.audit.rows = 0;
            b.audit.calls = e.drafter.?.calls;
            b.audit.lost = e.drafter.?.lost;
            try e.stream.synchronize();
            if (e.side) |side| try side.s.synchronize();
            try e.lib.check(e.lib.api.ncclAllGather(b.dev.ptr + e.share.rank, b.dev.ptr, 1, .u8, e.comm, e.stream.handle), "ncclAllGather");
            try e.stream.synchronize();
            b.before = audit_mod.counts(e);
            b.block = true;
            b.startClock(e.io);
            return;
        }
        if (action != 0 or expected == 0 or !b.block or b.audit.pending or b.audit.forwards != expected or
            b.audit.groups != expected) return error.UnfinishedReplay;
        try e.stream.synchronize();
        if (e.side) |side| try side.s.synchronize();
        const c = try audit_mod.delta(b.before, audit_mod.counts(e));
        var twins: u64 = 0;
        if (e.pages.?.shard) |sh| for (sh.parts.slots) |s| {
            twins += s.twin.items.len;
        };
        var r: Receipt = .{ .forwards = b.audit.forwards, .groups = b.audit.groups, .rows = b.audit.rows, .calls = e.drafter.?.calls - b.audit.calls, .lost = e.drafter.?.lost - b.audit.lost, .owner_layers = c.owner_layers, .owner_rows = c.owner_rows, .owner_bytes = c.owner_bytes, .checks = c.checks, .carries = c.carries, .twins = twins };
        const size = @sizeOf(Receipt);
        try b.dev.uploadAsync(e.share.rank * size, std.mem.asBytes(&r), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllGather(b.dev.ptr + e.share.rank * size, b.dev.ptr, size, .u8, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        b.stopClock(e.io, &r);
        // Exchange local completion clocks after the timed all-rank boundary has finished.
        try b.dev.uploadAsync(e.share.rank * size, std.mem.asBytes(&r), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllGather(b.dev.ptr + e.share.rank * size, b.dev.ptr, size, .u8, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        try b.dev.download(0, std.mem.asBytes(&b.receipts));
        try validateReceipts(b.receipts, expected, r.rows);
        const bytes = b.audit.used * @sizeOf(audit_mod.Forward);
        if (bytes > 0) {
            try b.records_dev.uploadAsync(e.share.rank * bytes, std.mem.sliceAsBytes(b.audit.records[0..b.audit.used]), e.stream.handle);
            try e.lib.check(e.lib.api.ncclAllGather(b.records_dev.ptr + e.share.rank * bytes, b.records_dev.ptr, bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
            try e.stream.synchronize();
            try b.records_dev.download(0, std.mem.sliceAsBytes(b.all_records[0 .. 2 * b.audit.used]));
        }
        if (b.audit.trace) |t| {
            const records = try t.resolve(b.local_timings);
            const timing_bytes = std.mem.sliceAsBytes(records);
            try b.timings_dev.uploadAsync(e.share.rank * timing_bytes.len, timing_bytes, e.stream.handle);
            try e.lib.check(e.lib.api.ncclAllGather(b.timings_dev.ptr + e.share.rank * timing_bytes.len, b.timings_dev.ptr, timing_bytes.len, .u8, e.comm, e.stream.handle), "ncclAllGather");
            try e.stream.synchronize();
            try b.timings_dev.download(0, std.mem.sliceAsBytes(b.all_timings[0 .. 2 * records.len]));
        }
        b.block = false;
    }

    fn startClock(b: *Boundary, io: std.Io) void {
        b.realtime_start_ns = @intCast(std.Io.Clock.real.now(io).toNanoseconds());
        b.started = std.Io.Clock.awake.now(io).toNanoseconds();
    }

    fn stopClock(b: *Boundary, io: std.Io, r: *Receipt) void {
        b.elapsed_ns = std.Io.Clock.awake.now(io).toNanoseconds() - b.started;
        r.realtime_end_ns = @intCast(std.Io.Clock.real.now(io).toNanoseconds());
        r.elapsed_ns = @intCast(b.elapsed_ns);
        r.realtime_start_ns = b.realtime_start_ns;
    }

    // Followers map real twin pages rather than allocating them; count that replay event once at begin.
    pub fn noteMappedTwin(b: *Boundary, e: anytype, slot: u32) !void {
        if (!e.settings.kv_shard.on) return;
        if (b.prompt_slot != slot) return error.ReplayPreparation;
        const sh = e.pages.?.shard orelse return error.ReplayPromptTwin;
        if (sh.parts.slots[slot].twin.items.len == 0) return error.ReplayPromptTwin;
        sh.parts.counts.twins_made += 1;
    }
};

pub fn validateReceipts(receipts: [2]Receipt, expected: u64, rows: u64) !void {
    for (receipts) |r| if (r.forwards != expected or r.groups != expected or r.rows != rows or
        r.calls != 0 or r.lost != 0 or r.carries != 0 or r.twins != 0 or r.elapsed_ns == 0) return error.ReplayCompletion;
}

test "replay completion rejects unfinished follower ingest and a proposer call on either rank" {
    const good: Receipt = .{ .forwards = 3, .groups = 3, .rows = 8, .calls = 0, .lost = 0, .owner_layers = 0, .owner_rows = 0, .owner_bytes = 0, .carries = 0, .twins = 0, .elapsed_ns = 10 };
    var receipts = [_]Receipt{ good, good };
    try validateReceipts(receipts, 3, 8);
    receipts[1].groups = 2;
    try std.testing.expectError(error.ReplayCompletion, validateReceipts(receipts, 3, 8));
    receipts[1] = good;
    receipts[0].calls = 1;
    try std.testing.expectError(error.ReplayCompletion, validateReceipts(receipts, 3, 8));
}

test "replay follower asserts the real mapped twin instead of accepting a host budget" {
    const kvs = @import("cuda_kv_shard.zig");
    const Sh = struct { parts: kvs.Parts };
    const P = struct { shard: ?*Sh };
    const E = struct { settings: struct { kv_shard: struct { on: bool = true } = .{} } = .{}, pages: ?*P };
    var sh: Sh = .{ .parts = try kvs.Parts.init(std.testing.allocator, 8, 4096, 4, false) };
    defer sh.parts.deinit();
    var p: P = .{ .shard = &sh };
    const e: E = .{ .pages = &p };
    var b: Boundary = undefined;
    b.prompt_slot = 0;
    try sh.parts.map(0, &.{ 0, 1 }, &.{8});
    const before = sh.parts.counts;
    try b.noteMappedTwin(e, 0);
    try audit_mod.checkTwin(before, sh.parts.counts);
    sh.parts.slots[0].twin.clearRetainingCapacity();
    try std.testing.expectError(error.ReplayPromptTwin, b.noteMappedTwin(e, 0));
    try std.testing.expectError(error.ReplayPreparation, b.noteMappedTwin(e, 1));
}

test "replay rank-local realtime windows enclose monotonic wall and retain independent clocks" {
    const Clock = struct {
        stamps: [4]i96,
        clocks: [4]std.Io.Clock = undefined,
        used: usize = 0,
        fn now(ptr: ?*anyopaque, clock: std.Io.Clock) std.Io.Timestamp {
            const c: *@This() = @ptrCast(@alignCast(ptr.?));
            c.clocks[c.used] = clock;
            const stamp = c.stamps[c.used];
            c.used += 1;
            return .fromNanoseconds(stamp);
        }
    };
    var vtable: std.Io.VTable = undefined;
    vtable.now = Clock.now;
    var ranks: [2]Receipt = undefined;
    const stamps = [2][4]i96{ .{ 1000, 10, 110, 1101 }, .{ 2000, 40, 213, 2174 } };
    for (stamps, &ranks) |local, *receipt| {
        var clock: Clock = .{ .stamps = local };
        const io: std.Io = .{ .userdata = &clock, .vtable = &vtable };
        var b: Boundary = undefined;
        b.startClock(io);
        b.stopClock(io, receipt);
        try std.testing.expectEqualSlices(std.Io.Clock, &.{ .real, .awake, .awake, .real }, &clock.clocks);
        try std.testing.expectEqual(@as(u64, @intCast(local[2] - local[1])), receipt.elapsed_ns);
        try std.testing.expectEqual(@as(u64, @intCast(local[0])), receipt.realtime_start_ns);
        try std.testing.expectEqual(@as(u64, @intCast(local[3])), receipt.realtime_end_ns);
    }
    try std.testing.expect(ranks[0].elapsed_ns != ranks[1].elapsed_ns);
    try std.testing.expect(ranks[0].realtime_start_ns != ranks[1].realtime_start_ns);
}
