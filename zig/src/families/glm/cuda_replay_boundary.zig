//! Preallocated developer control boundaries; no serving engine installs this state.
const std = @import("std");
const cuda = @import("cuda");
const audit_mod = @import("cuda_replay_audit.zig");
const timing = @import("cuda_replay_timing.zig");
pub const Receipt = extern struct {
    forwards: u64,
    groups: u64,
    rows: u64,
    calls: u64,
    lost: u64,
    owner_layers: u64,
    owner_rows: u64,
    owner_bytes: u64,
    carries: u64,
    twins: u64,
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
    block: bool = false,

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
        if (action == 5) {
            if (b.block or b.audit.pending or expected >= 4) return error.ReplayPreparation;
            try e.giveTwin(@intCast(expected));
            return;
        }
        if (action == 1) {
            if (b.block or b.audit.pending) return error.UnfinishedReplay;
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
            b.block = true;
            b.started = std.Io.Clock.awake.now(e.io).toNanoseconds();
            return;
        }
        if (action != 0 or expected == 0 or !b.block or b.audit.pending or b.audit.forwards != expected or
            b.audit.groups != expected) return error.UnfinishedReplay;
        try e.stream.synchronize();
        if (e.side) |side| try side.s.synchronize();
        const c = audit_mod.counts(e);
        var twins: u64 = 0;
        if (e.pages.?.shard) |sh| for (sh.parts.slots) |s| {
            twins += s.twin.items.len;
        };
        const r: Receipt = .{ .forwards = b.audit.forwards, .groups = b.audit.groups, .rows = b.audit.rows, .calls = e.drafter.?.calls - b.audit.calls, .lost = e.drafter.?.lost - b.audit.lost, .owner_layers = c.owner_layers, .owner_rows = c.owner_rows, .owner_bytes = c.owner_bytes, .carries = c.carries, .twins = twins };
        const size = @sizeOf(Receipt);
        try b.dev.uploadAsync(e.share.rank * size, std.mem.asBytes(&r), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllGather(b.dev.ptr + e.share.rank * size, b.dev.ptr, size, .u8, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        b.elapsed_ns = std.Io.Clock.awake.now(e.io).toNanoseconds() - b.started;
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
};

pub fn validateReceipts(receipts: [2]Receipt, expected: u64, rows: u64) !void {
    for (receipts) |r| if (r.forwards != expected or r.groups != expected or r.rows != rows or
        r.calls != 0 or r.lost != 0 or r.carries != 0 or r.twins != 0) return error.ReplayCompletion;
}

test "replay completion rejects unfinished follower ingest and a proposer call on either rank" {
    const good: Receipt = .{ .forwards = 3, .groups = 3, .rows = 8, .calls = 0, .lost = 0, .owner_layers = 0, .owner_rows = 0, .owner_bytes = 0, .carries = 0, .twins = 0 };
    var receipts = [_]Receipt{ good, good };
    try validateReceipts(receipts, 3, 8);
    receipts[1].groups = 2;
    try std.testing.expectError(error.ReplayCompletion, validateReceipts(receipts, 3, 8));
    receipts[1] = good;
    receipts[0].calls = 1;
    try std.testing.expectError(error.ReplayCompletion, validateReceipts(receipts, 3, 8));
}
