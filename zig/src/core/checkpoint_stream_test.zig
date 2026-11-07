const std = @import("std");
const cap = @import("checkpoint_capture.zig");
const stream = @import("checkpoint_stream.zig");
const kv = @import("kv_pool.zig");
const t = std.testing;

const Source = struct {
    at: u64 = 0,
    len: u64 = 64,

    fn next(ptr: *anyopaque, limit: usize) !?cap.Chunk {
        const s: *Source = @ptrCast(@alignCast(ptr));
        if (s.at == s.len) return null;
        const n = @min(s.len - s.at, limit);
        const chunk: cap.Chunk = .{ .offset = s.at, .location = .{ .device = 4096 + s.at }, .len = @intCast(n) };
        s.at += n;
        return chunk;
    }
    fn source(s: *Source, owner: *cap.Lease) cap.Source {
        return .{ .owner = owner, .ctx = s, .len = s.len, .next = next };
    }
};

test "checkpoint bounded: delayed DMA and delayed I/O cannot recycle a transfer slot" {
    var pool = try kv.Pool.init(t.allocator, 0, 4096);
    defer pool.deinit();
    const live: kv.Lease = .{};
    var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 0, .pages_per_descriptor = 0, .capsule_bytes = 0 } };
    var owner = try b.acquire(&live, 0, 0);
    defer owner.close() catch unreachable;
    var source: Source = .{};
    var slots: [2]stream.Slot = undefined;
    var storage: [32]u8 = undefined;
    var r = try stream.Ring.init(source.source(&owner), &storage, &slots, 16);
    const a = try r.acquire();
    const z = try r.acquire();
    try t.expectError(error.CheckpointBackpressure, r.acquire());
    try t.expectError(error.InvalidTransition, r.beginWrite(a.key));
    try t.expectError(error.CheckpointBusy, r.retire(a.key));
    try t.expectError(error.CheckpointBusy, owner.close());
    try r.dmaDone(z.key, true); // second DMA completes first
    const out = try r.beginWrite(z.key);
    try t.expectEqual(z.target.ptr, out.ptr);
    try t.expectError(error.CheckpointBusy, r.retire(z.key));
    try t.expectError(error.CheckpointBackpressure, r.acquire());
    try r.ioDone(z.key, true);
    try r.retire(z.key);
    const c = try r.acquire();
    try t.expectEqual(z.key.slot, c.key.slot);
    try t.expect(c.key.generation != z.key.generation);
    try t.expectError(error.StaleCompletion, r.ioDone(z.key, true));
    for ([_]stream.Ticket{ a, c }) |ticket| {
        try r.dmaDone(ticket.key, true);
        _ = try r.beginWrite(ticket.key);
        try r.ioDone(ticket.key, true);
        try r.retire(ticket.key);
    }
    const last = try r.acquire();
    try r.dmaDone(last.key, true);
    _ = try r.beginWrite(last.key);
    try r.ioDone(last.key, true);
    try r.retire(last.key);
    try t.expect(r.readyToSeal());
    try t.expectError(error.EndOfCheckpoint, r.acquire());
    try r.close();
}

test "checkpoint bounded: cancel and either worker failure drain both owners before freeing buffers" {
    for (0..3) |mode| {
        var pool = try kv.Pool.init(t.allocator, 0, 4096);
        defer pool.deinit();
        const live: kv.Lease = .{};
        var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 0, .pages_per_descriptor = 0, .capsule_bytes = 0 } };
        var owner = try b.acquire(&live, 0, 0);
        defer owner.close() catch unreachable;
        var source: Source = .{};
        var slots: [2]stream.Slot = undefined;
        var storage: [32]u8 = undefined;
        var r = try stream.Ring.init(source.source(&owner), &storage, &slots, 16);
        const dma = try r.acquire();
        const io = try r.acquire();
        try r.dmaDone(io.key, true);
        _ = try r.beginWrite(io.key);
        switch (mode) {
            0 => r.cancel(),
            1 => try r.dmaDone(dma.key, false),
            2 => try r.ioDone(io.key, false),
            else => unreachable,
        }
        try t.expectError(error.CheckpointStopped, r.acquire());
        try t.expectError(error.CheckpointBusy, r.close());
        try t.expectError(error.CheckpointBusy, owner.close());
        if (mode != 1) {
            try t.expectError(error.CheckpointBusy, r.retire(dma.key));
            try r.dmaDone(dma.key, true);
        }
        if (mode != 2) {
            try t.expectError(error.CheckpointBusy, r.retire(io.key));
            try r.ioDone(io.key, true);
        }
        try r.retire(dma.key);
        try r.retire(io.key);
        try t.expect(!r.readyToSeal());
        try r.close();
    }
}

test "checkpoint bounded: a multi GiB source uses only a fixed credit ring" {
    var pool = try kv.Pool.init(t.allocator, 0, 4096);
    defer pool.deinit();
    const live: kv.Lease = .{};
    var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 0, .pages_per_descriptor = 0, .capsule_bytes = 0 } };
    var owner = try b.acquire(&live, 0, 0);
    defer owner.close() catch unreachable;
    var source: Source = .{ .len = (12 << 30) + 7 };
    var slots: [2]stream.Slot = undefined;
    const storage = try t.allocator.alloc(u8, 2 << 20);
    defer t.allocator.free(storage);
    var r = try stream.Ring.init(source.source(&owner), storage, &slots, 1 << 20);
    while (r.emitted < source.len) {
        const a = try r.acquire();
        try r.dmaDone(a.key, true);
        _ = try r.beginWrite(a.key);
        try r.ioDone(a.key, true);
        try r.retire(a.key);
    }
    try t.expectEqual(source.len, r.written);
    try t.expectEqual(@as(usize, 2 << 20), r.storage.len);
    try t.expect(r.readyToSeal());
    try r.close();
}

test "checkpoint bounded: malformed source addresses stop before submitting DMA" {
    const Invalid = struct {
        location: cap.Location,
        fn next(ptr: *anyopaque, _: usize) !?cap.Chunk {
            const s: *@This() = @ptrCast(@alignCast(ptr));
            return .{ .location = s.location, .offset = 0, .len = 8 };
        }
    };
    var pool = try kv.Pool.init(t.allocator, 0, 4096);
    defer pool.deinit();
    const live: kv.Lease = .{};
    var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 0, .pages_per_descriptor = 0, .capsule_bytes = 0 } };
    var owner = try b.acquire(&live, 0, 0);
    defer owner.close() catch unreachable;
    for ([_]cap.Location{ .{ .host = "short" }, .{ .device = std.math.maxInt(u64) - 3 } }) |location| {
        var s: Invalid = .{ .location = location };
        var slots: [1]stream.Slot = undefined;
        var storage: [8]u8 = undefined;
        var r = try stream.Ring.init(.{ .owner = &owner, .ctx = &s, .len = 8, .next = Invalid.next }, &storage, &slots, 8);
        try t.expectError(error.InvalidChunk, r.acquire());
        try t.expectEqual(@as(usize, 0), r.active);
        try r.close();
    }
}
