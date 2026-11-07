const std = @import("std");
const kv = @import("kv_pool.zig");
const cap = @import("checkpoint_capture.zig");
const t = std.testing;

test "checkpoint bounded: four growing leases keep every promised page while checkpoints are undurable" {
    var pool = try kv.Pool.init(t.allocator, 20, 4096);
    defer pool.deinit();
    var live: [4]kv.Lease = @splat(.{});
    defer for (&live) |*l| pool.finish(l);
    for (&live) |*l| {
        try pool.promise(l, 5);
        try pool.cover(l, 2, null);
    }
    var budget: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 4, .unique_pages = 8, .pages_per_descriptor = 5, .capsule_bytes = 64 } };
    var saved: [4]cap.Lease = undefined;
    for (&live, &saved) |*l, *s| s.* = try budget.acquire(l, 2, 16);
    try t.expectEqual(@as(usize, 8), pool.npinned);
    try t.expectEqual(@as(usize, 12), pool.promised);
    try t.expectError(error.CheckpointBackpressure, budget.acquire(&live[0], 1, 1));
    for (&live) |*l| try pool.cover(l, 5, null);
    try t.expectEqual(@as(usize, 0), pool.nfree);
    for (&live) |*l| pool.finish(l);
    try t.expectEqual(@as(usize, 12), pool.available());
    try t.expectEqual(@as(usize, 0), pool.reclaimable());
    var next: kv.Lease = .{};
    defer pool.finish(&next);
    try t.expectError(error.LaneMemory, pool.promise(&next, 13));
    for (&saved) |*s| try s.close();
    try t.expectEqual(@as(usize, 20), pool.available());
    try t.expectEqual(@as(usize, 0), budget.active + budget.capsule_used + pool.npinned);
}

test "checkpoint bounded: forks count shared physical pages once and pins are not cache refs" {
    var pool = try kv.Pool.init(t.allocator, 6, 4096);
    defer pool.deinit();
    var parent: kv.Lease = .{};
    defer pool.finish(&parent);
    var fork: kv.Lease = .{};
    defer pool.finish(&fork);
    try pool.promise(&parent, 3);
    try pool.cover(&parent, 3, null);
    for (parent.ids.items[0..2]) |id| pool.retain(id);
    try pool.adopt(&fork, parent.ids.items[0..2]);
    var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 3, .unique_pages = 2, .pages_per_descriptor = 3, .capsule_bytes = 8 } };
    var a = try b.acquire(&parent, 2, 4);
    var f = try b.acquire(&fork, 2, 4);
    try t.expectEqual(@as(usize, 2), pool.npinned);
    try t.expectEqual(@as(u32, 2), pool.pins[0]);
    try t.expectError(error.NotHeld, pool.drop(0));
    try t.expectError(error.CheckpointBackpressure, b.acquire(&parent, 3, 0));
    pool.finish(&parent);
    pool.finish(&fork);
    try t.expectEqual(@as(usize, 0), pool.reclaimable());
    try a.close();
    try t.expectEqual(@as(usize, 0), pool.reclaimable());
    pool.retain(0); // an independent RAM-cache holder can release only its own reference
    try pool.drop(0);
    try t.expectError(error.NotHeld, pool.drop(0));
    try f.close();
    try t.expectEqual(@as(usize, 6), pool.available());
}

test "checkpoint bounded: admission rejects malformed and excess metadata without leaking credits" {
    var pool = try kv.Pool.init(t.allocator, 4, 4096);
    defer pool.deinit();
    var live: kv.Lease = .{};
    defer pool.finish(&live);
    try pool.promise(&live, 3);
    try pool.cover(&live, 2, null);
    var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 3, .pages_per_descriptor = 2, .capsule_bytes = 8 } };
    try t.expectError(error.InvalidPages, b.acquire(&live, 3, 0));
    try t.expectError(error.CheckpointBackpressure, b.acquire(&live, 1, 9));
    const id = live.ids.items[1];
    live.ids.items[1] = live.ids.items[0];
    try t.expectError(error.InvalidPages, b.acquire(&live, 2, 0));
    live.ids.items[1] = id;
    var saved = try b.acquire(&live, 2, 8);
    try saved.borrow();
    try t.expectError(error.CheckpointBusy, saved.close());
    saved.unborrow();
    try saved.close();
    try t.expectEqual(@as(usize, 0), pool.npinned + b.active + b.capsule_used);
    try t.expectEqual(@as(usize, 1), pool.promised);
}

fn allocationFailures(a: std.mem.Allocator) !void {
    var pool = try kv.Pool.init(a, 8, 4096);
    defer pool.deinit();
    var live: kv.Lease = .{};
    defer pool.finish(&live);
    try pool.promise(&live, 4);
    try pool.cover(&live, 2, null);
    var b: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 2, .pages_per_descriptor = 2, .capsule_bytes = 16 } };
    var saved = try b.acquire(&live, 2, 16);
    try saved.close();
    try t.expectEqual(@as(usize, 0), pool.npinned + b.active + b.capsule_used);
    try t.expectEqual(@as(usize, 2), pool.promised);
}

test "checkpoint bounded: every descriptor and pool allocation failure preserves ownership" {
    try t.checkAllAllocationFailures(t.allocator, allocationFailures, .{});
}
