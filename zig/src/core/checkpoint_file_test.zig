const std = @import("std");
const platform = @import("sys");
const file = @import("checkpoint_file.zig");
const rows = @import("rows_file.zig");
const Writer = file.Writer(rows);
const stream = @import("checkpoint_stream.zig");
const cap = @import("checkpoint_capture.zig");
const kv = @import("kv_pool.zig");
const t = std.testing;

const Source = struct {
    at: usize = 0,
    data: []const u8 = "checkpoint payload with a short last chunk",
    fn next(ptr: *anyopaque, n: usize) !?cap.Chunk {
        const s: *Source = @ptrCast(@alignCast(ptr));
        if (s.at == s.data.len) return null;
        const len = @min(n, s.data.len - s.at);
        const c: cap.Chunk = .{ .location = .{ .host = s.data[s.at..][0..len] }, .offset = s.at, .len = len };
        s.at += len;
        return c;
    }
};

const Fault = struct {
    mode: enum { ok, write, sync_file, link, sync_dir } = .ok,
    phase: u8 = 0,
    writes: usize = 0,

    fn ctx(ptr: ?*anyopaque) *Fault {
        return @ptrCast(@alignCast(ptr.?));
    }
    fn write(ptr: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        const f = ctx(ptr);
        if (f.mode == .write and f.writes > 1) return error.InjectedWrite;
        f.writes += 1;
        const n = platform.c.write(fd, bytes.ptr, @min(bytes.len, 3));
        if (n <= 0) return error.RowsWrite;
        return @intCast(n);
    }
    fn syncFile(ptr: ?*anyopaque, fd: c_int) !void {
        const f = ctx(ptr);
        try t.expectEqual(@as(u8, 0), f.phase);
        if (f.mode == .sync_file) return error.InjectedFileSync;
        if (platform.c.fsync(fd) != 0) return error.RowsWrite;
        f.phase = 1;
    }
    fn link(ptr: ?*anyopaque, from: [:0]const u8, to: [:0]const u8) !void {
        const f = ctx(ptr);
        try t.expectEqual(@as(u8, 1), f.phase);
        if (f.mode == .link) return error.InjectedLink;
        if (std.c.link(from, to) != 0) return error.CheckpointPublish;
        f.phase = 2;
    }
    fn syncDir(ptr: ?*anyopaque, fd: c_int) !void {
        const f = ctx(ptr);
        try t.expectEqual(@as(u8, 2), f.phase);
        if (f.mode == .sync_dir) return error.InjectedDirectorySync;
        if (platform.c.fsync(fd) != 0) return error.RowsWrite;
        f.phase = 3;
    }
};

fn run(mode: @FieldType(Fault, "mode")) !void {
    var dir_buf: [128]u8 = undefined;
    const dir = try std.fmt.bufPrintSentinel(&dir_buf, "/tmp/tf-checkpoint-050-{d}-{s}", .{ platform.c.getpid(), @tagName(mode) }, 0);
    try t.expectEqual(@as(c_int, 0), platform.c.mkdir(dir, 0o700));
    defer _ = platform.c.rmdir(dir);
    var pool = try kv.Pool.init(t.allocator, 0, 4096);
    defer pool.deinit();
    const live: kv.Lease = .{};
    var budget: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 0, .pages_per_descriptor = 0, .capsule_bytes = 0 } };
    var owner = try budget.acquire(&live, 0, 0);
    defer owner.close() catch unreachable;
    var s: Source = .{};
    var slots: [2]stream.Slot = undefined;
    var storage: [16]u8 = undefined;
    var ring = try stream.Ring.init(.{ .owner = &owner, .ctx = &s, .len = s.data.len, .next = Source.next }, &storage, &slots, 8);
    const identity: file.Identity = .{ .block = @splat(3), .rank = 1, .epoch = 12 };
    var w = try Writer.open(dir, identity, 123, s.data.len, 32);
    defer w.close();
    try t.expectError(error.RowsWrite, Writer.open(dir, identity, 123, s.data.len, 32));
    var fault: Fault = .{ .mode = mode };
    w.file.io = .{ .ctx = &fault, .write = Fault.write, .sync = Fault.syncFile };
    w.publication = .{ .ctx = &fault, .link = Fault.link, .sync_dir = Fault.syncDir };
    const dest = w.file.dest[0..w.file.dest_len :0];
    defer _ = platform.c.unlink(dest);
    if (mode == .ok) {
        const first = try ring.acquire();
        const second = try ring.acquire();
        @memcpy(second.target, second.chunk.location.host);
        try ring.dmaDone(second.key, true);
        try t.expectError(error.CheckpointSinkBackpressure, w.consume(&ring, second.key));
        @memcpy(first.target, first.chunk.location.host);
        try ring.dmaDone(first.key, true);
        try w.consume(&ring, first.key);
        try ring.retire(first.key);
        try w.consume(&ring, second.key);
        try ring.retire(second.key);
    }
    while (ring.emitted < s.data.len) {
        const ticket = try ring.acquire();
        @memcpy(ticket.target, ticket.chunk.location.host);
        try ring.dmaDone(ticket.key, true);
        if (mode == .write) {
            try t.expectError(error.InjectedWrite, w.consume(&ring, ticket.key));
            try ring.retire(ticket.key);
            try ring.close();
            try t.expectError(error.CheckpointFileState, w.seal(&ring, 2, &.{}));
            w.close();
            try t.expectError(error.RowsMissing, rows.open(dest, .state, null));
            return;
        }
        try w.consume(&ring, ticket.key);
        try t.expectError(error.CheckpointBusy, w.seal(&ring, 2, &.{})); // the slot is not retired yet
        try ring.retire(ticket.key);
    }
    try t.expectError(error.CheckpointMetadataLimit, w.seal(&ring, 2, &.{"this metadata is deliberately more than thirty two bytes"}));
    if (mode == .sync_file) {
        try t.expectError(error.InjectedFileSync, w.seal(&ring, 2, &.{"metadata"}));
    } else {
        try w.seal(&ring, 2, &.{"metadata"});
        try t.expectError(error.RowsMissing, rows.open(dest, .state, null));
        if (mode == .link) {
            try t.expectError(error.InjectedLink, w.publish());
        } else if (mode == .sync_dir) {
            try t.expectError(error.InjectedDirectorySync, w.publish());
        } else {
            const local = try w.publish();
            try t.expectEqual(@as(u64, s.data.len + 8 + 96), local.bytes);
            try t.expectEqual(@as(u8, 3), fault.phase);
            var opened = try rows.open(dest, .state, identity.name());
            defer opened.close();
            const stage = try rows.stage(t.allocator, s.data.len);
            defer t.allocator.free(stage);
            try t.expectEqualSlices(u8, s.data, try rows.payload(&opened, stage));
            const meta = try rows.meta(&opened, t.allocator);
            defer t.allocator.free(meta);
            try t.expectEqualStrings("metadata", meta);
            // Another nonce cannot replace an existing immutable rank/epoch file.
            var other = try Writer.open(dir, identity, 124, s.data.len, 32);
            defer other.close();
            try other.append(0, s.data);
            try other.seal(&ring, 2, &.{"changed"});
            try t.expectError(error.CheckpointPublish, other.publish());
            try t.expectEqualSlices(u8, s.data, try rows.payload(&opened, stage));
        }
    }
    try ring.close();
    w.close();
    if (mode != .ok) try t.expectError(error.RowsMissing, rows.open(dest, .state, null));
    const tmp = w.file.tmp[0..w.file.tmp_len :0];
    try t.expectError(error.RowsMissing, rows.open(tmp, .state, null));
}

test "checkpoint bounded: rank chunk writer streams short writes and orders fsync before exclusive publication" {
    try run(.ok);
}

test "checkpoint bounded: write, file fsync, link and directory fsync failures never return local publication" {
    for ([_]@FieldType(Fault, "mode"){ .write, .sync_file, .link, .sync_dir }) |mode| try run(mode);
}

test "checkpoint bounded: rank and epoch domain separate an existing rows identity" {
    const a: file.Identity = .{ .block = @splat(7), .rank = 1, .epoch = 19 };
    var b = a;
    b.rank = 2;
    var c = a;
    c.epoch = 20;
    try t.expect(!std.mem.eql(u8, &a.name(), &b.name()));
    try t.expect(!std.mem.eql(u8, &a.name(), &c.name()));
}
