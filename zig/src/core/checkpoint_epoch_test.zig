const std = @import("std");
const sys = @import("sys");
const epoch = @import("checkpoint_epoch.zig");
const rows = @import("rows_file.zig");
const rank = @import("checkpoint_file.zig");
const stream = @import("checkpoint_stream.zig");
const cap = @import("checkpoint_capture.zig");
const kv = @import("kv_pool.zig");
const Store = epoch.Store(rows);
const t = std.testing;

fn validateFixture(_: epoch.Envelope, detail: []const u8) !void {
    if (!std.mem.eql(u8, detail, "state")) return error.CheckpointFixtureMetadata;
}

pub const binding: epoch.Binding = .{ .model = @splat(1), .drafter = @splat(2), .arithmetic = @splat(3), .layout = @splat(4), .media = @splat(5), .prefix = @splat(6) };
fn agreement(n: u64) epoch.Agreement {
    return .{ .world = 2, .epoch = n, .mark = 1025, .binding = binding };
}

const Directory = struct {
    path: [128]u8 = undefined,
    len: usize,
    fn init(label: []const u8) !Directory {
        var d: Directory = .{ .len = 0 };
        const p = try std.fmt.bufPrintSentinel(&d.path, "/tmp/tf-checkpoint-080-{d}-{s}", .{ sys.c.getpid(), label }, 0);
        d.len = p.len;
        if (sys.c.mkdir(p, 0o700) != 0) return error.TestDirectory;
        return d;
    }
    fn name(d: *const Directory) [:0]const u8 {
        return d.path[0..d.len :0];
    }
    fn deinit(d: *Directory) void {
        const fd = sys.c.opendir(d.name()) orelse unreachable;
        while (sys.c.readdir(fd)) |ent| {
            const n = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, n, ".") or std.mem.eql(u8, n, "..")) continue;
            var path: [1200]u8 = undefined;
            _ = sys.c.unlink(std.fmt.bufPrintSentinel(&path, "{s}/{s}", .{ d.name(), n }, 0) catch unreachable);
        }
        _ = sys.c.closedir(fd);
        _ = sys.c.rmdir(d.name());
    }
};

const Source = struct {
    offset: usize = 0,
    fn next(ptr: *anyopaque, limit: usize) !?cap.Chunk {
        const s: *Source = @ptrCast(@alignCast(ptr));
        const body = "native rank data, with bounded chunks and real files";
        if (s.offset == body.len) return null;
        const n = @min(limit, body.len - s.offset);
        const result: cap.Chunk = .{ .offset = s.offset, .len = n, .location = .{ .host = body[s.offset..][0..n] } };
        s.offset += n;
        return result;
    }
};
fn writeRank(dir: [:0]const u8, a: epoch.Agreement, number: u32) !rank.Local {
    return writeEnvelopeRank(dir, a, number, a);
}
fn writeEnvelopeRank(dir: [:0]const u8, a: epoch.Agreement, number: u32, saved_agreement: epoch.Agreement) !rank.Local {
    var pool = try kv.Pool.init(t.allocator, 0, 4096);
    defer pool.deinit();
    var budget: cap.Budget = .{ .pool = &pool, .limits = .{ .descriptors = 1, .unique_pages = 0, .pages_per_descriptor = 0, .capsule_bytes = 0 } };
    var lease = try budget.acquire(&.{}, 0, 0);
    defer lease.close() catch unreachable;
    var source: Source = .{};
    var slots: [2]stream.Slot = undefined;
    var storage: [32]u8 = undefined;
    const n = "native rank data, with bounded chunks and real files".len;
    var ring = try stream.Ring.init(.{ .owner = &lease, .ctx = &source, .len = n, .next = Source.next }, &storage, &slots, 16);
    defer ring.close() catch unreachable;
    var w = try rank.Writer(rows).open(dir, a.identity(number), 10, n, epoch.max_metadata);
    defer w.close();
    while (ring.emitted < n) {
        const ticket = try ring.acquire();
        @memcpy(ticket.target, ticket.chunk.location.host);
        try ring.dmaDone(ticket.key, true);
        try w.consume(&ring, ticket.key);
        try ring.retire(ticket.key);
    }
    const env: epoch.Envelope = .{ .rank = number, .agreement = saved_agreement, .payload = n, .detail_bytes = 5 };
    var bytes: [epoch.wire.size(epoch.Envelope)]u8 = undefined;
    try w.seal(&ring, @intCast(a.mark), &.{ try epoch.wire.encode(env, &bytes), "state" });
    return w.publish();
}
fn committed(store: Store, a: epoch.Agreement, slab: []u8) !Store.Epoch {
    var c = try Store.Coordinator.init(store, a);
    for (0..a.world) |i| try c.accept(try writeRank(store.dir, a, @intCast(i)), slab);
    return c.commit(22, slab);
}
fn pathFor(buf: []u8, dir: [:0]const u8, a: epoch.Agreement, number: ?u32) ![:0]const u8 {
    const key = if (number) |i| a.identity(i).name() else a.manifestName();
    const stem = std.fmt.bytesToHex(key, .lower);
    return rows.path(buf, dir, &stem, if (number == null) ".epoch" else ".rank", false);
}
fn expectEpoch(store: Store, slab: []u8, n: ?u64) !void {
    const found = try store.recover(binding, 2, slab);
    try t.expectEqual(n, if (found) |e| e.manifest.agreement.epoch else null);
}

test "checkpoint restore: only complete all-rank manifests survive restart; rank-local receipt is not durable" {
    var dir = try Directory.init("epochs");
    defer dir.deinit();
    var slab: [17]u8 = undefined; // deliberately unaligned payload cuts
    const store: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    const local = try writeRank(dir.name(), agreement(1), 0);
    try expectEpoch(store, &slab, null);
    var c = try Store.Coordinator.init(store, agreement(1));
    try c.accept(local, &slab);
    try t.expectError(error.CheckpointMissingRank, c.commit(11, &slab));
    try expectEpoch(store, &slab, null);
    _ = try committed(store, agreement(2), &slab);
    _ = try committed(store, agreement(3), &slab);
    try expectEpoch(store, &slab, 3);
    // No in-memory coordinator, receipt or pointer survives this read from a newly constructed store.
    const restart: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    var path: [1200]u8 = undefined;
    try t.expectEqual(@as(c_int, 0), sys.c.unlink(try pathFor(&path, dir.name(), agreement(3), 1)));
    try expectEpoch(restart, &slab, 2);
}

test "checkpoint restore: stale, mismatched and duplicate ranks poison the epoch without touching old commit" {
    var dir = try Directory.init("identity");
    defer dir.deinit();
    var slab: [31]u8 = undefined;
    const store: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(store, agreement(1), &slab);
    for (0..9) |kind| {
        const wanted = agreement(2 + kind);
        var wrong = wanted;
        switch (kind) {
            0 => wrong.epoch += 99,
            1 => wrong.mark += 1,
            2 => wrong.binding.layout[0] ^= 1,
            3 => wrong.binding.model[0] ^= 1,
            4 => wrong.binding.drafter[0] ^= 1,
            5 => wrong.binding.arithmetic[0] ^= 1,
            6 => wrong.binding.media[0] ^= 1,
            7 => wrong.binding.prefix[0] ^= 1,
            8 => wrong.world = 3,
            else => unreachable,
        }
        var c = try Store.Coordinator.init(store, wanted);
        try t.expectError(error.CheckpointRankIdentity, c.accept(try writeRank(dir.name(), wrong, 0), &slab));
        try t.expectError(error.CheckpointEpochState, c.commit(1, &slab));
        try expectEpoch(store, &slab, 1);
    }
    const a = agreement(20);
    var c = try Store.Coordinator.init(store, a);
    const local = try writeRank(dir.name(), a, 0);
    try c.accept(local, &slab);
    try t.expectError(error.CheckpointDuplicateRank, c.accept(local, &slab));
    var cancel = try Store.Coordinator.init(store, agreement(21));
    cancel.cancel();
    try t.expectError(error.CheckpointEpochState, cancel.commit(1, &slab));
    try expectEpoch(store, &slab, 1);
}

test "checkpoint restore: corrupt payload, metadata, manifest and truncated or overflowing trailer choose old epoch" {
    var dir = try Directory.init("corrupt");
    defer dir.deinit();
    var slab: [23]u8 = undefined;
    const store: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(store, agreement(1), &slab);
    for (0..6) |kind| {
        const a = agreement(2 + kind);
        _ = try committed(store, a, &slab);
        var path: [1200]u8 = undefined;
        const p = try pathFor(&path, dir.name(), a, if (kind == 3) null else 1);
        const fd = sys.c.open(p, .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
        try t.expect(fd >= 0);
        defer _ = sys.c.close(fd);
        const end = sys.c.lseek(fd, 0, sys.c.SEEK.END);
        if (kind == 2) {
            try t.expectEqual(@as(c_int, 0), sys.c.ftruncate(fd, end - 1));
        } else if (kind == 4) {
            const huge: [16]u8 = @splat(0xff);
            try t.expectEqual(@as(isize, 16), sys.c.pwrite(fd, &huge, huge.len, end - @sizeOf(rows.Trailer) + @offsetOf(rows.Trailer, "payload")));
        } else if (kind == 5) {
            try t.expectEqual(@as(c_int, 0), sys.c.ftruncate(fd, 0));
        } else {
            const offset: i64 = if (kind == 1) 60 else 0;
            var b: [1]u8 = undefined;
            try t.expectEqual(@as(isize, 1), sys.c.pread(fd, &b, 1, offset));
            b[0] ^= 0x77;
            try t.expectEqual(@as(isize, 1), sys.c.pwrite(fd, &b, 1, offset));
        }
        try expectEpoch(.{ .dir = dir.name(), .validate_detail = validateFixture }, &slab, 1);
    }
}

const Fault = struct {
    mode: enum { ok, nospace, rank_sync, manifest_sync, rank_dir, publish, publish_dir, cancel },
    syncs: usize = 0,
    dirs: usize = 0,
    linked: bool = false,
    fn of(ptr: ?*anyopaque) *Fault {
        return @ptrCast(@alignCast(ptr.?));
    }
    fn write(ptr: ?*anyopaque, fd: c_int, data: []const u8) !usize {
        if (of(ptr).mode == .nospace) return error.NoSpaceLeft;
        const n = sys.c.write(fd, data.ptr, @min(data.len, 13));
        if (n <= 0) return error.RowsWrite;
        return @intCast(n);
    }
    fn sync(ptr: ?*anyopaque, fd: c_int) !void {
        const f = of(ptr);
        f.syncs += 1;
        if ((f.mode == .rank_sync and f.syncs == 2) or (f.mode == .manifest_sync and f.syncs == 3)) return error.InjectedSync;
        if (f.syncs == 3) try t.expectEqual(@as(usize, 1), f.dirs);
        if (sys.c.fsync(fd) != 0) return error.RowsWrite;
    }
    fn directory(ptr: ?*anyopaque, fd: c_int) !void {
        const f = of(ptr);
        f.dirs += 1;
        try t.expectEqual(if (f.dirs == 1) @as(usize, 2) else 3, f.syncs);
        if ((f.mode == .rank_dir and f.dirs == 1) or (f.mode == .publish_dir and f.dirs == 2)) return error.InjectedDirectorySync;
        if (f.dirs == 2) try t.expect(f.linked);
        if (sys.c.fsync(fd) != 0) return error.RowsWrite;
    }
    fn link(ptr: ?*anyopaque, from: [:0]const u8, to: [:0]const u8) !void {
        const f = of(ptr);
        try t.expectEqual(@as(usize, 3), f.syncs);
        if (f.mode == .publish) return error.InjectedPublish;
        if (std.c.link(from, to) != 0) return error.CheckpointPublish;
        f.linked = true;
    }
    fn check(ptr: ?*anyopaque) !void {
        const f = of(ptr);
        if (f.mode == .cancel and f.linked) return error.Cancelled;
    }
};

test "checkpoint restore: ENOSPC, file and directory fsync, publication and cancel failures preserve old durable epoch" {
    var dir = try Directory.init("faults");
    defer dir.deinit();
    var slab: [29]u8 = undefined;
    const store: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(store, agreement(1), &slab);
    for ([_]@FieldType(Fault, "mode"){ .nospace, .rank_sync, .manifest_sync, .rank_dir, .publish, .publish_dir, .cancel, .ok }, 0..) |mode, i| {
        var f: Fault = .{ .mode = mode };
        var s = store;
        s.io = .{ .ctx = &f, .write = Fault.write, .sync = Fault.sync };
        s.publication = .{ .ctx = &f, .link = Fault.link, .sync_dir = Fault.directory };
        s.control = .{ .ctx = &f, .check_fn = Fault.check };
        const a = agreement(2 + i);
        var c = try Store.Coordinator.init(s, a);
        for (0..2) |rank_num| try c.accept(try writeRank(dir.name(), a, @intCast(rank_num)), &slab);
        const result = c.commit(90, &slab);
        if (mode == .ok) {
            _ = try result;
            try expectEpoch(store, &slab, a.epoch);
        } else {
            if (result) |_| return error.ExpectedFailure else |_| {}
            try expectEpoch(.{ .dir = dir.name(), .validate_detail = validateFixture }, &slab, 1);
        }
    }
}

test "checkpoint restore: a rank changed after receipt and a changed expected binding cannot be committed or recovered" {
    var dir = try Directory.init("race");
    defer dir.deinit();
    var slab: [19]u8 = undefined;
    const s: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(s, agreement(1), &slab);
    const a = agreement(2);
    var c = try Store.Coordinator.init(s, a);
    for (0..2) |i| try c.accept(try writeRank(dir.name(), a, @intCast(i)), &slab);
    var path: [1200]u8 = undefined;
    const fd = sys.c.open(try pathFor(&path, dir.name(), a, 0), .{ .ACCMODE = .WRONLY }, @as(sys.c.mode_t, 0));
    try t.expect(fd >= 0);
    defer _ = sys.c.close(fd);
    try t.expectEqual(@as(isize, 1), sys.c.pwrite(fd, "X", 1, 0));
    try t.expectError(error.RowsHash, c.commit(1, &slab));
    try expectEpoch(s, &slab, 1);
    var wrong = binding;
    wrong.arithmetic[0] ^= 1;
    try t.expectEqual(@as(?Store.Epoch, null), try s.recover(wrong, 2, &slab));
}

test "checkpoint restore: a correctly named rank with stale envelope is rejected after reading its real metadata" {
    var dir = try Directory.init("stale-envelope");
    defer dir.deinit();
    var slab: [37]u8 = undefined;
    const s: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(s, agreement(1), &slab);
    const a = agreement(2);
    const local = try writeEnvelopeRank(dir.name(), a, 0, agreement(1));
    var c = try Store.Coordinator.init(s, a);
    try t.expectError(error.CheckpointRankIdentity, c.accept(local, &slab));
    try expectEpoch(s, &slab, 1);
}

test "checkpoint restore: payload with a recomputed rows checksum still fails the committed SHA256 rank digest" {
    var dir = try Directory.init("rank-sha");
    defer dir.deinit();
    var slab: [37]u8 = undefined;
    const s: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(s, agreement(1), &slab);
    const a = agreement(2);
    _ = try committed(s, a, &slab);
    var path: [1200]u8 = undefined;
    const fd = sys.c.open(try pathFor(&path, dir.name(), a, 0), .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
    try t.expect(fd >= 0);
    defer _ = sys.c.close(fd);
    const end = sys.c.lseek(fd, 0, sys.c.SEEK.END);
    var trailer: rows.Trailer = undefined;
    try t.expectEqual(@as(isize, @sizeOf(rows.Trailer)), sys.c.pread(fd, std.mem.asBytes(&trailer), @sizeOf(rows.Trailer), end - @sizeOf(rows.Trailer)));
    var bytes: [128]u8 = undefined;
    try t.expectEqual(@as(isize, @intCast(trailer.payload)), sys.c.pread(fd, &bytes, @intCast(trailer.payload), 0));
    bytes[0] ^= 1;
    trailer.payload_hash = std.hash.Wyhash.hash(0, bytes[0..@intCast(trailer.payload)]);
    try t.expectEqual(@as(isize, @intCast(trailer.payload)), sys.c.pwrite(fd, &bytes, @intCast(trailer.payload), 0));
    try t.expectEqual(@as(isize, @sizeOf(rows.Trailer)), sys.c.pwrite(fd, std.mem.asBytes(&trailer), @sizeOf(rows.Trailer), end - @sizeOf(rows.Trailer)));
    try expectEpoch(.{ .dir = dir.name(), .validate_detail = validateFixture }, &slab, 1);
}

test "checkpoint restore: same epoch cannot publish two different marks under the same binding" {
    var dir = try Directory.init("same-epoch");
    defer dir.deinit();
    var slab: [37]u8 = undefined;
    const s: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(s, agreement(1), &slab);
    var wrong = agreement(1);
    wrong.mark += 1;
    try t.expectError(error.CheckpointPublish, committed(s, wrong, &slab));
    const recovered = (try s.recover(binding, 2, &slab)).?;
    try t.expectEqual(agreement(1).mark, recovered.manifest.agreement.mark);
}

const Crash = struct {
    fn write(_: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        if (sys.c.write(fd, bytes.ptr, @min(bytes.len, 37)) <= 0) sys.c._exit(2);
        sys.c._exit(86); // no defer runs: incomplete temp file actually survives this writer process
    }
};

test "checkpoint restore: crashed manifest writer leaves real temp bytes and a new reader process recovers old epoch" {
    var dir = try Directory.init("process-crash");
    defer dir.deinit();
    var slab: [37]u8 = undefined;
    const s: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
    _ = try committed(s, agreement(1), &slab);
    const a = agreement(2);
    var c = try Store.Coordinator.init(s, a);
    for (0..2) |i| try c.accept(try writeRank(dir.name(), a, @intCast(i)), &slab);
    const writer = sys.c.fork();
    if (writer < 0) return error.ForkFailed;
    if (writer == 0) {
        c.store.io.write = Crash.write;
        _ = c.commit(999, &slab) catch sys.c._exit(3);
        sys.c._exit(4);
    }
    var status: c_int = 0;
    try t.expectEqual(writer, sys.c.waitpid(writer, &status, 0));
    try t.expectEqual(@as(c_int, 86 << 8), status);
    var tmp_path: [1200]u8 = undefined;
    const stem = std.fmt.bytesToHex(a.manifestName(), .lower);
    const tmp = try std.fmt.bufPrintSentinel(&tmp_path, "{s}/{s}.epoch.3e7.part", .{ dir.name(), stem }, 0);
    const fd = sys.c.open(tmp, .{ .ACCMODE = .RDONLY }, @as(sys.c.mode_t, 0));
    try t.expect(fd >= 0);
    defer _ = sys.c.close(fd);
    try t.expectEqual(@as(i64, 37), sys.c.lseek(fd, 0, sys.c.SEEK.END));
    const reader = sys.c.fork();
    if (reader < 0) return error.ForkFailed;
    if (reader == 0) {
        const fresh: Store = .{ .dir = dir.name(), .validate_detail = validateFixture };
        const recovered = fresh.recover(binding, 2, &slab) catch sys.c._exit(5);
        if (recovered == null or recovered.?.manifest.agreement.epoch != 1) sys.c._exit(6);
        sys.c._exit(0);
    }
    try t.expectEqual(reader, sys.c.waitpid(reader, &status, 0));
    try t.expectEqual(@as(c_int, 0), status);
}

test "checkpoint restore: wire bytes use explicit little endian fields without padding or pointer representations" {
    const Example = struct { tag: u8, word: u32, signed: i64, small: [2]u16 };
    const value: Example = .{ .tag = 0xaa, .word = 0x12345678, .signed = -2, .small = .{ 0x1234, 0x5678 } };
    const expected = [_]u8{ 0xaa, 0x78, 0x56, 0x34, 0x12, 0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x34, 0x12, 0x78, 0x56 };
    var out: [17]u8 = undefined;
    try t.expectEqualSlices(u8, &expected, try epoch.wire.encode(value, &out));
    try t.expectEqualDeep(value, try epoch.wire.decode(Example, &expected));
    try t.expectError(error.CheckpointMetadataSize, epoch.wire.decode(Example, expected[0..16]));
    try t.expectError(error.CheckpointMetadataLimit, epoch.wire.encode(value, out[0..16]));
}
