const std = @import("std");
const sys = @import("sys");
const x = @import("fixture.zig");
const Fault = @import("store_test.zig").Fault;
const t = std.testing;
extern "c" fn socketpair(c_int, c_int, c_int, *[2]c_int) c_int;
const Poll = extern struct { fd: c_int, events: c_short, revents: c_short = 0 };
extern "c" fn poll([*]Poll, c_uint, c_int) c_int;

const Link = struct {
    fd: c_int,
    rank: u32,
    largest: usize = 0,
    bytes: usize = 0,
    tamper: enum { none, duplicate, stale, wrong_scope } = .none,
    fn transfer(fd: c_int, bytes: []u8, writing: bool) !void {
        var done: usize = 0;
        while (done < bytes.len) {
            var p: [1]Poll = .{.{ .fd = fd, .events = if (writing) 4 else 1 }};
            if (poll(&p, 1, 5000) <= 0) return error.PeerTimeout;
            const n = if (writing) sys.c.send(fd, bytes[done..].ptr, @min(4096, bytes.len - done), sys.c.MSG.NOSIGNAL) else sys.c.read(fd, bytes[done..].ptr, bytes.len - done);
            if (n < 0 and sys.c.errno(n) == .INTR) continue;
            if (n <= 0) return error.PeerLost;
            done += @intCast(n);
        }
    }
    fn exchange(ptr: *anyopaque, _: x.protocol.Phase, _: u64, send: []const u8, recv: []u8) !void {
        const link: *Link = @ptrCast(@alignCast(ptr));
        if (recv.len != 2 * send.len) return error.TestTransport;
        link.largest = @max(link.largest, send.len);
        link.bytes += send.len;
        const own = recv[link.rank * send.len ..][0..send.len];
        const peer = recv[(1 - link.rank) * send.len ..][0..send.len];
        @memcpy(own, send);
        // Opposite ordering avoids socket-buffer deadlock, including a maximally sized manifest frame.
        if (link.rank == 0) {
            try transfer(link.fd, own, true);
            try transfer(link.fd, peer, false);
        } else {
            try transfer(link.fd, peer, false);
            try transfer(link.fd, own, true);
        }
        switch (link.tamper) {
            .none => {},
            .duplicate => @memcpy(peer, own),
            .stale => peer[72] ^= 1, // canonical round field, after version/session/scope/phase
            .wrong_scope => peer[36] ^= 1,
        }
    }
    fn adapter(link: *Link) x.protocol.Transport {
        return .{ .ctx = link, .world = 2, .rank = link.rank, .session = @splat(7), .scope = @splat(8), .exchange = exchange };
    }
};

pub const Job = struct {
    dirs: [2][:0]const u8,
    receipts: [2]x.f.Receipt,
    recover: bool = false,
    expected: ?u64 = null,
    expected_name: ?x.f.Name = null,
    failure: ?[2]anyerror = null,
    crash: bool = false,
    fsync_failure: bool = false,
    directory_failure: bool = false,
    cancelled: bool = false,
    read_failure: bool = false,
    missing_receipt: bool = false,
    tamper: @FieldType(Link, "tamper") = .none,
};

const Crash = struct {
    fn write(_: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        if (sys.c.write(fd, bytes.ptr, @min(bytes.len, 37)) != 37) sys.c._exit(85);
        sys.c._exit(86);
    }
};

fn readFailure(_: ?*anyopaque, _: c_int, _: []u8, _: u64) !usize {
    return error.BlockReadIO;
}
fn expectFailure(rank: u32, wanted: anyerror, actual: anyerror) !void {
    if (wanted != actual) {
        std.debug.print("rank {d}: expected {s}, got {s}\n", .{ rank, @errorName(wanted), @errorName(actual) });
        return error.WrongProtocolFailure;
    }
}
fn rankJob(job: Job, rank: u32, fd: c_int) !void {
    var files = try x.fs.Files.open(job.dirs[rank], 1 << 30, 700 + rank);
    defer files.close();
    var slab: [65537]u8 = undefined;
    var link: Link = .{ .fd = fd, .rank = rank, .tamper = job.tamper };
    var transport = link.adapter();
    var fault: Fault = .{ .fail_sync = if (job.fsync_failure) 1 else 0, .fail_dir = if (job.directory_failure) 2 else 0, .cancel = job.cancelled };
    if (rank == 1 and job.crash) files.io.write = Crash.write;
    if (rank == 1 and (job.fsync_failure or job.directory_failure or job.cancelled)) fault.attach(&files);
    if (rank == 1 and job.read_failure) files.read_io.call = readFailure;
    if (job.recover) {
        const epoch = x.protocol.recover(&files, x.selection(&job.receipts[rank].agreement.binding), &transport, &slab, 16) catch |err| {
            if (job.failure) |wanted| return expectFailure(rank, wanted[rank], err);
            return err;
        };
        if (job.failure != null) return error.UnexpectedAck;
        try t.expectEqual(job.expected, if (epoch) |e| e.manifest.agreement.epoch else null);
        if (job.expected_name) |name| try t.expectEqualDeep(name, epoch.?.manifest.agreement.manifestName());
    } else {
        const receipt: anyerror!?x.f.Receipt = if (rank == 1 and job.missing_receipt) error.LocalWorkerFailed else job.receipts[rank];
        _ = x.protocol.commit(&files, job.receipts[rank].agreement, receipt, &transport, &slab) catch |err| {
            if (job.failure) |wanted| return expectFailure(rank, wanted[rank], err);
            return err;
        };
        if (job.failure != null) return error.UnexpectedAck;
    }
    try t.expect(link.largest < 4096);
    try t.expect(link.bytes < 128 << 10);
}

/// Real processes exchange only wire records; each opens only its own independent NVMe stand-in directory.
pub fn pair(job: Job) !void {
    var sockets: [2]c_int = undefined;
    if (socketpair(1, 1, 0, &sockets) != 0) return error.SocketPair;
    const child = sys.c.fork();
    if (child < 0) return error.ForkFailed;
    if (child == 0) {
        _ = sys.c.close(sockets[0]);
        rankJob(job, 1, sockets[1]) catch |err| {
            std.debug.print("child protocol failure: {s}\n", .{@errorName(err)});
            sys.c._exit(87);
        };
        _ = sys.c.close(sockets[1]);
        sys.c._exit(0);
    }
    _ = sys.c.close(sockets[1]);
    const result = rankJob(job, 0, sockets[0]);
    _ = sys.c.close(sockets[0]);
    var status: c_int = 0;
    try t.expectEqual(child, sys.c.waitpid(child, &status, 0));
    try t.expectEqual(@as(c_int, if (job.crash) 86 << 8 else 0), status);
    try result;
}

pub fn prepare(fixture: *x.Fixture, dirs: [2][:0]const u8) ![2]x.f.Receipt {
    var receipts: [2]x.f.Receipt = undefined;
    for (dirs, 0..) |dir, rank| {
        var files = try x.fs.Files.open(dir, 1 << 30, fixture.agreement.epoch * 100);
        defer files.close();
        receipts[rank] = try fixture.save(&files, @intCast(rank));
    }
    return receipts;
}

fn disagree(dir: [:0]const u8, agreement: x.f.epoch.Agreement) !void {
    var files = try x.fs.Files.open(dir, 1 << 30, 900);
    defer files.close();
    var m = try x.local.readManifest(&files, agreement.manifestName());
    // Rank 1's own files still validate, but its view of rank 0 differs. This must fall back collectively.
    m.ranks[0].digest[0] ^= 1;
    var path: [1200]u8 = undefined;
    const fd = sys.c.open(try files.path(&path, agreement.manifestName(), ".commit"), .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = sys.c.close(fd);
    var bytes: [x.f.wire.size(x.f.epoch.Manifest)]u8 = undefined;
    const data = try x.f.wire.encode(m, &bytes);
    var trailer: x.f.rows.Trailer = undefined;
    if (sys.c.pread(fd, std.mem.asBytes(&trailer), @sizeOf(x.f.rows.Trailer), @intCast(data.len)) != @sizeOf(x.f.rows.Trailer)) return error.TestFile;
    trailer.payload_hash = std.hash.Wyhash.hash(0, data);
    if (sys.c.pwrite(fd, data.ptr, data.len, 0) != data.len or sys.c.pwrite(fd, std.mem.asBytes(&trailer), @sizeOf(x.f.rows.Trailer), @intCast(data.len)) != @sizeOf(x.f.rows.Trailer)) return error.TestFile;
    try files.io.sync(null, fd);
}

test "block store: separate processes commit and recover common epochs without shared directory reads" {
    var fixture: x.Fixture = undefined;
    try fixture.init(true, true, false, 2);
    defer fixture.deinit();
    var a = try x.Directory.init("rank0");
    defer a.deinit();
    var b = try x.Directory.init("rank1");
    defer b.deinit();
    const dirs = .{ a.name(), b.name() };
    const first = try prepare(&fixture, dirs);
    try pair(.{ .dirs = dirs, .receipts = first });
    try pair(.{ .dirs = dirs, .receipts = first, .recover = true, .expected = 1 });
    try fixture.mark(1025);
    fixture.agreement.epoch = 2;
    const next = try prepare(&fixture, dirs);
    // All-local durable files are still not a recovery epoch until every local manifest exists.
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 1 });
    try pair(.{ .dirs = dirs, .receipts = next, .directory_failure = true, .failure = .{ error.BlockCollectiveIO, error.BlockCollectiveIO } });
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 1 });
    try pair(.{ .dirs = dirs, .receipts = next });
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 2 });
    try disagree(b.name(), next[1].agreement);
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 1 });
    try disagree(b.name(), next[1].agreement); // restore the original manifest for the missing-file control
    // Delete rank 1's latest capsule, while every file on rank 0 is still valid.
    var files = try x.fs.Files.open(b.name(), 1 << 30, 500);
    var path: [1200]u8 = undefined;
    try t.expectEqual(@as(c_int, 0), sys.c.unlink(try files.path(&path, next[1].agreement.identity(1).name(), ".cap")));
    files.close();
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 1 });
    files = try x.fs.Files.open(b.name(), 1 << 30, 600);
    try t.expectEqual(@as(c_int, 0), sys.c.unlink(try files.path(&path, first[1].agreement.identity(1).name(), ".cap")));
    files.close();
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = null });
}

test "block store: interrupted rank manifest write requires common older epoch after process restart" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 2);
    defer fixture.deinit();
    var a = try x.Directory.init("crash0");
    defer a.deinit();
    var b = try x.Directory.init("crash1");
    defer b.deinit();
    const dirs = .{ a.name(), b.name() };
    const first = try prepare(&fixture, dirs);
    try pair(.{ .dirs = dirs, .receipts = first });
    try fixture.mark(1025);
    fixture.agreement.epoch = 2;
    const next = try prepare(&fixture, dirs);
    try pair(.{ .dirs = dirs, .receipts = next, .failure = .{ error.PeerLost, error.PeerLost }, .crash = true });
    var files = try x.fs.Files.open(b.name(), 1 << 30, 900);
    var path: [1200]u8 = undefined;
    const stem = std.fmt.bytesToHex(next[1].agreement.manifestName(), .lower);
    const temp = try std.fmt.bufPrintSentinel(&path, "{s}/{s}.commit.2be.part", .{ b.name(), stem }, 0);
    const fd = sys.c.open(temp, .{ .ACCMODE = .RDONLY }, @as(sys.c.mode_t, 0));
    try t.expect(fd >= 0);
    try t.expectEqual(@as(i64, 37), sys.c.lseek(fd, 0, sys.c.SEEK.END));
    _ = sys.c.close(fd);
    var slab: [65537]u8 = undefined;
    try x.local.collect(&files, &slab);
    files.close();
    try pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 1 });
}

test "block store: missing failed duplicate stale or disagreeing ranks cannot acknowledge" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 2);
    defer fixture.deinit();
    var a = try x.Directory.init("votes0");
    defer a.deinit();
    var b = try x.Directory.init("votes1");
    defer b.deinit();
    const dirs = .{ a.name(), b.name() };
    const first = try prepare(&fixture, dirs);
    try pair(.{ .dirs = dirs, .receipts = first, .failure = .{ error.BlockCollectiveAbort, error.BlockCollectiveAbort }, .missing_receipt = true });
    try pair(.{ .dirs = dirs, .receipts = first, .failure = .{ error.BlockCollectiveIO, error.BlockCollectiveIO }, .fsync_failure = true });
    for ([_]@FieldType(Link, "tamper"){ .duplicate, .stale, .wrong_scope }) |tamper| {
        try pair(.{ .dirs = dirs, .receipts = first, .failure = .{ error.BlockTransport, error.BlockTransport }, .tamper = tamper });
    }
    try pair(.{ .dirs = dirs, .receipts = first, .cancelled = true, .failure = .{ error.BlockCollectiveCancelled, error.BlockCollectiveCancelled } });
    try pair(.{ .dirs = dirs, .receipts = first, .read_failure = true, .failure = .{ error.BlockCollectiveIO, error.BlockCollectiveIO } });
    var wrong = first;
    wrong[1].agreement.mark += 1;
    try pair(.{ .dirs = dirs, .receipts = wrong, .failure = .{ error.BlockCollectiveAbort, error.BlockCollectiveAbort } });
    wrong = first;
    wrong[1].agreement.epoch += 1;
    try pair(.{ .dirs = dirs, .receipts = wrong, .failure = .{ error.BlockCollectiveAbort, error.BlockCollectiveAbort } });
    try pair(.{ .dirs = dirs, .receipts = first });
    try pair(.{ .dirs = dirs, .receipts = first, .recover = true, .expected = 1 });
    try pair(.{ .dirs = dirs, .receipts = first, .recover = true, .read_failure = true, .failure = .{ error.BlockCollectiveIO, error.BlockCollectiveIO } });
    try pair(.{ .dirs = dirs, .receipts = first, .recover = true, .cancelled = true, .failure = .{ error.BlockCollectiveCancelled, error.BlockCollectiveCancelled } });
}
