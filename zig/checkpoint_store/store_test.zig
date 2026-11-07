const std = @import("std");
const sys = @import("sys");
const x = @import("fixture.zig");
const t = std.testing;
extern "c" fn getenv([*:0]const u8) ?[*:0]const u8;

fn measurement(kv8: bool, fixture: *x.Fixture) !void {
    const path = getenv("TF_GLM_BLOCK_RECEIPT") orelse return;
    const fd = sys.c.open(path, .{ .ACCMODE = .WRONLY, .APPEND = true, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
    if (fd < 0) return error.TestReceipt;
    defer _ = sys.c.close(fd);
    var buf: [512]u8 = undefined;
    const line = try std.fmt.bufPrint(&buf, "{{\"fp8\":{s},\"mark\":1025,\"logical_bytes\":{d},\"new_page_bytes\":{d},\"capsule_bytes\":{d},\"transfer_bytes\":{d},\"max_transfer\":{d},\"new_pages\":{d},\"reused_pages\":{d}}}\n", .{ if (kv8) "true" else "false", fixture.saved.?.plan.len, fixture.stats.immutable_written, fixture.stats.mutable_written, fixture.stats.transferred, fixture.stats.largest_transfer, fixture.stats.new_pages, fixture.stats.reused_pages });
    if (sys.c.write(fd, line.ptr, line.len) != line.len) return error.TestReceipt;
}

test "block store: actual GLM growing marks and a token fork reuse shared pages and restore old state" {
    for ([_]bool{ false, true }) |kv8| {
        var fixture: x.Fixture = undefined;
        try fixture.init(kv8, true, false, 1);
        defer fixture.deinit();
        var dir = try x.Directory.init("grow");
        defer dir.deinit();
        var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
        defer files.close();
        const old_bytes = try fixture.oracle();
        defer t.allocator.free(old_bytes);
        const first = try fixture.save(&files, 0);
        const old = try x.commitOne(&files, first, fixture.slab.slice(u8));
        try t.expectEqual(@as(u64, 1), fixture.stats.new_pages);
        try t.expectEqual(fixture.saved.?.plan.len, fixture.stats.transferred);
        var one = try x.local.Point.open(&files, first.agreement, 0, first.point, fixture.slab.slice(u8));
        defer one.close();
        const shared = try one.page(0);
        try fixture.mark(1025);
        fixture.agreement.epoch = 2;
        const next_bytes = try fixture.oracle();
        defer t.allocator.free(next_bytes);
        const second = try fixture.save(&files, 0);
        const next = try x.commitOne(&files, second, fixture.slab.slice(u8));
        try t.expectEqual(@as(u64, 1), fixture.stats.new_pages);
        try t.expectEqual(@as(u64, 1), fixture.stats.reused_pages);
        try t.expectEqual(fixture.stats.immutable_written + fixture.stats.mutable_written, fixture.stats.transferred);
        try t.expectEqual(fixture.saved.?.plan.len - fixture.stats.immutable_written, fixture.stats.transferred);
        try t.expect(fixture.stats.largest_transfer <= fixture.slab.slice(u8).len);
        try measurement(kv8, &fixture);
        var two = try x.local.Point.open(&files, second.agreement, 0, second.point, fixture.slab.slice(u8));
        defer two.close();
        try t.expectEqualDeep(shared, try two.page(0));
        try fixture.restore(&next, next_bytes);
        // The first complete page is shared by the fork. Its second page and mutable state really change.
        fixture.tokens[700] ^= 63;
        x.fill(x.rec.bytes(fixture.device.ptr + @as(u64, fixture.live.ids.items[1]) * fixture.pool.page_bytes, fixture.pool.page_bytes), 97);
        try fixture.mark(1537);
        fixture.agreement.epoch = 3;
        fixture.agreement.binding.prefix[0] ^= 1;
        const fork_bytes = try fixture.oracle();
        defer t.allocator.free(fork_bytes);
        const third = try fixture.save(&files, 0);
        const fork = try x.commitOne(&files, third, fixture.slab.slice(u8));
        try t.expectEqual(@as(u64, 2), fixture.stats.new_pages);
        try t.expectEqual(@as(u64, 1), fixture.stats.reused_pages);
        try fixture.restore(&fork, fork_bytes);
        try fixture.restore(&old, old_bytes);
        const used = files.used;
        try x.local.collect(&files, fixture.slab.slice(u8));
        try t.expectEqual(used, files.used);
        try t.expectError(error.BlockAlreadyExists, fixture.save(&files, 0));
        try t.expectEqual(@as(u64, 0), files.reserved);
        try t.expectEqual(@as(usize, 0), fixture.saved.?.pages.readers);
    }
}

test "block store: real 40 MiB DFlash capsule resumes directly without prompt-cache admission" {
    var fixture: x.Fixture = undefined;
    try fixture.init(true, false, true, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("draft");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const expected = try fixture.oracle();
    defer t.allocator.free(expected);
    const receipt = try fixture.save(&files, 0);
    const epoch = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    try fixture.restore(&epoch, expected);
    try t.expectEqual(@as(u64, 40 << 20), fixture.metadata.draft_bytes);
    try t.expect(fixture.stats.largest_transfer <= 65537);
}

pub const Fault = struct {
    writes: usize = 0,
    syncs: usize = 0,
    dirs: usize = 0,
    fail_write: bool = false,
    fail_sync: usize = 0,
    fail_dir: usize = 0,
    fail_link: bool = false,
    cancel: bool = false,
    checks: usize = 0,
    cancel_at: usize = 0,
    pub fn write(ptr: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        const fault: *Fault = @ptrCast(@alignCast(ptr.?));
        fault.writes += 1;
        if (fault.fail_write and fault.writes == 2) return error.NoSpaceLeft;
        return (x.f.rows.IO{}).write(null, fd, bytes);
    }
    pub fn sync(ptr: ?*anyopaque, fd: c_int) !void {
        const fault: *Fault = @ptrCast(@alignCast(ptr.?));
        fault.syncs += 1;
        if (fault.syncs == fault.fail_sync) return error.InjectedFsync;
        try (x.f.rows.IO{}).sync(null, fd);
    }
    pub fn link(ptr: ?*anyopaque, from: [:0]const u8, to: [:0]const u8) !void {
        const fault: *Fault = @ptrCast(@alignCast(ptr.?));
        if (fault.fail_link) return error.InjectedLink;
        try (x.core.checkpoint_file.Publication{}).link(null, from, to);
    }
    pub fn syncDir(ptr: ?*anyopaque, fd: c_int) !void {
        const fault: *Fault = @ptrCast(@alignCast(ptr.?));
        fault.dirs += 1;
        if (fault.dirs == fault.fail_dir) return error.InjectedDirectoryFsync;
        try (x.core.checkpoint_file.Publication{}).sync_dir(null, fd);
    }
    pub fn check(ptr: ?*anyopaque) !void {
        const fault: *Fault = @ptrCast(@alignCast(ptr.?));
        fault.checks += 1;
        if (fault.cancel or (fault.cancel_at != 0 and fault.checks >= fault.cancel_at)) return error.Cancelled;
    }
    pub fn attach(fault: *Fault, files: *x.fs.Files) void {
        files.io = .{ .ctx = fault, .write = write, .sync = sync };
        files.publication = .{ .ctx = fault, .link = link, .sync_dir = syncDir };
        files.control = .{ .ctx = fault, .check_fn = check };
    }
};

test "block store: ENOSPC and each file/publication barrier preserve an older committed point" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("io-faults");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const bytes = try fixture.oracle();
    defer t.allocator.free(bytes);
    const receipt = try fixture.save(&files, 0);
    const old = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    const old_used = files.used;
    try fixture.mark(1025);
    for (0..8) |i| {
        fixture.agreement.epoch = i + 2;
        var fault: Fault = switch (i) {
            0 => .{ .fail_write = true },
            1 => .{ .fail_sync = 1 },
            2 => .{ .fail_sync = 2 },
            3 => .{ .fail_sync = 3 },
            4 => .{ .fail_link = true },
            5 => .{ .fail_dir = 1 },
            6 => .{ .fail_dir = 2 },
            7 => .{ .fail_dir = 3 },
            else => unreachable,
        };
        fault.attach(&files);
        if (fixture.save(&files, 0)) |_| return error.ExpectedFailure else |_| {}
        files.io = .{};
        files.publication = .{};
        files.control = .{};
        try t.expectEqual(@as(u64, 0), files.reserved);
        try t.expectEqual(@as(usize, 0), fixture.saved.?.pages.readers);
        try x.local.collect(&files, fixture.slab.slice(u8));
        try t.expectEqual(old_used, files.used);
        try fixture.restore(&old, bytes);
    }
}

test "block store: quota and admission reject before writes; allocation DMA cancellation and worker failure drain" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("admit");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    try t.expectError(error.BlockBusy, x.fs.Files.open(dir.name(), 1 << 30, 200));
    fixture.admission_fail = true;
    try t.expectError(error.AdmissionRefused, fixture.save(&files, 0));
    fixture.admission_fail = false;
    files.quota = 1;
    try t.expectError(error.BlockQuota, fixture.save(&files, 0));
    files.quota = 1 << 30;
    var failing = std.testing.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    var request = fixture.request(0);
    request.gpa = failing.allocator();
    try t.expectError(error.OutOfMemory, x.capture.save(.experimental, &files, request));
    var fault: Fault = .{ .cancel = true };
    fault.attach(&files);
    try t.expectError(error.Cancelled, fixture.save(&files, 0));
    fault.cancel = false;
    fault.checks = 0;
    fault.cancel_at = 20;
    try t.expectError(error.Cancelled, fixture.save(&files, 0));
    files.control = .{};
    files.io = .{};
    files.publication = .{};
    x.rec.fail_copy = x.rec.copies + 2;
    if (fixture.save(&files, 0)) |_| return error.ExpectedFailure else |_| {}
    x.rec.fail_copy = null;
    try t.expectEqual(@as(usize, 0), fixture.saved.?.pages.readers);
    try t.expectEqual(@as(usize, 1), fixture.budget.active);
    try t.expectEqual(@as(u64, 0), files.reserved);
    try x.local.collect(&files, fixture.slab.slice(u8));
    try t.expectEqual(@as(u64, 0), files.used);
    const receipt = try fixture.save(&files, 0);
    const epoch = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    var reader = try epoch.reader(fixture.slab.slice(u8));
    defer reader.close();
    try t.expectError(error.BlockBusy, x.local.collect(&files, fixture.slab.slice(u8)));
}

test "block store: post-selection corrupt page and capsule fail segment SHA before restore ready" {
    for ([_]bool{ false, true }) |capsule| {
        var fixture: x.Fixture = undefined;
        try fixture.init(false, false, false, 1);
        defer fixture.deinit();
        var dir = try x.Directory.init("corruption");
        defer dir.deinit();
        var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
        defer files.close();
        const receipt = try fixture.save(&files, 0);
        const epoch = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
        var reader = try epoch.reader(fixture.slab.slice(u8));
        defer reader.close();
        const obj = if (capsule) reader.point.header.capsule else try reader.point.page(0);
        var path: [1200]u8 = undefined;
        const fd = sys.c.open(try files.path(&path, obj.name, if (capsule) ".cap" else ".blk"), .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
        try t.expect(fd >= 0);
        defer _ = sys.c.close(fd);
        var byte: [1]u8 = undefined;
        try t.expectEqual(@as(isize, 1), sys.c.pread(fd, &byte, 1, 0));
        byte[0] ^= 17;
        try t.expectEqual(@as(isize, 1), sys.c.pwrite(fd, &byte, 1, 0));
        while (true) {
            const n = reader.read(fixture.slab.slice(u8)) catch |err| {
                try t.expectEqual(error.BlockHash, err);
                break;
            };
            if (n == 0) return error.AcceptedCorruption;
        }
        try t.expect(!reader.finished);
        var transport = x.single();
        try t.expectEqual(@as(?x.protocol.Epoch, null), try x.protocol.recover(&files, x.selection(&fixture.agreement.binding), &transport, fixture.slab.slice(u8), 8));
    }
}
