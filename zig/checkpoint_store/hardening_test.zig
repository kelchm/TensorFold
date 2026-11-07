//! Regression gates for identity, receipt lifetime, bounded work, and repair from trusted live bytes.
const std = @import("std");
const sys = @import("sys");
const x = @import("fixture.zig");
const names = @import("names.zig");
const Fault = @import("store_test.zig").Fault;
const corrupt = @import("integrity_test.zig").corruptWithRowsHash;
const t = std.testing;

fn headRow(f: *x.Fixture, salt: usize) !void {
    for (f.saved.?.plan.parts[0..f.saved.?.plan.count]) |p| {
        if (p.span != 0 and p.live == f.st.mtp_caches.lc) {
            const row = p.span / 512;
            const piece = f.saved.?.plan.piece(p, 511 * row);
            x.fill(x.rec.bytes(piece.live, row), salt);
            return;
        }
    }
    return error.NoHeadPart;
}
fn bind(f: *x.Fixture) !void {
    f.agreement.binding = f.metadata.bind(f.agreement.binding);
    f.agreement.binding.prefix = try names.prefix(f.input(), &f.metadata);
}
fn boundaryMedia() x.glm.checkpoint_restore.Media {
    var media: x.glm.checkpoint_restore.Media = .{ .processor = @splat(7), .count = 1 };
    media.images[0] = .{ .digest = @splat(8), .at = 512, .tokens = 4, .gh = 4, .gw = 4 };
    return media;
}

test "hardening: MTP boundary token and boundary image forks restore every byte, including head row 511" {
    for ([_]bool{ false, true }) |image| {
        var fixture: x.Fixture = undefined;
        try fixture.init(false, true, false, 1);
        defer fixture.deinit();
        var dir = try x.Directory.init("hardening-head");
        defer dir.deinit();
        var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
        defer files.close();
        if (image) {
            fixture.metadata.media = boundaryMedia();
            try bind(&fixture);
        }
        const old_bytes = try fixture.oracle();
        defer t.allocator.free(old_bytes);
        const old = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
        var before: [1]x.f.Name = undefined;
        var after: [1]x.f.Name = undefined;
        try names.build(fixture.input(), &fixture.metadata, &before);
        if (image) fixture.metadata.media.images[0].digest[0] ^= 1 else fixture.tokens[512] ^= 63;
        try names.build(fixture.input(), &fixture.metadata, &after);
        try t.expect(!std.mem.eql(u8, &before[0], &after[0]));
        try headRow(&fixture, 97);
        try fixture.mark(1025);
        if (image) {
            fixture.metadata.media = boundaryMedia();
            fixture.metadata.media.images[0].digest[0] ^= 1;
        }
        try bind(&fixture);
        fixture.agreement.epoch = 2;
        const expected = try fixture.oracle();
        defer t.allocator.free(expected);
        const fresh = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
        try t.expectEqual(@as(u64, 0), fixture.stats.reused_pages);
        try fixture.restore(&fresh, expected);
        try fixture.restore(&old, old_bytes);
    }
}

test "hardening: explicit GLM rules, canonical spans, mark lookahead, and binding reject aliases before writes" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, true, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-identity");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    try t.expect(names.rules(&fixture.metadata).planned);
    try t.expectEqual(@as(u32, 1), names.rules(&fixture.metadata).lookahead);
    var request = fixture.request(0);
    request.input.tokens = fixture.tokens[0..512];
    try t.expectError(error.BlockIdentity, x.capture.save(.experimental, &files, request));
    request = fixture.request(0);
    request.input.spans = &.{.{ 2, 8 }};
    try t.expectError(error.BlockIdentity, x.capture.save(.experimental, &files, request));
    var before: [1]x.f.Name = undefined;
    var after: [1]x.f.Name = undefined;
    try names.build(fixture.input(), &fixture.metadata, &before);
    try names.build(request.input, &fixture.metadata, &after);
    try t.expect(!std.mem.eql(u8, &before[0], &after[0]));
    request.input.spans = &.{ .{ 2, 8 }, .{ 8, 9 } };
    try t.expectError(error.BlockIdentity, names.prefix(request.input, &fixture.metadata));
    request.input = fixture.input();
    request.input.starts = &.{ 0, 256, 512 };
    try names.build(request.input, &fixture.metadata, &after);
    try t.expect(!std.mem.eql(u8, &before[0], &after[0]));
    try t.expectError(error.BlockIdentity, x.capture.save(.experimental, &files, request));
    request.input.starts = &.{};
    try t.expectError(error.BlockIdentity, names.prefix(request.input, &fixture.metadata));
    fixture.tokens[512] ^= 1;
    try t.expectError(error.BlockIdentity, fixture.save(&files, 0));
    try t.expectEqual(@as(u64, 0), files.used);
    fixture.tokens[512] ^= 1;
    const prefix = try names.prefix(fixture.input(), &fixture.metadata);
    fixture.tokens[900] ^= 1;
    try t.expectEqualDeep(prefix, try names.prefix(fixture.input(), &fixture.metadata));
}

test "hardening: pending receipt pins saved data; abandonment and post-link cancellation permit exact same-point retry" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-pending");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const receipt = try fixture.save(&files, 0);
    const used = files.used;
    try t.expectError(error.BlockBusy, x.local.collect(&files, fixture.slab.slice(u8)));
    try t.expectEqual(used, files.used);
    try t.expectError(error.BlockBusy, fixture.save(&files, 0));
    try files.abandon(receipt);
    try t.expectError(error.BlockReceipt, files.abandon(receipt));
    try x.local.collect(&files, fixture.slab.slice(u8));
    try t.expectEqual(@as(u64, 0), files.used);
    const bytes = try fixture.oracle();
    defer t.allocator.free(bytes);
    const old = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
    try fixture.mark(1025);
    fixture.agreement.epoch = 2;
    var fault: Fault = .{ .cancel_after_point_link = true };
    fault.attach(&files);
    try t.expectError(error.Cancelled, fixture.save(&files, 0));
    files.control = .{};
    files.io = .{};
    files.publication = .{};
    try t.expect(files.pending == null and !files.transaction);
    try t.expect(!try files.rawExists(fixture.agreement.identity(0).name(), ".point"));
    try t.expect(!try files.rawExists(fixture.agreement.identity(0).name(), ".cap"));
    const expected = try fixture.oracle();
    defer t.allocator.free(expected);
    const retried = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
    try fixture.restore(&retried, expected);
    try fixture.restore(&old, bytes);
}

fn firstPage(files: *x.fs.Files, receipt: x.f.Receipt, slab: []u8) !x.f.Object {
    var point = try x.local.Point.open(files, receipt.agreement, receipt.rank, receipt.point, slab);
    defer point.close();
    return point.page(0);
}

test "hardening: repair matches every retained reference, fences readers, and preserves both published epochs" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-repair");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const old_bytes = try fixture.oracle();
    defer t.allocator.free(old_bytes);
    const receipt = try fixture.save(&files, 0);
    const old = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    const shared = try firstPage(&files, receipt, fixture.slab.slice(u8));
    try fixture.mark(1025);
    fixture.agreement.epoch = 2;
    const next_bytes = try fixture.oracle();
    defer t.allocator.free(next_bytes);
    const next = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
    var path: [1200]u8 = undefined;
    try corrupt(try files.path(&path, shared.name, ".blk"));
    fixture.agreement.epoch = 3;
    var reader = try old.reader(fixture.slab.slice(u8));
    try t.expectError(error.BlockRepairBusy, fixture.save(&files, 0));
    reader.close();
    // An altered live capture cannot bless the damaged inode under the retained token identity.
    const address = paged: {
        for (fixture.saved.?.plan.parts[0..fixture.saved.?.plan.count]) |part| if (part.span != 0) {
            break :paged fixture.saved.?.plan.piece(part, 0).live;
        };
        return error.NoPage;
    };
    x.rec.bytes(address, 1)[0] ^= 1;
    try t.expectError(error.BlockRepairMismatch, fixture.save(&files, 0));
    try t.expect(!try files.rawExists(shared.name, ".heal"));
    x.rec.bytes(address, 1)[0] ^= 1;
    var fault: Fault = .{ .fail_dir = 1 };
    fault.attach(&files);
    try t.expectError(error.InjectedDirectoryFsync, fixture.save(&files, 0));
    files.io = .{};
    files.publication = .{};
    files.control = .{};
    try t.expect(!try files.rawExists(shared.name, ".heal"));
    try t.expect(try files.rawExists(shared.name, ".blk"));
    const repaired = try fixture.save(&files, 0);
    try t.expectEqual(@as(u64, 1), fixture.stats.repaired_pages);
    try t.expectError(error.BlockBusy, x.local.collect(&files, fixture.slab.slice(u8)));
    _ = try x.commitOne(&files, repaired, fixture.slab.slice(u8));
    try fixture.restore(&old, old_bytes);
    try fixture.restore(&next, next_bytes);
    try t.expectEqualDeep(shared, try x.fs.inspectObject(&files, shared.name, ".blk", fixture.slab.slice(u8)));
    try x.local.collect(&files, fixture.slab.slice(u8));
    try t.expect(!try files.rawExists(shared.name, ".blk"));
    try t.expect(try files.rawExists(shared.name, ".heal"));
    try fixture.restore(&old, old_bytes);
    try corrupt(try files.path(&path, shared.name, ".heal"));
    fixture.agreement.epoch = 4;
    try t.expectError(error.BlockRepairExhausted, fixture.save(&files, 0));
}

fn readFailure(_: ?*anyopaque, _: c_int, _: []u8, _: u64) !usize {
    return error.BlockReadIO;
}
test "hardening: selection propagates operational I/O and cancellation, with explicit bounded-index refusal" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-errors");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    _ = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
    files.read_io.call = readFailure;
    try t.expectError(error.BlockReadIO, x.local.latest(&files, x.selection(&fixture.agreement.binding), 1, 0, 100, fixture.slab.slice(u8)));
    files.read_io = .{};
    var fault: Fault = .{ .cancel = true };
    fault.attach(&files);
    try t.expectError(error.Cancelled, x.local.latest(&files, x.selection(&fixture.agreement.binding), 1, 0, 100, fixture.slab.slice(u8)));
    files.control = .{};
    files.io = .{};
    files.publication = .{};
    const used = files.used;
    files.maintenance_bytes = 512;
    try t.expectError(error.BlockIndexBudget, x.local.collect(&files, fixture.slab.slice(u8)));
    try t.expectEqual(used, files.used);
    files.maintenance_bytes = 8 << 20;
    files.max_work = 1;
    try t.expectError(error.BlockWorkBudget, x.local.latest(&files, x.selection(&fixture.agreement.binding), 1, 0, 100, fixture.slab.slice(u8)));
    try t.expectError(error.BlockWorkBudget, x.local.collect(&files, fixture.slab.slice(u8)));
    try t.expectEqual(used, files.used);
}

fn measurement(mark: usize, payload: usize, m: x.fs.disk.Metrics) !void {
    const path = sys.getenv("TF_GLM_HARDENING_RECEIPT") orelse return;
    const fd = sys.c.open(path, .{ .ACCMODE = .WRONLY, .APPEND = true, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
    if (fd < 0) return error.TestReceipt;
    defer _ = sys.c.close(fd);
    var buffer: [1024]u8 = undefined;
    const line = try std.fmt.bufPrint(&buffer, "{{\"mark\":{d},\"payload\":{d},\"read_bytes\":{d},\"read_calls\":{d},\"file_opens\":{d},\"payload_verified\":{d},\"objects_verified\":{d},\"points_verified\":{d},\"manifests\":{d},\"directory_entries\":{d},\"index_bytes\":{d},\"index_entries\":{d}}}\n", .{ mark, payload, m.read_bytes, m.read_calls, m.file_opens, m.payload_verified, m.objects_verified, m.points_verified, m.manifests, m.directory_entries, m.index_bytes, m.index_entries });
    if (sys.c.write(fd, line.ptr, line.len) != line.len) return error.TestReceipt;
}

test "hardening: growing marks verify payload twice per save+commit; newest selection and GC avoid historic payload scans" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, true, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-scale");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    var metrics: x.fs.disk.Metrics = .{};
    files.metrics = &metrics;
    for ([_]usize{ 512, 1024, 1536, 2000 }, 1..) |mark, epoch| {
        try fixture.mark(mark);
        fixture.agreement.epoch = epoch;
        metrics = .{};
        _ = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
        const payload = fixture.saved.?.plan.len;
        try t.expectEqual(@as(u64, 2 * payload), metrics.payload_verified);
        try t.expectEqual(@as(u64, 2 * (fixture.saved.?.pages.ids.len + 1)), metrics.objects_verified);
        try t.expect(metrics.read_bytes < 2 * payload + (1 << 20));
        try t.expect(metrics.index_bytes <= files.maintenance_bytes);
        try measurement(mark, payload, metrics);
    }
    metrics = .{};
    const latest = (try x.local.latest(&files, x.selection(&fixture.agreement.binding), 1, 0, 100, fixture.slab.slice(u8))).?;
    try t.expectEqual(@as(u64, 4), latest.agreement.epoch);
    try t.expectEqual(@as(u64, 1), metrics.points_verified);
    try t.expectEqual(@as(u64, fixture.saved.?.plan.len), metrics.payload_verified);
    const used = files.used;
    files.maintenance_bytes = 2048;
    try t.expectError(error.BlockIndexBudget, x.local.collect(&files, fixture.slab.slice(u8)));
    try t.expectEqual(used, files.used);
    files.maintenance_bytes = 8 << 20;
    metrics = .{};
    try x.local.collect(&files, fixture.slab.slice(u8));
    try t.expectEqual(@as(u64, 4), metrics.points_verified);
    try t.expectEqual(@as(u64, 0), metrics.payload_verified);
    try t.expectEqual(@as(u64, 0), metrics.objects_verified);
    try measurement(0, 0, metrics);
}

test "hardening: direct-read bounce handles unaligned spans, truncation, and operational read failure" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-direct");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const receipt = try fixture.save(&files, 0);
    var point = try x.local.Point.open(&files, receipt.agreement, 0, receipt.point, fixture.slab.slice(u8));
    defer point.close();
    var raw = try files.opened(point.header.capsule.name, ".cap", .block);
    defer raw.close();
    var aligned: [65537]u8 = undefined;
    var expected: [65537]u8 = undefined;
    try x.fs.disk.readAt(&files, raw.fd, &expected, 37);
    try x.fs.disk.alignedReadAt(&files, raw.fd, &aligned, 37);
    try t.expectEqualSlices(u8, &expected, &aligned);
    const end = raw.trailer.payload + raw.trailer.meta + @sizeOf(x.f.rows.Trailer);
    try t.expectError(error.BlockTruncated, x.fs.disk.alignedReadAt(&files, raw.fd, aligned[0..3], end - 1));
    files.read_io.call = readFailure;
    try t.expectError(error.BlockReadIO, x.fs.disk.alignedReadAt(&files, raw.fd, &aligned, 37));
}
