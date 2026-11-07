const std = @import("std");
const sys = @import("sys");
const x = @import("fixture.zig");
const Fault = @import("store_test.zig").Fault;
const t = std.testing;

pub fn corruptWithRowsHash(path: [:0]const u8) !void {
    const fd = sys.c.open(path, .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = sys.c.close(fd);
    const end = sys.c.lseek(fd, 0, sys.c.SEEK.END);
    var tr: x.f.rows.Trailer = undefined;
    if (sys.c.pread(fd, std.mem.asBytes(&tr), @sizeOf(x.f.rows.Trailer), end - @sizeOf(x.f.rows.Trailer)) != @sizeOf(x.f.rows.Trailer)) return error.TestFile;
    var byte: [1]u8 = undefined;
    if (sys.c.pread(fd, &byte, 1, 0) != 1) return error.TestFile;
    byte[0] ^= 17;
    if (sys.c.pwrite(fd, &byte, 1, 0) != 1) return error.TestFile;
    var hash = std.hash.Wyhash.init(0);
    var slab: [4096]u8 = undefined;
    var at: u64 = 0;
    while (at < tr.payload) {
        const n: usize = @intCast(@min(slab.len, tr.payload - at));
        if (!x.f.rows.readUnaligned(fd, slab[0..n], at)) return error.TestFile;
        hash.update(slab[0..n]);
        at += n;
    }
    tr.payload_hash = hash.final();
    if (sys.c.pwrite(fd, std.mem.asBytes(&tr), @sizeOf(x.f.rows.Trailer), end - @sizeOf(x.f.rows.Trailer)) != @sizeOf(x.f.rows.Trailer)) return error.TestFile;
}

test "block store: recomputed row checksums cannot hide changed block content or rank references" {
    for ([_]bool{ false, true }) |point_file| {
        var fixture: x.Fixture = undefined;
        try fixture.init(false, false, false, 1);
        defer fixture.deinit();
        var dir = try x.Directory.init("sha");
        defer dir.deinit();
        var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
        defer files.close();
        const first = try fixture.save(&files, 0);
        _ = try x.commitOne(&files, first, fixture.slab.slice(u8));
        try fixture.mark(1025);
        fixture.agreement.epoch = 2;
        const next = try fixture.save(&files, 0);
        _ = try x.commitOne(&files, next, fixture.slab.slice(u8));
        var p = try x.local.Point.open(&files, next.agreement, 0, next.point, fixture.slab.slice(u8));
        defer p.close();
        var path: [1200]u8 = undefined;
        try corruptWithRowsHash(try files.path(&path, if (point_file) next.agreement.identity(0).name() else (try p.page(1)).name, if (point_file) ".point" else ".blk"));
        var transport = x.single();
        const recovered = (try x.protocol.recover(&files, x.selection(&fixture.agreement.binding), &transport, fixture.slab.slice(u8), 8)).?;
        try t.expectEqual(@as(u64, 1), recovered.manifest.agreement.epoch);
        if (!point_file) {
            fixture.agreement.epoch = 3;
            const repaired = try fixture.save(&files, 0);
            try t.expectEqual(@as(u64, 1), fixture.stats.repaired_pages);
            _ = try x.commitOne(&files, repaired, fixture.slab.slice(u8));
        }
    }
}

test "block store: reference-table mutation after opening is rechecked before finish" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("ref-race");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const receipt = try fixture.save(&files, 0);
    const epoch = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    var reader = try epoch.reader(fixture.slab.slice(u8));
    defer reader.close();
    while (try reader.read(fixture.slab.slice(u8)) != 0) {}
    var path: [1200]u8 = undefined;
    try corruptWithRowsHash(try files.path(&path, receipt.agreement.identity(0).name(), ".point"));
    try t.expectError(error.RowsHash, reader.finish());
}

test "block store: no-replace epoch slot preserves the committed mark and collects the losing proposal" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("epoch-slot");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const receipt = try fixture.save(&files, 0);
    const epoch = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    const used = files.used;
    // Canonical prefix binding prevents a forged same-slot/different-mark capture before any publication.
    try fixture.mark(1025);
    fixture.agreement.binding.prefix = receipt.agreement.binding.prefix;
    try t.expectError(error.BlockIdentity, fixture.save(&files, 0));
    var conflicting = epoch.manifest;
    conflicting.ranks[0].digest[0] ^= 1;
    try t.expectError(error.BlockHash, x.local.publish(&files, &conflicting, 0, fixture.slab.slice(u8)));
    try x.local.collect(&files, fixture.slab.slice(u8));
    try t.expectEqual(used, files.used);
    try x.local.validateLocal(&files, &epoch.manifest, 0, fixture.slab.slice(u8));
}

test "block store: cancellation during actual inactive restore poisons and releases its destination" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("load-cancel");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    const expected = try fixture.oracle();
    defer t.allocator.free(expected);
    const receipt = try fixture.save(&files, 0);
    const epoch = try x.commitOne(&files, receipt, fixture.slab.slice(u8));
    var fault: Fault = .{ .cancel_at = 20 };
    fault.attach(&files);
    try t.expectError(error.Cancelled, fixture.restore(&epoch, expected));
    files.control = .{};
    files.io = .{};
    files.publication = .{};
    try t.expectEqual(@as(usize, 0), files.readers);
    try t.expectEqual(@as(usize, 0), fixture.pool.promised);
    try fixture.restore(&epoch, expected);
}

test "block store: prepared media identities and row boundaries control reuse across forks" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    try fixture.mark(1537);
    var media: x.glm.checkpoint_restore.Media = .{ .processor = @splat(7), .count = 1 };
    media.images[0] = .{ .digest = @splat(8), .at = 700, .tokens = 4, .gh = 4, .gw = 4 };
    fixture.metadata.media = media;
    var before: [3]x.f.Name = undefined;
    var after: [3]x.f.Name = undefined;
    const names = @import("names.zig");
    try names.build(fixture.input(), &fixture.metadata, &before);
    fixture.metadata.media.images[0].digest[0] ^= 1;
    try names.build(fixture.input(), &fixture.metadata, &after);
    try t.expectEqualDeep(before[0], after[0]);
    try t.expect(!std.mem.eql(u8, &before[1], &after[1]));
    try t.expect(!std.mem.eql(u8, &before[2], &after[2]));
    try t.expectError(error.BlockIdentity, names.build(.{ .tokens = fixture.tokens[0..1024], .spans = &.{}, .starts = fixture.input().starts }, &fixture.metadata, &after));
}
