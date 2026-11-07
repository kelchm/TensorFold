//! Same-epoch ordering, contradictory committed references, and conservative budget refusal.
const std = @import("std");
const sys = @import("sys");
const x = @import("fixture.zig");
const t = std.testing;

fn rawWrite(fd: c_int, bytes: []const u8, offset: u64) !void {
    if (sys.c.pwrite(fd, bytes.ptr, bytes.len, @intCast(offset)) != bytes.len) return error.TestWrite;
}
/// Recompute even the enclosing SHA/rows checksum: only cross-reference consistency can reject this forgery.
fn contradict(files: *x.fs.Files, receipt: x.f.Receipt, slab: []u8) !void {
    var p = try x.local.Point.open(files, receipt.agreement, 0, receipt.point, slab);
    defer p.close();
    var object = try p.page(0);
    object.record.digest[0] ^= 1;
    var path: [1200]u8 = undefined;
    const fd = sys.c.open(try files.path(&path, receipt.agreement.identity(0).name(), ".point"), .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
    if (fd < 0) return error.TestOpen;
    defer _ = sys.c.close(fd);
    const payload = try t.allocator.alloc(u8, @intCast(p.opened.trailer.payload));
    defer t.allocator.free(payload);
    try x.fs.disk.readAt(files, fd, payload, 0);
    _ = try x.f.wire.encode(object, payload);
    var trailer = p.opened.trailer;
    trailer.payload_hash = std.hash.Wyhash.hash(0, payload);
    try rawWrite(fd, payload, 0);
    try rawWrite(fd, std.mem.asBytes(&trailer), trailer.payload + trailer.meta);
    var meta: [x.f.wire.size(x.f.Header)]u8 = undefined;
    var sha = x.f.Sha.init(.{});
    sha.update(payload);
    const new_record = x.f.record(&sha, try x.f.wire.encode(p.header, &meta), &trailer);
    var m = try x.local.readManifest(files, receipt.agreement.manifestName());
    m.ranks[0] = new_record;
    var manifest = try files.opened(m.agreement.manifestName(), ".commit", .state);
    defer manifest.close();
    const out = sys.c.open(try files.path(&path, m.agreement.manifestName(), ".commit"), .{ .ACCMODE = .RDWR }, @as(sys.c.mode_t, 0));
    if (out < 0) return error.TestOpen;
    defer _ = sys.c.close(out);
    var encoded: [x.f.wire.size(x.f.epoch.Manifest)]u8 = undefined;
    const data = try x.f.wire.encode(m, &encoded);
    trailer = manifest.trailer;
    trailer.payload_hash = std.hash.Wyhash.hash(0, data);
    try rawWrite(out, data, 0);
    try rawWrite(out, std.mem.asBytes(&trailer), data.len);
}

test "hardening: contradictory retained references refuse repair and GC before deleting anything" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var dir = try x.Directory.init("hardening-conflict");
    defer dir.deinit();
    var files = try x.fs.Files.open(dir.name(), 1 << 30, 100);
    defer files.close();
    _ = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
    try fixture.mark(1025);
    fixture.agreement.epoch = 2;
    const next = try fixture.save(&files, 0);
    _ = try x.commitOne(&files, next, fixture.slab.slice(u8));
    try contradict(&files, next, fixture.slab.slice(u8));
    const used = files.used;
    fixture.agreement.epoch = 3;
    try t.expectError(error.BlockReferenceConflict, fixture.save(&files, 0));
    try t.expectError(error.BlockReferenceConflict, x.local.collect(&files, fixture.slab.slice(u8)));
    try t.expectEqual(used, files.used);
    try t.expect(!try files.rawExists(fixture.agreement.identity(0).name(), ".point"));
}

test "hardening: equal epochs use the same name order despite reversed creation order" {
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 1);
    defer fixture.deinit();
    var a = try x.Directory.init("hardening-tie0");
    defer a.deinit();
    var b = try x.Directory.init("hardening-tie1");
    defer b.deinit();
    const dirs = [2][:0]const u8{ a.name(), b.name() };
    var chosen: [2]x.f.Name = undefined;
    for (dirs, 0..) |dir, order| {
        var files = try x.fs.Files.open(dir, 1 << 30, 100);
        defer files.close();
        var candidates: [2]x.f.Name = undefined;
        for (0..2) |i| {
            fixture.tokens[0] = if (order == i) 1 else 99;
            try fixture.mark(512);
            const epoch = try x.commitOne(&files, try fixture.save(&files, 0), fixture.slab.slice(u8));
            candidates[i] = epoch.manifest.agreement.manifestName();
        }
        var metrics: x.fs.disk.Metrics = .{};
        files.metrics = &metrics;
        const newest = (try x.local.latest(&files, x.selection(&fixture.agreement.binding), 1, 0, 100, fixture.slab.slice(u8))).?;
        chosen[order] = newest.agreement.manifestName();
        const expected = if (std.mem.order(u8, &candidates[0], &candidates[1]) == .gt) candidates[0] else candidates[1];
        try t.expectEqualDeep(expected, chosen[order]);
        try t.expectEqual(@as(u64, 1), metrics.points_verified);
        // Remove only the chosen capsule: the other same-epoch candidate must still be considered locally.
        try files.remove(newest.agreement.identity(0).name(), ".cap");
        const fallback = (try x.local.latest(&files, x.selection(&fixture.agreement.binding), 1, 0, 100, fixture.slab.slice(u8))).?;
        try t.expect(!std.mem.eql(u8, &fallback.agreement.manifestName(), &chosen[order]));
    }
    try t.expectEqualDeep(chosen[0], chosen[1]);
}

test "hardening: processes intersect alternate same-epoch candidates instead of skipping the whole epoch" {
    const process = @import("protocol_test.zig");
    var fixture: x.Fixture = undefined;
    try fixture.init(false, false, false, 2);
    defer fixture.deinit();
    var a = try x.Directory.init("hardening-common0");
    defer a.deinit();
    var b = try x.Directory.init("hardening-common1");
    defer b.deinit();
    const dirs = [2][:0]const u8{ a.name(), b.name() };
    const first = try process.prepare(&fixture, dirs);
    try process.pair(.{ .dirs = dirs, .receipts = first });
    fixture.tokens[0] ^= 63;
    try fixture.mark(512);
    const next = try process.prepare(&fixture, dirs);
    try process.pair(.{ .dirs = dirs, .receipts = next });
    const k1 = first[1].agreement.manifestName();
    const k2 = next[1].agreement.manifestName();
    const first_wins = std.mem.order(u8, &k1, &k2) == .gt;
    var files = try x.fs.Files.open(b.name(), 1 << 30, 900);
    try files.remove((if (first_wins) first[1] else next[1]).agreement.identity(1).name(), ".cap");
    files.close();
    try process.pair(.{ .dirs = dirs, .receipts = next, .recover = true, .expected = 1, .expected_name = if (first_wins) k2 else k1 });
}
