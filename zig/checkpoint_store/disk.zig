//! Store-specific reads distinguish absent/invalid data from operational errors; all work is counted.
const std = @import("std");
const sys = @import("sys");
const f = @import("format.zig");
pub const Metrics = struct {
    read_bytes: u64 = 0,
    file_opens: u64 = 0,
    read_calls: u64 = 0,
    payload_verified: u64 = 0,
    objects_verified: u64 = 0,
    points_verified: u64 = 0,
    manifests: u64 = 0,
    directory_entries: u64 = 0,
    index_bytes: usize = 0,
    index_entries: usize = 0,
};
pub const Read = struct {
    ctx: ?*anyopaque = null,
    call: *const fn (?*anyopaque, c_int, []u8, u64) anyerror!usize = read,
    fn read(_: ?*anyopaque, fd: c_int, bytes: []u8, offset: u64) !usize {
        while (true) {
            const n = sys.c.pread(fd, bytes.ptr, bytes.len, @intCast(offset));
            if (n < 0 and sys.c.errno(n) == .INTR) continue;
            if (n < 0) return error.BlockReadIO;
            return @intCast(n);
        }
    }
};
pub fn readAt(s: anytype, fd: c_int, bytes: []u8, offset: u64) !void {
    if (@import("builtin").os.tag == .linux) return alignedReadAt(s, fd, bytes, offset);
    var done: usize = 0;
    while (done < bytes.len) {
        try s.control.check();
        const n = try s.read_io.call(s.read_io.ctx, fd, bytes[done..], offset + done);
        if (s.metrics) |m| {
            m.read_calls += 1;
            m.read_bytes += n;
        }
        if (n == 0) return error.BlockTruncated;
        if (n > bytes.len - done) return error.BlockReadIO;
        done += n;
    }
}
/// Linux O_DIRECT needs aligned I/O even for the metadata and arbitrary caller slabs; memory stays at 64 KiB.
pub fn alignedReadAt(s: anytype, fd: c_int, bytes: []u8, offset: u64) !void {
    var bounce: [64 << 10]u8 align(4096) = undefined;
    var done: usize = 0;
    while (done < bytes.len) {
        try s.control.check();
        const pos = offset + done;
        const lo = std.mem.alignBackward(u64, pos, 4096);
        const skip: usize = @intCast(pos - lo);
        const want = @min(bytes.len - done, bounce.len - skip);
        const span = std.mem.alignForward(usize, skip + want, 4096);
        const got = try s.read_io.call(s.read_io.ctx, fd, bounce[0..span], lo);
        if (s.metrics) |m| {
            m.read_calls += 1;
            m.read_bytes += got;
        }
        if (got > span) return error.BlockReadIO;
        if (got <= skip) return error.BlockTruncated;
        const available = @min(want, got - skip);
        @memcpy(bytes[done..][0..available], bounce[skip..][0..available]);
        done += available;
    }
}
pub fn open(s: anytype, path: [:0]const u8, kind: f.rows.Kind, name: f.Name) !f.rows.Opened {
    try s.control.check();
    if (s.metrics) |m| m.file_opens += 1;
    var fd: c_int = -1;
    if (@import("builtin").os.tag == .linux) {
        fd = sys.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true, .DIRECT = true }, @as(sys.c.mode_t, 0));
        if (fd < 0 and sys.c.errno(fd) != .INVAL and sys.c.errno(fd) != .OPNOTSUPP)
            return if (sys.c.errno(fd) == .NOENT) error.RowsMissing else error.BlockReadIO;
    }
    if (fd < 0) fd = sys.c.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
    if (fd < 0) return if (sys.c.errno(fd) == .NOENT) error.RowsMissing else error.BlockReadIO;
    errdefer _ = sys.c.close(fd);
    if (@import("builtin").os.tag == .macos) _ = sys.c.fcntl(fd, sys.c.F.NOCACHE, @as(c_int, 1));
    const end = sys.c.lseek(fd, 0, sys.c.SEEK.END);
    if (end < 0) return error.BlockReadIO;
    if (end < @sizeOf(f.rows.Trailer)) return error.BlockTruncated;
    var tr: f.rows.Trailer = undefined;
    try readAt(s, fd, std.mem.asBytes(&tr), @intCast(end - @sizeOf(f.rows.Trailer)));
    if (tr.magic != 0x42524654 or tr.version != 1 or tr.kind != @backingInt(kind) or tr.flags != 0 or tr.block != 0 or tr.reserved != 0 or tr.meta > f.epoch.max_metadata) return error.BlockFormat;
    const body = std.math.add(u64, tr.payload, tr.meta) catch return error.BlockFormat;
    const whole = std.math.add(u64, body, @sizeOf(f.rows.Trailer)) catch return error.BlockFormat;
    if (whole != end or !std.mem.eql(u8, &tr.name, &name)) return error.BlockFormat;
    return .{ .fd = fd, .trailer = tr };
}
/// Only explicitly recognized invalid data may be skipped. I/O, quota, cancellation and budget failures propagate.
pub fn invalid(err: anyerror) bool {
    return switch (err) {
        error.RowsMissing, error.BlockTruncated, error.BlockFormat, error.BlockIdentity, error.BlockHash, error.RowsHash, error.CheckpointAgreement, error.CheckpointRankIdentity, error.CheckpointRank, error.CheckpointMetadata, error.CheckpointMedia, error.CheckpointDraft, error.UnsupportedHeadLag, error.CheckpointMetadataSize, error.Overflow => true,
        else => false,
    };
}
pub fn next(dir: *sys.c.DIR) !?[]const u8 {
    sys.c._errno().* = 0;
    const ent = sys.c.readdir(dir) orelse {
        if (sys.c._errno().* != 0) return error.BlockDirectoryIO;
        return null;
    };
    return std.mem.sliceTo(&ent.name, 0);
}
