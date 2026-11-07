//! Exclusive rank-local directory ownership, quota reservation, native row files and ordered publication.
const std = @import("std");
const sys = @import("sys");
const core = @import("core");
const f = @import("format.zig");
const rows = f.rows;
extern "c" fn flock(c_int, c_int) c_int;

pub const Files = struct {
    dir: [:0]const u8,
    fd: c_int,
    lock: c_int,
    quota: u64,
    used: u64 = 0,
    reserved: u64 = 0,
    readers: usize = 0,
    nonce: u64,
    io: rows.IO = .{},
    publication: core.checkpoint_file.Publication = .{},
    control: f.epoch.Control = .{},

    /// The owner provisions and fsyncs the directory and its parent before opening this store.
    pub fn open(dir: [:0]const u8, quota: u64, nonce: u64) !Files {
        if (quota == 0 or quota > f.max_quota) return error.BlockQuota;
        const fd = sys.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
        if (fd < 0) return error.CheckpointDirectory;
        errdefer _ = sys.c.close(fd);
        var lock_path: [1200]u8 = undefined;
        const lock = sys.c.open(try std.fmt.bufPrintSentinel(&lock_path, "{s}/.writer", .{dir}, 0), .{ .ACCMODE = .RDWR, .CREAT = true, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0o600));
        if (lock < 0) return error.BlockLock;
        errdefer _ = sys.c.close(lock);
        if (flock(lock, 2 | 4) != 0) return error.BlockBusy;
        var result: Files = .{ .dir = dir, .fd = fd, .lock = lock, .quota = quota, .nonce = nonce };
        try result.recount();
        return result;
    }
    pub fn close(s: *Files) void {
        std.debug.assert(s.reserved == 0 and s.readers == 0);
        _ = sys.c.close(s.lock);
        _ = sys.c.close(s.fd);
    }
    pub fn path(s: *const Files, buf: []u8, name: f.Name, ext: []const u8) ![:0]const u8 {
        const stem = std.fmt.bytesToHex(name, .lower);
        return rows.path(buf, s.dir, &stem, ext, false);
    }
    pub fn exists(s: *const Files, name: f.Name, ext: []const u8) !bool {
        var buf: [1200]u8 = undefined;
        const fd = sys.c.open(try s.path(&buf, name, ext), .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
        if (fd < 0) {
            if (sys.c.errno(fd) == .NOENT) return false;
            return error.RowsRead;
        }
        _ = sys.c.close(fd);
        return true;
    }
    pub fn recount(s: *Files) !void {
        const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
        defer _ = sys.c.closedir(dir);
        var total: u64 = 0;
        while (sys.c.readdir(dir)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..") or std.mem.eql(u8, name, ".writer")) continue;
            var path_buf: [1200]u8 = undefined;
            const p = try std.fmt.bufPrintSentinel(&path_buf, "{s}/{s}", .{ s.dir, name }, 0);
            const fd = sys.c.open(p, .{ .ACCMODE = .RDONLY, .CLOEXEC = true, .NOFOLLOW = true }, @as(sys.c.mode_t, 0));
            if (fd < 0) return error.RowsRead;
            defer _ = sys.c.close(fd);
            const n = sys.c.lseek(fd, 0, sys.c.SEEK.END);
            if (n < 0) return error.RowsRead;
            total = try std.math.add(u64, total, @intCast(n));
        }
        s.used = total;
    }
    pub fn reserve(s: *Files, bytes: u64) !Reservation {
        try s.control.check();
        if (s.reserved != 0) return error.BlockBusy;
        if (s.quota > f.max_quota or s.used > s.quota or bytes > s.quota - s.used) return error.BlockQuota;
        s.reserved = bytes;
        return .{ .files = s, .left = bytes };
    }
    pub fn writer(s: *Files, name: f.Name, ext: []const u8) !rows.Writer {
        try s.control.check();
        s.nonce = try std.math.add(u64, s.nonce, 1);
        const stem = std.fmt.bytesToHex(name, .lower);
        var w = try rows.Writer.openUnique(s.dir, &stem, ext, s.nonce);
        w.io = s.io;
        return w;
    }
    /// File fsync precedes the no-replace link; the directory barrier precedes any successful receipt.
    pub fn publish(s: *Files, r: *Reservation, w: *rows.Writer, name: f.Name, kind: rows.Kind, mark: u32, meta: []const u8) !u64 {
        const bytes = w.payload + meta.len + @sizeOf(rows.Trailer);
        if (r.files != s or bytes > r.left or meta.len > f.epoch.max_metadata) return error.BlockReservation;
        _ = try w.seal(.{ .kind = @backingInt(kind), .name = name, .payload = 0, .payload_hash = 0, .at = mark }, &.{meta}, true);
        try s.control.check();
        const dest = w.dest[0..w.dest_len :0];
        try s.publication.link(s.publication.ctx, w.tmp[0..w.tmp_len :0], dest);
        errdefer {
            _ = sys.c.unlink(dest);
            _ = sys.c.fsync(s.fd);
        }
        if (sys.c.unlink(w.tmp[0..w.tmp_len :0]) != 0) return error.CheckpointPublish;
        try s.control.check();
        try s.publication.sync_dir(s.publication.ctx, s.fd);
        s.used += bytes;
        r.left -= bytes;
        s.reserved -= bytes;
        return bytes;
    }
    pub fn opened(s: *const Files, name: f.Name, ext: []const u8, kind: rows.Kind) !rows.Opened {
        try s.control.check();
        var buf: [1200]u8 = undefined;
        var o = try rows.open(try s.path(&buf, name, ext), kind, name);
        errdefer o.close();
        if (o.trailer.flags != 0 or o.trailer.block != 0 or o.trailer.reserved != 0 or o.trailer.meta > f.epoch.max_metadata) return error.BlockFormat;
        return o;
    }
};

pub const Reservation = struct {
    files: *Files,
    left: u64,
    pub fn close(r: *Reservation) void {
        r.files.reserved -= r.left;
        r.left = 0;
    }
};

pub fn metadata(o: *const rows.Opened, into: []u8) ![]const u8 {
    if (o.trailer.meta > into.len) return error.BlockFormat;
    const bytes = into[0..@intCast(o.trailer.meta)];
    if (!rows.readUnaligned(o.fd, bytes, o.trailer.payload)) return error.RowsRead;
    if (std.hash.Wyhash.hash(1, bytes) != o.trailer.meta_hash) return error.RowsHash;
    return bytes;
}

/// Read and hash every local payload byte using only the supplied slab; metadata stays format-bounded.
pub fn verify(o: *const rows.Opened, meta: []const u8, slab: []u8, control: f.epoch.Control) !f.epoch.Record {
    if (slab.len == 0 or slab.len > f.epoch.max_chunk) return error.CheckpointReadState;
    var sha = f.Sha.init(.{});
    var hash = std.hash.Wyhash.init(0);
    var at: u64 = 0;
    while (at < o.trailer.payload) {
        try control.check();
        const n: usize = @intCast(@min(slab.len, o.trailer.payload - at));
        if (!rows.readUnaligned(o.fd, slab[0..n], at)) return error.RowsRead;
        hash.update(slab[0..n]);
        sha.update(slab[0..n]);
        at += n;
    }
    if (hash.final() != o.trailer.payload_hash) return error.RowsHash;
    return f.record(&sha, meta, &o.trailer);
}

pub const ObjectFile = struct {
    opened: rows.Opened,
    chunks: f.Chunks,

    pub fn open(s: *const Files, object: f.Object, ext: []const u8) !ObjectFile {
        var o = try s.opened(object.name, ext, .block);
        errdefer o.close();
        if (o.trailer.at != 0 or o.trailer.payload + o.trailer.meta + @sizeOf(rows.Trailer) != object.record.bytes) return error.BlockFormat;
        var buf: [f.wire.size(f.Chunks)]u8 = undefined;
        const meta = try metadata(&o, &buf);
        var digest: f.Name = undefined;
        f.Sha.hash(meta, &digest, .{});
        if (!std.mem.eql(u8, &digest, &object.metadata)) return error.BlockHash;
        const chunks = try f.wire.decode(f.Chunks, meta);
        try chunks.validate(o.trailer.payload);
        return .{ .opened = o, .chunks = chunks };
    }
    pub fn close(o: *ObjectFile) void {
        o.opened.close();
    }
};

pub fn inspectObject(s: *const Files, name: f.Name, ext: []const u8, slab: []u8) !f.Object {
    if (slab.len == 0 or slab.len > f.epoch.max_chunk) return error.CheckpointReadState;
    var o = try s.opened(name, ext, .block);
    defer o.close();
    if (o.trailer.at != 0) return error.BlockFormat;
    var buf: [f.wire.size(f.Chunks)]u8 = undefined;
    const meta = try metadata(&o, &buf);
    const chunks = try f.wire.decode(f.Chunks, meta);
    try chunks.validate(o.trailer.payload);
    var digest: f.Name = undefined;
    f.Sha.hash(meta, &digest, .{});
    var sha = f.Sha.init(.{});
    var hash = std.hash.Wyhash.init(0);
    var at: u64 = 0;
    for (chunks.lengths[0..chunks.count], chunks.hashes[0..chunks.count]) |len, expected| {
        if (len == 0) continue;
        var part = f.Sha.init(.{});
        var done: u64 = 0;
        while (done < len) {
            try s.control.check();
            const n: usize = @intCast(@min(slab.len, len - done));
            if (!f.rows.readUnaligned(o.fd, slab[0..n], at)) return error.RowsRead;
            part.update(slab[0..n]);
            hash.update(slab[0..n]);
            sha.update(slab[0..n]);
            at += n;
            done += n;
        }
        if (!std.mem.eql(u8, &part.finalResult(), &expected)) return error.BlockHash;
    }
    if (hash.final() != o.trailer.payload_hash) return error.RowsHash;
    return .{ .name = name, .record = f.record(&sha, meta, &o.trailer), .metadata = digest };
}
