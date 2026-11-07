//! Locally checked points and manifests. This module never opens another rank's directory.
const std = @import("std");
const sys = @import("sys");
const f = @import("format.zig");
const fs = @import("files.zig");
pub const Reader = @import("reader.zig").Reader;
pub const collect = @import("collect.zig").collect;

pub const Point = struct {
    files: *const fs.Files,
    opened: f.rows.Opened,
    header: f.Header,
    record: f.epoch.Record,

    pub fn byName(s: *const fs.Files, name: f.Name, expected: ?f.epoch.Record, slab: []u8) !Point {
        var o = try s.opened(name, ".point", .state);
        errdefer o.close();
        var bytes: [f.wire.size(f.Header)]u8 = undefined;
        const meta = try fs.metadata(s, &o, &bytes);
        const h = try f.wire.decode(f.Header, meta);
        try h.validate();
        if (o.trailer.payload != @as(u64, h.pages) * f.wire.size(f.Object) or o.trailer.at != h.envelope.agreement.mark or !std.mem.eql(u8, &name, &h.envelope.agreement.identity(h.envelope.rank).name()) or !std.mem.eql(u8, &name, &h.capsule.name)) return error.BlockIdentity;
        const record = try fs.verify(s, &o, meta, slab);
        if (expected) |want| if (!std.meta.eql(record, want)) return error.BlockHash;
        if (s.metrics) |m| m.points_verified += 1;
        return .{ .files = s, .opened = o, .header = h, .record = record };
    }
    pub fn open(s: *const fs.Files, a: f.epoch.Agreement, rank: u32, expected: ?f.epoch.Record, slab: []u8) !Point {
        try a.validate();
        if (rank >= a.world) return error.CheckpointRank;
        var p = try byName(s, a.identity(rank).name(), expected, slab);
        errdefer p.close();
        if (!std.meta.eql(p.header.envelope.agreement, a) or p.header.envelope.rank != rank) return error.BlockIdentity;
        return p;
    }
    pub fn close(p: *Point) void {
        p.opened.close();
    }
    pub fn page(p: *const Point, index: usize) !f.Object {
        if (index >= p.header.pages) return error.BlockFormat;
        var bytes: [f.wire.size(f.Object)]u8 = undefined;
        try fs.disk.readAt(p.files, p.opened.fd, &bytes, index * bytes.len);
        return f.wire.decode(f.Object, &bytes);
    }
    pub fn validateObjects(p: *const Point, s: *const fs.Files, slab: []u8) !void {
        try p.validateObject(s, p.header.capsule, ".cap", false, slab);
        for (0..p.header.pages) |i| try p.validateObject(s, try p.page(i), ".blk", true, slab);
    }
    fn validateObject(p: *const Point, s: *const fs.Files, want: f.Object, ext: []const u8, page_object: bool, slab: []u8) !void {
        const actual = try fs.inspectObject(s, want.name, ext, slab);
        if (!std.meta.eql(actual, want)) return error.BlockHash;
        var o = try fs.ObjectFile.open(s, want, ext);
        defer o.close();
        try checkChunks(&p.header, &o.chunks, page_object);
    }
    /// Re-sync existing shared blocks as well as new ones before publishing the local manifest.
    pub fn sync(p: *const Point, s: *const fs.Files) !void {
        try s.io.sync(s.io.ctx, p.opened.fd);
        for (0..p.header.pages + 1) |i| {
            const obj = if (i == p.header.pages) p.header.capsule else try p.page(i);
            var opened = try s.opened(obj.name, if (i == p.header.pages) ".cap" else ".blk", .block);
            defer opened.close();
            try s.io.sync(s.io.ctx, opened.fd);
        }
    }
};

pub fn checkChunks(h: *const f.Header, c: *const f.Chunks, page: bool) !void {
    if (h.count != c.count) return error.BlockFormat;
    for (h.parts[0..h.count], 0..) |p, i| {
        if (c.lengths[i] != if (page) p.span else p.len - p.immutable) return error.BlockFormat;
    }
}

pub fn validateManifest(m: *const f.epoch.Manifest) !void {
    if (m.magic != 0x31454b4346474c4d or m.version != 1 or m.agreement.epoch == std.math.maxInt(u64)) return error.BlockFormat;
    try m.agreement.validate();
    for (m.ranks, 0..) |record, i| {
        if (i < m.agreement.world) {
            if (record.bytes < f.wire.size(f.Header) + @sizeOf(f.rows.Trailer) or std.mem.eql(u8, &record.digest, &f.zero)) return error.BlockFormat;
        } else if (!std.meta.eql(record, f.epoch.Record{})) return error.BlockFormat;
    }
}

pub fn readManifest(s: *const fs.Files, name: f.Name) !f.epoch.Manifest {
    var o = try s.opened(name, ".commit", .state);
    defer o.close();
    if (o.trailer.payload != f.wire.size(f.epoch.Manifest) or o.trailer.meta != 0 or o.trailer.meta_hash != std.hash.Wyhash.hash(1, "")) return error.BlockFormat;
    var bytes: [f.wire.size(f.epoch.Manifest)]u8 = undefined;
    try fs.disk.readAt(s, o.fd, &bytes, 0);
    if (std.hash.Wyhash.hash(0, &bytes) != o.trailer.payload_hash) return error.RowsHash;
    if (s.metrics) |metrics| metrics.manifests += 1;
    const m = try f.wire.decode(f.epoch.Manifest, &bytes);
    try validateManifest(&m);
    if (!std.mem.eql(u8, &m.agreement.manifestName(), &name) or o.trailer.at != m.agreement.mark) return error.BlockIdentity;
    return m;
}

pub fn validateLocal(s: *const fs.Files, m: *const f.epoch.Manifest, rank: u32, slab: []u8) !void {
    try validateManifest(m);
    if (rank >= m.agreement.world) return error.CheckpointRank;
    var p = try Point.open(s, m.agreement, rank, m.ranks[rank], slab);
    defer p.close();
    try p.validateObjects(s, slab);
}

pub fn publish(s: *fs.Files, m: *const f.epoch.Manifest, rank: u32, slab: []u8) !bool {
    errdefer s.recount() catch {
        s.used = s.quota;
    };
    try validateManifest(m);
    if (rank >= m.agreement.world) return error.CheckpointRank;
    var p = try Point.open(s, m.agreement, rank, m.ranks[rank], slab);
    defer p.close();
    try p.validateObjects(s, slab);
    try p.sync(s);
    try s.publication.sync_dir(s.publication.ctx, s.fd);
    const name = m.agreement.manifestName();
    if (try s.exists(name, ".commit")) {
        if (!std.meta.eql(try readManifest(s, name), m.*)) return error.BlockAlreadyExists;
        var o = try s.opened(name, ".commit", .state);
        defer o.close();
        try s.io.sync(s.io.ctx, o.fd);
        try s.publication.sync_dir(s.publication.ctx, s.fd);
        return false;
    }
    var reservation = try s.reserve(f.wire.size(f.epoch.Manifest) + @sizeOf(f.rows.Trailer));
    defer reservation.close();
    var w = try s.writer(name, ".commit");
    defer w.abort();
    var bytes: [f.wire.size(f.epoch.Manifest)]u8 = undefined;
    try w.append(try f.wire.encode(m.*, &bytes));
    _ = try s.publish(&reservation, &w, name, .state, @intCast(m.agreement.mark), "");
    return true;
}

/// Caller selection must establish request-prefix compatibility; it may be stricter than exact Binding equality.
pub const Selection = struct {
    ctx: *const anyopaque,
    accepts: *const fn (*const anyopaque, f.epoch.Agreement) bool,
};

pub const Key = struct { epoch: u64, name: f.Name };
pub const Bound = struct { key: Key, inclusive: bool = false };
pub fn key(m: f.epoch.Manifest) Key {
    return .{ .epoch = m.agreement.epoch, .name = m.agreement.manifestName() };
}
pub fn less(a: Key, b: Key) bool {
    return a.epoch < b.epoch or (a.epoch == b.epoch and std.mem.order(u8, &a.name, &b.name) == .lt);
}
pub fn within(m: f.epoch.Manifest, bound: Bound) bool {
    return less(key(m), bound.key) or (bound.inclusive and std.meta.eql(key(m), bound.key));
}
pub fn latest(s: *const fs.Files, selection: Selection, world: u32, rank: u32, ceiling: u64, slab: []u8) !?f.epoch.Manifest {
    return latestBefore(s, selection, world, rank, .{ .key = .{ .epoch = ceiling, .name = f.zero } }, slab);
}

/// Metadata-only selection, then one payload validation of the newest candidate; ties use descending names.
pub fn latestBefore(s: *const fs.Files, selection: Selection, world: u32, rank: u32, bound: Bound, slab: []u8) !?f.epoch.Manifest {
    var before: ?f.epoch.Manifest = null;
    var work: usize = 0;
    while (work < s.max_work) {
        const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
        defer _ = sys.c.closedir(dir);
        var best: ?f.epoch.Manifest = null;
        while (try fs.disk.next(dir)) |text| {
            try s.control.check();
            work += 1;
            if (work >= s.max_work) return error.BlockWorkBudget;
            if (s.metrics) |m| m.directory_entries += 1;
            if (text.len != 71 or !std.mem.endsWith(u8, text, ".commit")) continue;
            const name = decodeName(text[0..64]) catch continue;
            const m = readManifest(s, name) catch |err| {
                if (!fs.disk.invalid(err)) return err;
                continue;
            };
            if (m.agreement.world != world or rank >= world or !within(m, bound) or !selection.accepts(selection.ctx, m.agreement)) continue;
            if (before) |previous| if (!newer(previous, m)) continue;
            if (best) |old| if (!newer(m, old)) continue;
            best = m;
        }
        const candidate = best orelse return null;
        validateLocal(s, &candidate, rank, slab) catch |err| {
            if (!fs.disk.invalid(err)) return err;
            before = candidate;
            continue;
        };
        try s.control.check();
        return candidate;
    }
    return error.BlockWorkBudget;
}

fn newer(a: f.epoch.Manifest, b: f.epoch.Manifest) bool {
    return less(key(b), key(a));
}

pub fn decodeName(text: []const u8) !f.Name {
    var name: f.Name = undefined;
    if (text.len != 64) return error.BlockFormat;
    _ = std.fmt.hexToBytes(&name, text) catch return error.BlockFormat;
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(name, .lower), text)) return error.BlockFormat;
    return name;
}
