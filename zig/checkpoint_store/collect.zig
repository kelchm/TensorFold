//! Offline orphan collection under the directory lock. Committed points are never evicted to satisfy a new save.
const std = @import("std");
const sys = @import("sys");
const f = @import("format.zig");
const fs = @import("files.zig");
const local = @import("local.zig");

fn remove(s: *fs.Files, name: []const u8) !void {
    try s.control.check();
    var buf: [1200]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ s.dir, name }, 0);
    if (sys.c.unlink(path) != 0) return error.BlockCleanup;
}

fn referenced(s: *fs.Files, wanted: ?f.Name, slab: []u8) !bool {
    const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
    defer _ = sys.c.closedir(dir);
    while (sys.c.readdir(dir)) |ent| {
        try s.control.check();
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name.len != 70 or !std.mem.endsWith(u8, name, ".point")) continue;
        var p = try local.Point.byName(s, try local.decodeName(name[0..64]), null, slab);
        defer p.close();
        const m = try local.readManifest(s, p.header.envelope.agreement.manifestName());
        if (!std.meta.eql(m.agreement, p.header.envelope.agreement) or !std.meta.eql(m.ranks[p.header.envelope.rank], p.record)) return error.BlockCleanup;
        if (wanted) |key| {
            if (std.mem.eql(u8, &p.header.capsule.name, &key)) return true;
            for (0..p.header.pages) |i| if (std.mem.eql(u8, &(try p.page(i)).name, &key)) return true;
        }
    }
    return false;
}

/// Deliberately bounded memory and conservative: corrupt retained references abort GC before any object is removed.
pub fn collect(s: *fs.Files, slab: []u8) !void {
    if (s.reserved != 0 or s.readers != 0) return error.BlockBusy;
    const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
    defer _ = sys.c.closedir(dir);
    while (sys.c.readdir(dir)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.endsWith(u8, name, ".part")) {
            try remove(s, name);
        } else if (name.len == 70 and std.mem.endsWith(u8, name, ".point")) {
            var p = try local.Point.byName(s, try local.decodeName(name[0..64]), null, slab);
            defer p.close();
            const commit_name = p.header.envelope.agreement.manifestName();
            if (!try s.exists(commit_name, ".commit")) {
                try remove(s, name);
            } else {
                const m = try local.readManifest(s, commit_name);
                // A losing same-epoch proposal has a different point name and cannot be the committed state.
                if (!std.meta.eql(m.agreement, p.header.envelope.agreement)) try remove(s, name);
            }
        }
    }
    try s.publication.sync_dir(s.publication.ctx, s.fd);
    _ = try referenced(s, null, slab);
    const objects = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
    defer _ = sys.c.closedir(objects);
    while (sys.c.readdir(objects)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (name.len != 68 or !(std.mem.endsWith(u8, name, ".blk") or std.mem.endsWith(u8, name, ".cap"))) continue;
        if (!try referenced(s, try local.decodeName(name[0..64]), slab)) try remove(s, name);
    }
    try s.publication.sync_dir(s.publication.ctx, s.fd);
    try s.recount();
}
