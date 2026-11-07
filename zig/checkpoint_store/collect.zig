//! Offline collection under a serialized directory owner, with an admitted bounded reference index.
const std = @import("std");
const sys = @import("sys");
const fs = @import("files.zig");
const local = @import("local.zig");
const retained = @import("retained.zig");

fn remove(s: *fs.Files, name: []const u8) !void {
    try s.control.check();
    var buf: [1200]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ s.dir, name }, 0);
    if (sys.c.unlink(path) != 0) return error.BlockCleanup;
}

/// Index validation finishes before deletion. Pending receipts/readers exclude collection, including failed votes.
pub fn collect(s: *fs.Files, slab: []u8) !void {
    if (s.readers != 0) return error.BlockBusy;
    try s.begin(null);
    defer s.transaction = false;
    var changed = false;
    errdefer if (changed) {
        s.recount() catch {
            s.used = s.quota;
        };
    };
    var idx = try retained.Index.init(s);
    defer idx.close();
    try idx.load(s, slab);
    const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
    defer _ = sys.c.closedir(dir);
    while (try fs.disk.next(dir)) |text| {
        try idx.tick();
        if (s.metrics) |m| m.directory_entries += 1;
        if (std.mem.endsWith(u8, text, ".part")) {
            changed = true;
            try remove(s, text);
            continue;
        }
        const kind: retained.Kind = if (text.len == 70 and std.mem.endsWith(u8, text, ".point")) .point else if ((text.len == 68 and std.mem.endsWith(u8, text, ".blk")) or (text.len == 69 and std.mem.endsWith(u8, text, ".heal"))) .block else if (text.len == 68 and std.mem.endsWith(u8, text, ".cap")) .capsule else continue;
        const name = try local.decodeName(text[0..64]);
        const wanted = idx.get(name, kind);
        if (wanted == null) {
            changed = true;
            try remove(s, text);
        } else if (std.mem.endsWith(u8, text, ".blk") and try s.rawExists(name, ".heal")) {
            // A durable corrective copy can retire the corrupt inode; it must still match every committed reference.
            const actual = try fs.inspectObject(s, name, ".heal", slab);
            if (!std.meta.eql(actual, wanted.?)) return error.BlockHash;
            changed = true;
            try remove(s, text);
        }
    }
    try s.publication.sync_dir(s.publication.ctx, s.fd);
    try s.recount();
}
