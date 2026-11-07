//! A fixed-capacity index of committed references, never payloads. Budget exhaustion refuses mutation.
const std = @import("std");
const sys = @import("sys");
const f = @import("format.zig");
const fs = @import("files.zig");
const local = @import("local.zig");
pub const Kind = enum { empty, block, capsule, point, commit };
const Entry = struct { kind: Kind = .empty, object: f.Object = .{} };
pub const Index = struct {
    allocator: std.mem.Allocator,
    table: []Entry,
    count: usize = 0,
    work: usize = 0,
    limit: usize,

    pub fn init(s: *const fs.Files) !Index {
        if (s.maintenance_bytes < 8 * @sizeOf(Entry) or s.maintenance_bytes > 64 << 20) return error.BlockIndexBudget;
        var capacity: usize = 8;
        while (capacity <= s.maintenance_bytes / @sizeOf(Entry) / 2) capacity *= 2;
        const table = try s.maintenance_allocator.alloc(Entry, capacity);
        @memset(table, .{});
        if (s.metrics) |m| m.index_bytes = @max(m.index_bytes, table.len * @sizeOf(Entry));
        return .{ .allocator = s.maintenance_allocator, .table = table, .limit = s.max_work };
    }
    pub fn close(idx: *Index) void {
        idx.allocator.free(idx.table);
    }
    pub fn tick(idx: *Index) !void {
        if (idx.work >= idx.limit) return error.BlockWorkBudget;
        idx.work += 1;
    }
    fn slot(idx: *const Index, name: f.Name, kind: Kind) usize {
        var at: usize = @intCast(std.hash.Wyhash.hash(@backingInt(kind), &name) & (idx.table.len - 1));
        while (idx.table[at].kind != .empty) : (at = (at + 1) & (idx.table.len - 1)) {
            if (idx.table[at].kind == kind and std.mem.eql(u8, &idx.table[at].object.name, &name)) break;
        }
        return at;
    }
    pub fn get(idx: *const Index, name: f.Name, kind: Kind) ?f.Object {
        const e = idx.table[idx.slot(name, kind)];
        return if (e.kind == .empty) null else e.object;
    }
    pub fn add(idx: *Index, object: f.Object, kind: Kind) !void {
        try idx.tick();
        const e = &idx.table[idx.slot(object.name, kind)];
        if (e.kind != .empty) {
            if (!std.meta.eql(e.object, object)) return error.BlockReferenceConflict;
            return;
        }
        if (idx.count >= idx.table.len / 2) return error.BlockIndexBudget;
        e.* = .{ .kind = kind, .object = object };
        idx.count += 1;
    }
    /// Every retained point is SHA-checked once against its commit, regardless of the number of stored objects.
    pub fn load(idx: *Index, s: *const fs.Files, slab: []u8) !void {
        const dir = sys.c.opendir(s.dir) orelse return error.CheckpointDirectory;
        defer _ = sys.c.closedir(dir);
        while (try fs.disk.next(dir)) |text| {
            try s.control.check();
            try idx.tick();
            if (s.metrics) |m| m.directory_entries += 1;
            if (text.len != 71 or !std.mem.endsWith(u8, text, ".commit")) continue;
            const name = try local.decodeName(text[0..64]);
            const m = try local.readManifest(s, name);
            var found = false;
            for (0..m.agreement.world) |rank| {
                const point_name = m.agreement.identity(@intCast(rank)).name();
                if (!try s.rawExists(point_name, ".point")) continue;
                var p = try local.Point.open(s, m.agreement, @intCast(rank), m.ranks[rank], slab);
                defer p.close();
                found = true;
                try idx.add(.{ .name = point_name, .record = p.record }, .point);
                try idx.add(p.header.capsule, .capsule);
                for (0..p.header.pages) |i| try idx.add(try p.page(i), .block);
            }
            if (!found) return error.BlockRetainedPointMissing;
            try idx.add(.{ .name = name }, .commit);
        }
        if (s.metrics) |m| m.index_entries = @max(m.index_entries, idx.count);
    }
};
