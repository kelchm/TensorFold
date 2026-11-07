//! Reconstructs Plan order directly from local page segments and the point capsule into Inactive.load's slab.
const std = @import("std");
const glm = @import("glm");
const f = @import("format.zig");
const fs = @import("files.zig");
const local = @import("local.zig");

pub const Reader = struct {
    files: *fs.Files,
    point: local.Point,
    envelope: f.epoch.Envelope,
    detail_bytes: [f.wire.size(glm.checkpoint_restore.Metadata)]u8,
    object: ?fs.ObjectFile = null,
    part: usize = 0,
    done: u64 = 0,
    offset: u64 = 0,
    segment_offset: u64 = 0,
    segment_left: u64 = 0,
    sha: f.Sha = .init(.{}),
    finished: bool = false,

    /// Use only after the transport has established a complete common epoch; local validation alone is insufficient.
    pub fn open(s: *fs.Files, manifest: *const f.epoch.Manifest, rank: u32, slab: []u8) !Reader {
        try local.validateManifest(manifest);
        if (rank >= manifest.agreement.world) return error.CheckpointRank;
        var p = try local.Point.open(s, manifest.agreement, rank, manifest.ranks[rank], slab);
        errdefer p.close();
        var r: Reader = .{ .files = s, .point = p, .envelope = p.header.envelope, .detail_bytes = undefined };
        _ = try f.wire.encode(p.header.metadata, &r.detail_bytes);
        s.readers += 1;
        return r;
    }
    pub fn close(r: *Reader) void {
        if (r.object) |*o| o.close();
        r.object = null;
        r.point.close();
        r.files.readers -= 1;
    }
    pub fn check(r: *const Reader) !void {
        try r.files.control.check();
    }
    pub fn durable(_: *const Reader) bool {
        return true;
    }
    pub fn detail(r: *const Reader) []const u8 {
        return &r.detail_bytes;
    }
    pub fn read(r: *Reader, into: []u8) !usize {
        try r.check();
        if (r.finished or into.len == 0 or into.len > f.epoch.max_chunk) return error.CheckpointReadState;
        const h = &r.point.header;
        while (r.part < h.count and r.done == h.parts[r.part].len) {
            r.part += 1;
            r.done = 0;
        }
        if (r.part == h.count) return 0;
        const p = h.parts[r.part];
        if (r.object == null) {
            const page = r.done < p.immutable;
            const reference = if (page) try r.point.page(@intCast(r.done / p.span)) else h.capsule;
            var object = try fs.ObjectFile.open(r.files, reference, if (page) ".blk" else ".cap");
            errdefer object.close();
            try local.checkChunks(h, &object.chunks, page);
            r.segment_offset = if (page) p.block else p.capsule;
            r.segment_left = if (page) p.span else p.len - p.immutable;
            r.sha = .init(.{});
            r.object = object;
        }
        const n: usize = @intCast(@min(into.len, r.segment_left));
        if (!f.rows.readUnaligned(r.object.?.opened.fd, into[0..n], r.segment_offset)) return error.RowsRead;
        r.sha.update(into[0..n]);
        r.segment_offset += n;
        r.segment_left -= n;
        r.done += n;
        r.offset += n;
        if (r.segment_left == 0) {
            if (!std.mem.eql(u8, &r.sha.finalResult(), &r.object.?.chunks.hashes[r.part])) return error.BlockHash;
            r.object.?.close();
            r.object = null;
        }
        return n;
    }
    pub fn finish(r: *Reader) !f.epoch.Record {
        try r.check();
        if (r.finished or r.offset != r.envelope.payload or r.object != null) return error.CheckpointReadState;
        // References are read on demand. Recheck their original digest before Ready, closing a file-mutation race.
        var scratch: [f.wire.size(f.Object)]u8 = undefined;
        var meta: [f.wire.size(f.Header)]u8 = undefined;
        const record = try fs.verify(&r.point.opened, try f.wire.encode(r.point.header, &meta), &scratch, r.files.control);
        if (!std.meta.eql(record, r.point.record)) return error.BlockHash;
        r.finished = true;
        return r.point.record;
    }
};
