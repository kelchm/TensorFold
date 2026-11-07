//! Native rows files, canonical references, and the bounded reconstruction map; no device pointers on disk.
const std = @import("std");
const core = @import("core");
const glm = @import("glm");
pub const rows = @import("../src/core/rows_file.zig");
pub const chain = @import("../src/core/rows_chain.zig");
pub const epoch = core.checkpoint_epoch;
pub const wire = epoch.wire;
pub const Sha = std.crypto.hash.sha2.Sha256;
pub const Name = chain.Name;
pub const max_parts = @typeInfo(@FieldType(glm.snapshots.layout.Plan, "parts")).array.len;
pub const max_pages = 4096;
pub const max_quota = @as(u64, 64) << 30;
pub const zero: Name = @splat(0);

pub const Object = struct {
    name: Name = zero,
    record: epoch.Record = .{},
    metadata: Name = zero,
};
pub const Part = struct {
    len: u64 = 0,
    span: u64 = 0,
    immutable: u64 = 0,
    capsule: u64 = 0,
    block: u64 = 0,
};
pub const Chunks = struct {
    magic: u64 = 0x31534b4e48434c47,
    count: u32,
    lengths: [max_parts]u64 = @splat(0),
    hashes: [max_parts]Name = @splat(zero),

    pub fn validate(c: *const Chunks, payload: u64) !void {
        if (c.magic != 0x31534b4e48434c47 or c.count == 0 or c.count > max_parts) return error.BlockFormat;
        var total: u64 = 0;
        for (c.lengths, c.hashes, 0..) |len, hash, i| {
            if ((i >= c.count or len == 0) and (len != 0 or !std.mem.eql(u8, &hash, &zero))) return error.BlockFormat;
            total = try std.math.add(u64, total, len);
        }
        if (total != payload) return error.BlockFormat;
    }
};

pub const Header = struct {
    magic: u64 = 0x32544e494f504c47,
    envelope: epoch.Envelope,
    metadata: glm.checkpoint_restore.Metadata,
    count: u32,
    pages: u32,
    parts: [max_parts]Part = @splat(.{}),
    capsule: Object,

    pub fn validate(h: *const Header) !void {
        if (h.magic != 0x32544e494f504c47 or h.count == 0 or h.count > max_parts or h.pages > max_pages or h.envelope.agreement.epoch == std.math.maxInt(u64)) return error.BlockFormat;
        if (!std.meta.eql(h.envelope, try h.metadata.envelope(h.envelope.agreement, h.envelope.rank))) return error.BlockIdentity;
        var total: u64 = 0;
        var capsule: u64 = 0;
        var block: u64 = 0;
        for (h.parts, 0..) |p, i| {
            if (i >= h.count) {
                if (!std.meta.eql(p, Part{})) return error.BlockFormat;
                continue;
            }
            const imm = try std.math.mul(u64, p.span, h.pages);
            if (p.len == 0 or p.immutable != imm or imm > p.len or p.capsule != capsule or p.block != block) return error.BlockFormat;
            capsule = try std.math.add(u64, capsule, p.len - imm);
            block = try std.math.add(u64, block, p.span);
            total = try std.math.add(u64, total, p.len);
        }
        const completed = @min(h.metadata.mark, if (h.metadata.shape.flags & 2 != 0) h.metadata.mtp_len else h.metadata.mark) / glm.state.page_rows;
        if (h.pages != completed or total != h.envelope.payload or block > h.metadata.shape.page_bytes) return error.BlockFormat;
        if (h.capsule.record.bytes != capsule + wire.size(Chunks) + @sizeOf(rows.Trailer)) return error.BlockFormat;
    }
};

pub const Receipt = struct { agreement: epoch.Agreement, rank: u32, point: epoch.Record };

pub fn record(sha: *Sha, metadata: []const u8, trailer: *const rows.Trailer) epoch.Record {
    sha.update(metadata);
    sha.update(std.mem.asBytes(trailer));
    return .{ .bytes = trailer.payload + trailer.meta + @sizeOf(rows.Trailer), .digest = sha.finalResult() };
}

pub fn pageName(context: Name, row: Name) Name {
    var h = Sha.init(.{});
    h.update("GLM immutable page 0960 v2");
    h.update(&context);
    h.update(&row);
    return h.finalResult();
}

comptime {
    std.debug.assert(wire.size(Header) <= epoch.max_metadata);
    std.debug.assert(wire.size(Chunks) <= epoch.max_metadata);
}
