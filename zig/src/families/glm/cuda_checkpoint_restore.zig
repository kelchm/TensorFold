//! Versioned GLM metadata and bounded restore into a newly admitted, inactive State/lease.
//! This is not wired to the server: only the scheduler can prove inactivity and coordinate TP activation.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const epoch = core.checkpoint_epoch;
const wire = epoch.wire;
const state = @import("cuda_state.zig");
const layout = @import("cuda_snapshot_layout.zig");
const cp = @import("cuda_checkpoint.zig");
const Config = @import("config.zig").Config;
const max_layers = @import("config.zig").max_layers;
const df = @import("dflash.zig");
const sparse = @import("cuda_sparse.zig");
const Sha = std.crypto.hash.sha2.Sha256;

// Production permits 50 pictures plus four clips of 128 frames, grouped two frames per tower input.
// Every accepted group needs its own persisted identity even when its pixel/token geometry is tiny.
pub const max_images = 50 + 4 * ((128 + 1) / 2);
pub const Image = struct {
    digest: [32]u8 = @splat(0), // canonical prepared-image identity; not a URL or a cache id
    at: u32 = 0,
    tokens: u32 = 0,
    gh: u32 = 0,
    gw: u32 = 0,
};
pub const Media = struct {
    processor: [32]u8 = @splat(0),
    count: u32 = 0,
    images: [max_images]Image = @splat(.{}),

    pub fn validate(m: *const Media, mark: u64) !void {
        if (m.count > max_images) return error.CheckpointMedia;
        if (m.count > 0 and std.mem.allEqual(u8, &m.processor, 0)) return error.CheckpointMedia;
        if (m.count == 0 and !std.mem.allEqual(u8, &m.processor, 0)) return error.CheckpointMedia;
        var end: u64 = 0;
        for (m.images[0..m.count]) |im| {
            const next = @as(u64, im.at) + im.tokens;
            if (std.mem.allEqual(u8, &im.digest, 0) or im.tokens == 0 or im.at < end or next > mark or im.gh == 0 or im.gw == 0 or im.gh % 2 != 0 or im.gw % 2 != 0 or @as(u64, im.gh) * im.gw != @as(u64, im.tokens) * @import("vision_config.zig").patches_per_token) return error.CheckpointMedia;
            end = next;
        }
        for (m.images[m.count..]) |im| if (!std.meta.eql(im, Image{})) return error.CheckpointMedia;
    }
};

/// Structural layout, independent of device addresses, physical page order and committed length.
/// FP8 codes *and* FP32 scales use the existing Plan; this format never synthesizes or converts state.
pub const Shape = struct {
    version: u32 = 1,
    flags: u32, // bit 0 FP8, bit 1 target MTP head
    page_bytes: u64,
    arena_bytes: u64, // through prompt_hidden, excluding the destination's own page table
    heads: u64,
    total_heads: u32,
    kda_layers: u64,
    caches: u64,
    ring_rows: u64,
    ring_row: u64,
    conv_c: u64,
    rec_layer: u64,
    proj_layer: u64,
    proj_row: u64,
    kv_lora: u32,
    i_dim: u32,
    hidden: u32,

    /// Global layout identity: uneven TP ranks have different recurrent widths and arena offsets.
    /// Their exact local shapes remain in each rank's hashed metadata and are checked against its real destination.
    pub fn layoutIdentity(shape: Shape) [32]u8 {
        var common = shape;
        common.arena_bytes = 0;
        common.heads = 0;
        common.conv_c = 0;
        common.rec_layer = 0;
        common.proj_layer = 0;
        common.proj_row = 0;
        return wire.hash(common);
    }

    pub fn of(st: *const state.State, c: *const Config) !Shape {
        if (st.table == 0 or st.table < st.arena.buf.ptr or st.table - st.arena.buf.ptr >= st.arena.buf.len or st.caches.len == 0 or st.kda_layers > max_layers or st.cur.len != st.kda_layers) return error.CheckpointShape;
        const page = st.caches[0].ps;
        if (page == 0) return error.CheckpointShape;
        for (st.caches) |x| if (x.pt != st.table or x.ps != page or (x.ls != 0) != st.kv8) return error.CheckpointShape;
        if (st.head and (st.mtp_caches.pt != st.table or st.mtp_caches.ps != page or (st.mtp_caches.ls != 0) != st.kv8)) return error.CheckpointShape;
        return .{ .flags = @intFromBool(st.kv8) | (@as(u32, @intFromBool(st.head)) << 1), .page_bytes = page, .arena_bytes = st.table - st.arena.buf.ptr, .heads = st.heads, .total_heads = c.kda_heads, .kda_layers = st.kda_layers, .caches = st.caches.len, .ring_rows = st.ring_rows, .ring_row = st.ring_row, .conv_c = st.conv_c, .rec_layer = st.rec_layer, .proj_layer = st.proj_layer, .proj_row = st.proj_row, .kv_lora = c.kv_lora, .i_dim = c.i_dim, .hidden = c.hidden };
    }
};

pub const Metadata = struct {
    magic: u64 = 0x32534b4346474c4d,
    version: u32 = 2,
    shape: Shape,
    mark: u64,
    mtp_len: u64,
    bases: [2]u64,
    cur_count: u32,
    cur: [max_layers]u8 = @splat(0),
    ring_from: i64 = -1,
    draft_bytes: u64 = 0,
    payload: u64,
    geometry: [32]u8,
    pages_required: u64,
    media: Media,

    /// Call while capture and state still refer to the same paused committed mark.
    pub fn fromCapture(saved: *const cp.Capture, st: *const state.State, c: *const Config, media: Media) !Metadata {
        if (st.pos != saved.pos or st.mtp_len != saved.mtp_len or st.pending != 0 or st.base != saved.bases[0] or st.mtp_base != saved.bases[1] or !std.mem.eql(u1, st.cur, saved.cur[0..saved.cur_len])) return error.NotAtMark;
        var m: Metadata = .{ .shape = try Shape.of(st, c), .mark = saved.pos, .mtp_len = saved.mtp_len, .bases = .{ saved.bases[0], saved.bases[1] }, .cur_count = @intCast(saved.cur_len), .ring_from = saved.ring_from orelse -1, .draft_bytes = if (saved.ring_from != null) df.ring_bytes else 0, .payload = saved.plan.len, .geometry = geometry(&saved.plan, st, saved.ring_from != null), .pages_required = requiredPages(&saved.plan), .media = media };
        for (saved.cur[0..saved.cur_len], 0..) |v, i| m.cur[i] = v;
        try m.validate();
        return m;
    }
    pub fn validate(m: *const Metadata) !void {
        if (m.magic != 0x32534b4346474c4d or m.version != 2 or m.shape.version != 1 or m.shape.flags & ~@as(u32, 3) != 0 or m.cur_count > max_layers or m.cur_count != m.shape.kda_layers or m.mark > std.math.maxInt(i32) or m.mtp_len > std.math.maxInt(i32) or m.mtp_len > m.mark + state.window_rows) return error.CheckpointMetadata;
        const head = m.shape.flags & 2 != 0;
        if (!head and (m.mtp_len != 0 or m.bases[1] != 0)) return error.CheckpointMetadata;
        if (head and m.mark / state.page_rows > m.mtp_len / state.page_rows + 1) return error.UnsupportedHeadLag;
        if (m.bases[0] > m.mark or m.mark - m.bases[0] > m.shape.ring_rows or m.bases[1] > m.mtp_len or m.mtp_len - m.bases[1] > m.shape.ring_rows or m.bases[0] % sparse.pool != 0 or m.bases[1] % sparse.pool != 0) return error.CheckpointMetadata;
        for (m.cur[0..m.cur_count]) |v| if (v > 1) return error.CheckpointMetadata;
        for (m.cur[m.cur_count..]) |v| if (v != 0) return error.CheckpointMetadata;
        if (m.ring_from == -1) {
            if (m.draft_bytes != 0) return error.CheckpointDraft;
        } else {
            if (m.ring_from < 0 or @as(u64, @intCast(m.ring_from)) >= m.mark or m.mark - @as(u64, @intCast(m.ring_from)) > df.WIN or m.draft_bytes != df.ring_bytes) return error.CheckpointDraft;
        }
        const rows = @max(m.mark, m.mtp_len);
        if (m.pages_required < (rows + state.page_rows - 1) / state.page_rows or m.pages_required > (rows + 2 * sparse.pool + state.page_rows - 1) / state.page_rows) return error.CheckpointMetadata;
        try m.media.validate(m.mark);
    }
    pub fn bind(m: *const Metadata, identity: epoch.Binding) epoch.Binding {
        var b = identity;
        b.layout = m.shape.layoutIdentity();
        b.media = wire.hash(m.media);
        return b;
    }
    pub fn envelope(m: *const Metadata, agreement: epoch.Agreement, rank: u32) !epoch.Envelope {
        try agreement.validate();
        try m.validate();
        if (rank >= agreement.world or agreement.mark != m.mark or !std.mem.eql(u8, &agreement.binding.layout, &m.shape.layoutIdentity()) or !std.mem.eql(u8, &agreement.binding.media, &wire.hash(m.media)) or (m.ring_from == -1) != std.mem.allEqual(u8, &agreement.binding.drafter, 0)) return error.CheckpointRankIdentity;
        const local_heads = m.shape.total_heads / agreement.world + @intFromBool(rank < m.shape.total_heads % agreement.world);
        if (local_heads == 0 or m.shape.heads != local_heads) return error.CheckpointRankIdentity;
        return .{ .rank = rank, .agreement = agreement, .payload = m.payload, .detail_bytes = @intCast(wire.size(Metadata)) };
    }
};

/// Required family validator for epoch.Store(rows). It runs on every rank before commit and during recovery.
pub fn validateEnvelope(envelope: epoch.Envelope, detail: []const u8) !void {
    const m = try wire.decode(Metadata, detail);
    const expected = try m.envelope(envelope.agreement, envelope.rank);
    if (!std.meta.eql(envelope, expected)) return error.CheckpointRankIdentity;
}

fn requiredPages(plan: *const layout.Plan) usize {
    var pages: usize = 0;
    for (plan.parts[0..plan.count]) |p| if (p.span != 0) {
        pages = @max(pages, (p.len - 1) / p.span + 1);
    };
    return pages;
}

fn geometry(plan: *const layout.Plan, st: *const state.State, draft: bool) [32]u8 {
    var h = Sha.init(.{});
    h.update("GLM checkpoint Plan geometry 1");
    for (plan.parts[0..plan.count], 0..) |p, i| {
        const outside = draft and i + 1 == plan.count;
        const origin = if (outside) p.live else if (p.span == 0) st.arena.buf.ptr else st.caches[0].lc;
        const desc = [_]u64{ @intFromBool(outside), p.live - origin, p.len, p.span, p.page };
        var bytes: [wire.size(@TypeOf(desc))]u8 = undefined;
        h.update(wire.encode(desc, &bytes) catch unreachable);
    }
    return h.finalResult();
}

pub const DraftDestination = struct { live: u64, bytes: u64 };
pub const Ready = struct { mark: usize, ring_from: ?i64 }; // local completion only; not TP activation

/// Owns the admitted lease while inactive. Success retains it for the scheduler; failure permits only release.
/// State, drafter ring, transfer slab and stream are exclusive to this restore until it returns.
pub const Inactive = struct {
    pool: *core.kv_pool.Pool,
    live: *core.kv_pool.Lease,
    st: *state.State,
    stream: cuda.Stream,
    draft: ?DraftDestination = null,
    phase: enum { admitted, loading, ready, poisoned, released } = .admitted,

    pub fn init(pool: *core.kv_pool.Pool, live: *core.kv_pool.Lease, st: *state.State, stream: cuda.Stream, draft: ?DraftDestination) !Inactive {
        if (live.ids.items.len != 0 or live.promised == 0 or st.pages.len != 0 or st.capacity != 0 or st.pos != 0 or st.mtp_len != 0 or st.pending != 0 or st.table == 0) return error.CheckpointNotInactive;
        if (stream.d != st.arena.buf.d) return error.CheckpointNotInactive;
        return .{ .pool = pool, .live = live, .st = st, .stream = stream, .draft = draft };
    }

    /// Reads each payload byte once into a bounded slab, then directly into the destination Plan's physical pieces.
    /// Hashes are verified before restoreHost or Ready escapes. Slab reuse is fenced even on a failed copy.
    pub fn load(dst: *Inactive, reader: anytype, c: *const Config, slab: []u8) !Ready {
        if (dst.phase != .admitted) return error.CheckpointRestoreState;
        dst.phase = .loading;
        errdefer {
            cuda.completion.drain(dst.stream.d, dst.stream, "a failed GLM checkpoint restore did not drain");
            dst.phase = .poisoned;
        }
        if (!reader.durable()) return error.CheckpointNotDurable;
        if (slab.len < 4 or slab.len > epoch.max_chunk) return error.CheckpointReadState;
        const m = try wire.decode(Metadata, reader.detail());
        _ = try m.envelope(reader.envelope.agreement, reader.envelope.rank);
        if (reader.envelope.payload != m.payload or !std.meta.eql(m.shape, try Shape.of(dst.st, c)) or dst.pool.page_bytes != m.shape.page_bytes) return error.CheckpointShape;
        if (m.pages_required > dst.live.promised or m.pages_required > (dst.st.arena.buf.len - @as(usize, @intCast(dst.st.table - dst.st.arena.buf.ptr))) / 4) return error.PastReservation;
        if ((m.ring_from != -1) != (dst.draft != null)) return error.CheckpointDraft;
        if (dst.draft) |d| if (d.live == 0 or d.bytes != m.draft_bytes or d.live > std.math.maxInt(u64) - d.bytes) return error.CheckpointDraft;
        // Uses the existing pool's reservation, never a prompt-cache snapshot or a second full-prefix allocation.
        try dst.pool.cover(dst.live, @intCast(m.pages_required), null);
        dst.st.pages = dst.live.ids.items;
        dst.st.capacity = dst.live.ids.items.len * state.page_rows;
        var plan = try layout.Plan.init(dst.st, c, @intCast(m.mark), if (dst.st.head) @as(usize, @intCast(m.mtp_len)) else null);
        if (dst.draft) |d| try plan.outside(d.live, @intCast(d.bytes));
        if (plan.len != m.payload or requiredPages(&plan) != m.pages_required or !std.mem.eql(u8, &m.geometry, &geometry(&plan, dst.st, dst.draft != null))) return error.CheckpointShape;
        // Map the newly allocated physical ids; never restore the source's page table.
        const table = std.mem.sliceAsBytes(dst.live.ids.items);
        var offset: usize = 0;
        while (offset < table.len) {
            const n = @min(slab.len, table.len - offset);
            @memcpy(slab[0..n], table[offset..][0..n]);
            try dst.upload(dst.st.table + offset, slab[0..n]);
            offset += n;
        }
        var part: usize = 0;
        var done: usize = 0;
        var total: u64 = 0;
        while (true) {
            const n = try reader.read(slab);
            if (n == 0) break;
            if (n > slab.len or total > plan.len or n > plan.len - total) return error.CheckpointPayloadSize;
            var used: usize = 0;
            while (used < n) {
                while (part < plan.count and done == plan.parts[part].len) {
                    part += 1;
                    done = 0;
                }
                if (part == plan.count) return error.CheckpointPayloadSize;
                const piece = plan.piece(plan.parts[part], done);
                const len = @min(n - used, piece.len);
                try dst.upload(piece.live, slab[used..][0..len]);
                used += len;
                done += len;
            }
            total += n;
        }
        if (total != plan.len) return error.CheckpointPayloadSize;
        _ = try reader.finish();
        var cur: [max_layers]u1 = undefined;
        for (m.cur[0..m.cur_count], 0..) |v, i| cur[i] = @intCast(v);
        try dst.st.restoreHost(dst.stream, @intCast(m.mark), @intCast(m.mtp_len), cur[0..m.cur_count], .{ @intCast(m.bases[0]), @intCast(m.bases[1]) });
        try dst.stream.synchronize();
        try reader.check();
        dst.phase = .ready;
        return .{ .mark = @intCast(m.mark), .ring_from = if (m.ring_from < 0) null else m.ring_from };
    }

    fn upload(dst: *Inactive, address: u64, bytes: []const u8) !void {
        const d = dst.stream.d;
        try d.check(d.api.cuMemcpyHtoDAsync_v2(address, bytes.ptr, bytes.len, dst.stream.handle), "cuMemcpyHtoDAsync checkpoint restore");
        try dst.stream.synchronize();
    }

    /// After ready, the scheduler may retain this owner with the live sequence. Release also handles failure/cancel.
    pub fn release(dst: *Inactive) void {
        if (dst.phase == .released) return;
        cuda.completion.drain(dst.stream.d, dst.stream, "checkpoint restore release did not drain");
        dst.st.pages = &.{};
        dst.st.capacity = 0;
        dst.pool.finish(dst.live);
        dst.phase = .released;
    }
};

comptime {
    std.debug.assert(wire.size(Metadata) + wire.size(epoch.Envelope) <= epoch.max_metadata);
}
