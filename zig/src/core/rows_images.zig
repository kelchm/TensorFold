//! Disk state metadata encoding, decoding, and conservative validation for durable image conversations.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const cache_images = @import("cache_images.zig");
const Image = @import("lanes").stream.Image;
const Allocator = std.mem.Allocator;

pub const Identity = cache_images.Identity;

/// Trailer.flags bit 1 marks versioned image metadata appended after chunk starts (bit 0 is `shared`).
pub const flag: u32 = 2;
pub const flag_mask: u32 = 3;
pub const format_version: u32 = 1;
pub const header_bytes: usize = 12; // version, image_frame, count (all u32 LE)
pub const record_bytes: usize = 24; // at, tokens, gh, gw (u32 LE), hash (u64 LE)

pub const Meta = struct {
    tokens: []u32 = &.{},
    spans: []modes.Span = &.{},
    starts: []u32 = &.{},
    images: []Identity = &.{},
    image_frame: u32 = 0,
};

pub fn hasMeta(images: []const Identity, image_frame: u32) bool {
    return image_frame != 0 or images.len > 0;
}

pub fn metaBytes(images: []const Identity, image_frame: u32) usize {
    return if (hasMeta(images, image_frame)) header_bytes + images.len * record_bytes else 0;
}

/// Tight upper bound on serialized state metadata bytes, checked before allocating in readState and coldCharge.
pub fn maxMetaBytes(at: u32, lookahead: u32, flags_word: u32) u64 {
    const nt: u64 = @as(u64, at) + lookahead;
    const base: u64 = 12 + nt * 16; // counts + nt tokens + <=nt spans (8 B) + <=nt starts (4 B)
    return if (flags_word & flag == 0) base else base + header_bytes + nt * record_bytes;
}

pub fn flags(shared: bool, images: []const Identity, image_frame: u32) u32 {
    return @as(u32, @intFromBool(shared)) | (if (hasMeta(images, image_frame)) flag else 0);
}

/// Marker words matching server/image_key.zig: open holds hash bits [0, 31), close holds [31, 62).
pub fn imageOpen(hash: u64) u32 {
    return 0x8000_0000 | @as(u32, @truncate(hash & 0x7fff_ffff));
}

pub fn imageClose(hash: u64) u32 {
    return 0x8000_0000 | @as(u32, @truncate((hash >> 31) & 0x7fff_ffff));
}

fn attributedBy(img: Identity, i: usize, word: u32) bool {
    const at: usize = img.at;
    const end: usize = at + img.tokens;
    if (i >= at and i < end) return true;
    if (at > 0 and i == at - 1) return word == imageOpen(img.hash);
    if (i == end) return word == imageClose(img.hash);
    return false;
}

/// Every reserved-bit cache word must belong to the framed prefix word, a valid image span, or its exact markers.
pub fn tokensAttributed(tokens: []const u32, images: []const Identity, image_frame: u32) bool {
    for (images, 0..) |img, k| {
        if (img.at >= tokens.len or img.tokens == 0 or img.gh == 0 or img.gw == 0) return false;
        _ = std.math.add(u32, img.at, img.tokens) catch return false;
        if (k > 0) {
            const prev_end = std.math.add(u32, images[k - 1].at, images[k - 1].tokens) catch return false;
            if (img.at < prev_end) return false;
        }
    }
    var cur: usize = 0;
    for (tokens, 0..) |word, i| {
        if (word & 0x8000_0000 == 0) continue;
        if (i == 0 and image_frame != 0) continue;
        while (cur < images.len and @as(usize, images[cur].at) + images[cur].tokens < i) cur += 1;
        if (cur < images.len and attributedBy(images[cur], i, word)) continue;
        if (cur + 1 < images.len and attributedBy(images[cur + 1], i, word)) continue;
        return false;
    }
    return true;
}

/// Salted entries and any unattributed reserved-bit cache word stay RAM-only; plain and attributed states persist.
pub fn diskEligible(e: *const pc.Entry) bool {
    if (e.salt != null) return false;
    return tokensAttributed(e.tokens, e.images, e.image_frame);
}

pub fn matches(images: []const Identity, image_frame: u32, tokens_len: usize, query: ?[]const Image, query_frame: u32) bool {
    return image_frame == query_frame and cache_images.equal(images, query orelse &.{}, @intCast(tokens_len));
}

/// Encode versioned image metadata in explicit little-endian wire order (empty when neither frame nor images exist).
pub fn encode(a: Allocator, images: []const Identity, image_frame: u32) ![]u8 {
    if (!hasMeta(images, image_frame)) return a.alloc(u8, 0);
    const out = try a.alloc(u8, header_bytes + images.len * record_bytes);
    std.mem.writeInt(u32, out[0..4], format_version, .little);
    std.mem.writeInt(u32, out[4..8], image_frame, .little);
    std.mem.writeInt(u32, out[8..12], @intCast(images.len), .little);
    for (images, 0..) |img, i| {
        const rec = out[header_bytes + i * record_bytes ..][0..record_bytes];
        std.mem.writeInt(u32, rec[0..4], img.at, .little);
        std.mem.writeInt(u32, rec[4..8], img.tokens, .little);
        std.mem.writeInt(u32, rec[8..12], img.gh, .little);
        std.mem.writeInt(u32, rec[12..16], img.gw, .little);
        std.mem.writeInt(u64, rec[16..24], img.hash, .little);
    }
    return out;
}

/// Parse state metadata (tokens, spans, starts, and optional versioned image metadata); invalid inputs return RowsRead.
pub fn parseMeta(a: Allocator, m: []const u8, flags_word: u32) !Meta {
    if (flags_word & ~flag_mask != 0 or m.len < 12) return error.RowsRead;
    const nt: usize = std.mem.readInt(u32, m[0..4], .little);
    const ns: usize = std.mem.readInt(u32, m[4..8], .little);
    const nst: usize = std.mem.readInt(u32, m[8..12], .little);
    if (nt == 0 or ns > nt or nst > nt) return error.RowsRead;
    const tok_bytes = std.math.mul(usize, nt, 4) catch return error.RowsRead;
    const span_bytes = std.math.mul(usize, ns, 8) catch return error.RowsRead;
    const start_bytes = std.math.mul(usize, nst, 4) catch return error.RowsRead;
    const base_1 = std.math.add(usize, 12, tok_bytes) catch return error.RowsRead;
    const base_2 = std.math.add(usize, base_1, span_bytes) catch return error.RowsRead;
    const base = std.math.add(usize, base_2, start_bytes) catch return error.RowsRead;
    if (m.len < base) return error.RowsRead;
    var out: Meta = .{};
    errdefer {
        a.free(out.tokens);
        a.free(out.spans);
        a.free(out.starts);
        a.free(out.images);
    }
    out.tokens = try a.alloc(u32, nt);
    @memcpy(std.mem.sliceAsBytes(out.tokens), m[12..][0..tok_bytes]);
    out.spans = try a.alloc(modes.Span, ns);
    @memcpy(std.mem.sliceAsBytes(out.spans), m[base_1..][0..span_bytes]);
    out.starts = try a.alloc(u32, nst);
    @memcpy(std.mem.sliceAsBytes(out.starts), m[base_2..][0..start_bytes]);
    const tail = m[base..];
    if (flags_word & flag == 0) {
        if (tail.len != 0 or !tokensAttributed(out.tokens, &.{}, 0)) return error.RowsRead;
        return out;
    }
    if (tail.len < header_bytes) return error.RowsRead;
    const ver = std.mem.readInt(u32, tail[0..4], .little);
    const frame = std.mem.readInt(u32, tail[4..8], .little);
    const count: usize = std.mem.readInt(u32, tail[8..12], .little);
    if (ver != format_version or (frame == 0 and count == 0) or count > nt) return error.RowsRead;
    const body = std.math.mul(usize, count, record_bytes) catch return error.RowsRead;
    const need = std.math.add(usize, header_bytes, body) catch return error.RowsRead;
    if (tail.len != need) return error.RowsRead;
    out.images = try a.alloc(Identity, count);
    out.image_frame = frame;
    for (out.images, 0..) |*dst, k| {
        const rec = tail[header_bytes + k * record_bytes ..][0..record_bytes];
        dst.* = .{
            .at = std.mem.readInt(u32, rec[0..4], .little),
            .tokens = std.mem.readInt(u32, rec[4..8], .little),
            .gh = std.mem.readInt(u32, rec[8..12], .little),
            .gw = std.mem.readInt(u32, rec[12..16], .little),
            .hash = std.mem.readInt(u64, rec[16..24], .little),
        };
    }
    if (!tokensAttributed(out.tokens, out.images, out.image_frame)) return error.RowsRead;
    return out;
}
