//! Row blocks' and states' names: SHA-256 over the identity, the tokens through them, decoded spans and chunk starts.
const std = @import("std");
const Sha = std.crypto.hash.sha2.Sha256;
const Span = [2]u32;
const Identity = @import("cache_images.zig").Identity;

pub const Name = [32]u8;

/// A name's first 16 bytes: its file's stem, and the store's key for it (the trailer carries the whole name).
pub const Stem = [16]u8;

pub fn stem(n: Name) Stem {
    return n[0..16].*;
}

/// What every name of one store hashes first: its identity, block size, lookahead and whether it plans chunks.
pub const Keying = struct {
    identity: u64,
    block: u32,
    lookahead: u32 = 0,
    planned: bool = false,

    fn seed(k: Keying) Name {
        var h = Sha.init(.{});
        h.update("tensorfold rows 1");
        words(&h, &.{ @truncate(k.identity), @truncate(k.identity >> 32), k.block, k.lookahead, @intFromBool(k.planned) });
        return h.finalResult();
    }
};

fn words(h: *Sha, w: []const u32) void {
    for (w) |x| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, x, .little);
        h.update(&b);
    }
}

/// The spans and starts of [lo, hi), each clipped to it, after their counts.
fn edges(h: *Sha, spans: []const Span, starts: []const u32, lo: u32, hi: u32, planned: bool) void {
    var n: u32 = 0;
    for (spans) |s| n += @intFromBool(s[0] < hi and s[1] > lo);
    words(h, &.{n});
    for (spans) |s| if (s[0] < hi and s[1] > lo) words(h, &.{ @max(s[0], lo), @min(s[1], hi) });
    if (!planned) return;
    n = 0;
    for (starts) |s| n += @intFromBool(s >= lo and s < hi);
    words(h, &.{n});
    for (starts) |s| if (s >= lo and s < hi) words(h, &.{s});
}

/// The first chunk start at or past `at` (a planned state's position is one, so a block's last chunk ends by it).
fn nextStart(starts: []const u32, at: u32, end: u32) u32 {
    var best = end;
    for (starts) |s| if (s >= at and s < best) {
        best = s;
    };
    return best;
}

fn hashImage(h: *Sha, img: Identity) void {
    words(h, &.{ img.at, img.tokens, img.gh, img.gw, @truncate(img.hash), @truncate(img.hash >> 32) });
}

/// Names of blocks [0, n) of group `window` for a state at `at` (`tokens` holds at + lookahead), into `out`.
pub fn blocks(k: Keying, tokens: []const u32, spans: []const Span, starts: []const u32, at: u32, window: u32, n: u32, out: []Name) void {
    blocksImages(k, tokens, spans, starts, &.{}, 0, at, window, n, out);
}

/// Names of blocks [0, n), folding image_frame (all blocks) and image identities entering each block or its lookahead.
pub fn blocksImages(k: Keying, tokens: []const u32, spans: []const Span, starts: []const u32, images: []const Identity, image_frame: u32, at: u32, window: u32, n: u32, out: []Name) void {
    var d = k.seed();
    if (image_frame != 0) {
        var fh = Sha.init(.{});
        fh.update(&d);
        fh.update("image frame 1");
        words(&fh, &.{image_frame});
        d = fh.finalResult();
    }
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        const lo = i * k.block;
        const hi = lo + k.block;
        var h = Sha.init(.{});
        h.update(&d);
        h.update(std.mem.sliceAsBytes(tokens[lo..hi]));
        edges(&h, spans, starts, lo, hi, k.planned);
        var n_in: u32 = 0;
        for (images) |img| n_in += @intFromBool(img.at >= lo and img.at < hi);
        if (n_in > 0) {
            h.update("images 1");
            words(&h, &.{n_in});
            for (images) |img| if (img.at >= lo and img.at < hi) hashImage(&h, img);
        }
        d = h.finalResult();
        var b = Sha.init(.{});
        b.update(&d);
        words(&b, &.{ window, k.lookahead });
        b.update(std.mem.sliceAsBytes(tokens[hi..][0..k.lookahead]));
        if (k.planned) words(&b, &.{nextStart(starts, hi, at)});
        if (k.lookahead > 0) {
            const la_end = hi + k.lookahead;
            var n_la: u32 = 0;
            for (images) |img| n_la += @intFromBool(img.at >= hi and img.at < la_end);
            if (n_la > 0) {
                b.update("images lookahead 1");
                words(&b, &.{n_la});
                for (images) |img| if (img.at >= hi and img.at < la_end) hashImage(&b, img);
            }
        }
        out[i] = b.finalResult();
    }
}

/// A state's name: its position, every token it read, its decoded spans and (planned) its chunk starts below it.
pub fn state(k: Keying, tokens: []const u32, spans: []const Span, starts: []const u32, at: u32) Name {
    return stateImages(k, tokens, spans, starts, &.{}, 0, at);
}

/// A state's name including optional image frame and image identities (identical to `state` when both are absent).
pub fn stateImages(k: Keying, tokens: []const u32, spans: []const Span, starts: []const u32, images: []const Identity, image_frame: u32, at: u32) Name {
    var h = Sha.init(.{});
    h.update(&k.seed());
    words(&h, &.{ at, @intCast(tokens.len) });
    h.update(std.mem.sliceAsBytes(tokens));
    edges(&h, spans, starts, 0, at, k.planned);
    if (image_frame != 0 or images.len > 0) {
        h.update("images 1");
        words(&h, &.{ image_frame, @intCast(images.len) });
        for (images) |img| hashImage(&h, img);
    }
    return h.finalResult();
}

/// A name's stem in hex (a Name or a Stem): its file's name before the extension.
pub fn hex(n: anytype) [32]u8 {
    return std.fmt.bytesToHex(n[0..16].*, .lower);
}

test "a block's name follows every token through it and its lookahead, never a later one" {
    const k: Keying = .{ .identity = 9, .block = 4, .lookahead = 1 };
    var t1 = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    var t2 = t1;
    t2[9] = 99; // past block 1's lookahead (token 8)
    var a: [2]Name = undefined;
    var b: [2]Name = undefined;
    blocks(k, &t1, &.{}, &.{}, 9, 0, 2, &a);
    blocks(k, &t2, &.{}, &.{}, 9, 0, 2, &b);
    try std.testing.expectEqualSlices(Name, &a, &b);
    t2[8] = 77; // block 1's lookahead token
    blocks(k, &t2, &.{}, &.{}, 9, 0, 2, &b);
    try std.testing.expectEqual(a[0], b[0]);
    try std.testing.expect(!std.mem.eql(u8, &a[1], &b[1]));
    t2 = t1;
    t2[0] = 0; // block 0's tokens change every later block
    blocks(k, &t2, &.{}, &.{}, 9, 0, 2, &b);
    try std.testing.expect(!std.mem.eql(u8, &a[0], &b[0]) and !std.mem.eql(u8, &a[1], &b[1]));
}

test "decoded spans, a planned chunk's end, the group and the identity all move a name" {
    const k: Keying = .{ .identity = 9, .block = 4, .planned = true };
    const t = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    var base: [2]Name = undefined;
    var x: [2]Name = undefined;
    blocks(k, &t, &.{}, &.{ 6, 12 }, 12, 0, 2, &base);
    blocks(k, &t, &.{.{ 5, 7 }}, &.{ 6, 12 }, 12, 0, 2, &x); // a decoded span in block 1 only
    try std.testing.expectEqual(base[0], x[0]);
    try std.testing.expect(!std.mem.eql(u8, &base[1], &x[1]));
    blocks(k, &t, &.{}, &.{ 6, 10, 12 }, 12, 0, 2, &x); // block 1's last chunk ends at 10, not 12
    try std.testing.expectEqual(base[0], x[0]);
    try std.testing.expect(!std.mem.eql(u8, &base[1], &x[1]));
    blocks(k, &t, &.{}, &.{ 3, 6, 12 }, 12, 0, 2, &x); // a start inside block 0 moves both
    try std.testing.expect(!std.mem.eql(u8, &base[0], &x[0]) and !std.mem.eql(u8, &base[1], &x[1]));
    blocks(k, &t, &.{}, &.{ 6, 12 }, 12, 7, 2, &x);
    try std.testing.expect(!std.mem.eql(u8, &base[0], &x[0]));
    blocks(.{ .identity = 8, .block = 4, .planned = true }, &t, &.{}, &.{ 6, 12 }, 12, 0, 2, &x);
    try std.testing.expect(!std.mem.eql(u8, &base[0], &x[0]));
}

test "an unplanned family's names ignore chunk starts" {
    const k: Keying = .{ .identity = 9, .block = 4 };
    const t = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    var a: [2]Name = undefined;
    var b: [2]Name = undefined;
    blocks(k, &t, &.{}, &.{3}, 8, 0, 2, &a);
    blocks(k, &t, &.{}, &.{ 2, 5 }, 8, 0, 2, &b);
    try std.testing.expectEqualSlices(Name, &a, &b);
    try std.testing.expectEqual(state(k, &t, &.{}, &.{3}, 8), state(k, &t, &.{}, &.{ 2, 5 }, 8));
}

test "image identities share pre-image blocks, separate on image fields, lookahead boundaries and frame" {
    const k: Keying = .{ .identity = 9, .block = 4, .lookahead = 1 };
    const t = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const img_a = [_]Identity{.{ .at = 5, .tokens = 2, .gh = 2, .gw = 2, .hash = 101 }};
    const img_b = [_]Identity{.{ .at = 5, .tokens = 2, .gh = 2, .gw = 2, .hash = 102 }};
    var plain: [2]Name = undefined;
    var a: [2]Name = undefined;
    var b: [2]Name = undefined;
    blocks(k, &t, &.{}, &.{}, 8, 0, 2, &plain);
    blocksImages(k, &t, &.{}, &.{}, &.{}, 0, 8, 0, 2, &a);
    try std.testing.expectEqualSlices(Name, &plain, &a);
    try std.testing.expectEqual(state(k, &t, &.{}, &.{}, 8), stateImages(k, &t, &.{}, &.{}, &.{}, 0, 8));
    // Image in block 1 shares block 0 and separates block 1 and state name.
    blocksImages(k, &t, &.{}, &.{}, &img_a, 0, 8, 0, 2, &a);
    blocksImages(k, &t, &.{}, &.{}, &img_b, 0, 8, 0, 2, &b);
    try std.testing.expectEqual(plain[0], a[0]);
    try std.testing.expectEqual(a[0], b[0]);
    try std.testing.expect(!std.mem.eql(u8, &a[1], &b[1]));
    try std.testing.expect(!std.mem.eql(u8, &stateImages(k, &t, &.{}, &.{}, &img_a, 0, 8), &stateImages(k, &t, &.{}, &.{}, &img_b, 0, 8)));
    // Image at block 0's lookahead boundary (at = 4) separates block 0 as well as block 1.
    const la_a = [_]Identity{.{ .at = 4, .tokens = 1, .gh = 2, .gw = 2, .hash = 101 }};
    const la_b = [_]Identity{.{ .at = 4, .tokens = 1, .gh = 2, .gw = 2, .hash = 102 }};
    blocksImages(k, &t, &.{}, &.{}, &la_a, 0, 8, 0, 2, &a);
    blocksImages(k, &t, &.{}, &.{}, &la_b, 0, 8, 0, 2, &b);
    try std.testing.expect(!std.mem.eql(u8, &a[0], &b[0]) and !std.mem.eql(u8, &a[1], &b[1]));
    // Nonzero image_frame separates every block from block 0 onward.
    blocksImages(k, &t, &.{}, &.{}, &.{}, 7, 8, 0, 2, &a);
    blocksImages(k, &t, &.{}, &.{}, &.{}, 8, 8, 0, 2, &b);
    try std.testing.expect(!std.mem.eql(u8, &plain[0], &a[0]) and !std.mem.eql(u8, &a[0], &b[0]));
}

/// A planned state's starts with its own position last: its blocks' last chunk ends by it.
pub fn withAt(a: std.mem.Allocator, starts: []const u32, at: u32, planned: bool) ![]u32 {
    if (!planned) return a.alloc(u32, 0);
    var out = try a.alloc(u32, starts.len + 1);
    @memcpy(out[0..starts.len], starts);
    out[starts.len] = at;
    return out;
}

/// Whether a prompt's chunk starts below `at` are the state's, and `at` is one of them.
pub fn sameBelow(starts: []const u32, kept: []const u32, at: u32) bool {
    var n: usize = 0;
    while (n < starts.len and starts[n] < at) n += 1;
    return std.mem.eql(u32, starts[0..n], kept) and std.mem.indexOfScalar(u32, starts, at) != null;
}

/// Whether `kept` are the grid's starts below `at`.
pub fn onGrid(kept: []const u32, at: u32, grid: u32) bool {
    if (kept.len != (at -| 1) / grid) return false;
    for (kept, 1..) |k, i| if (k != i * grid) return false;
    return true;
}

/// Whether `at` is usable under `starts` (marks mode): on `grid` when `starts` is empty, else in `starts`.
pub fn marksMatch(starts: []const u32, at: u32, grid: u32) bool {
    if (starts.len == 0) return grid > 0 and at % grid == 0;
    return std.mem.indexOfScalar(u32, starts, at) != null;
}

/// Copy `starts` strictly below `at` without grid expansion (used for family-reported executed starts).
pub fn belowSlice(a: std.mem.Allocator, starts: []const u32, at: u32) ![]u32 {
    var n: usize = 0;
    for (starts) |c| n += @intFromBool(c < at);
    const out = try a.alloc(u32, n);
    var i: usize = 0;
    for (starts) |c| if (c < at) {
        out[i] = c;
        i += 1;
    };
    return out;
}
