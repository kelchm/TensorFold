//! Durable image conversations on disk: exact identity persistence, boundaries, restart, and corrupt fallback.
const std = @import("std");
const platform = @import("sys");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const fake = @import("rows_fake.zig");
const Image = @import("lanes").stream.Image;
const rmTree = @import("prompt_imprint_test.zig").rmTree;
const t = std.testing;
const a = t.allocator;

fn opts(root: []const u8, sync: bool) rows.Options {
    return .{ .root = root, .identity = 71, .cap = 1 << 30, .daily = 1 << 30, .floor = 0, .lookahead = 1, .sync_writes = sync };
}

fn drainAll(r: *rows.Rows) void {
    const deadline = rows.write.nowNs() + 5 * std.time.ns_per_s;
    while (rows.write.pending(r) and rows.write.nowNs() < deadline) {
        if (!rows.write.step(r)) rows.write.io.nap();
    }
    rows.write.reap(r);
}

fn makePrompt(buf: []u32, img_at: usize, img_tokens: usize) void {
    for (buf, 0..) |*w, i| w.* = @intCast(100 + i % 997);
    for (img_at..img_at + img_tokens) |i| buf[i] = 0x8000_0100 + @as(u32, @intCast(i - img_at));
}

fn lookupAndRestore(s: *pc.Store, f: *fake.Fake, prompt: []const u32, imgs: []const Image, frame: u32) !u32 {
    f.prompt = prompt;
    f.images = imgs;
    f.image_frame = frame;
    const l = try s.lookupUserFrame(a, prompt, @intCast(prompt.len), 0, 0, 0, &.{}, &.{}, &.{}, imgs, frame);
    defer a.free(l.marks);
    const e = l.entry orelse return 0;
    defer s.release(e);
    try s.family.vtable.restore(s.family.ptr, null, e.saved);
    return e.at;
}

test "image disk: same-text different-pixel and geometry variants survive eviction and restart without cross-contamination" {
    for ([_]bool{ true, false }) |sync| {
        const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
        const root = try std.fmt.allocPrint(a, "{s}/tf-img-disk-{any}-{d}", .{ tmp, sync, platform.c.getpid() });
        defer a.free(root);
        defer rmTree(root);
        var tokens: [900]u32 = undefined;
        makePrompt(&tokens, 300, 16);
        const img_a = [_]Image{.{ .pixels = &.{}, .gh = 4, .gw = 4, .at = 300, .tokens = 16, .hash = 0x1111_2222_3333_4444 }};
        const img_b = [_]Image{.{ .pixels = &.{}, .gh = 4, .gw = 4, .at = 300, .tokens = 16, .hash = 0x5555_6666_7777_8888 }};
        const img_geo = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 8, .at = 300, .tokens = 16, .hash = 0x1111_2222_3333_4444 }};
        const img_other = [_]Image{.{ .pixels = &.{}, .gh = 4, .gw = 4, .at = 300, .tokens = 16, .hash = 0x9999_aaaa_bbbb_cccc }};
        var shared_blocks_before_b: usize = 0;
        {
            var f: fake.Fake = .{ .gpa = a, .prompt = &tokens, .at = 800 };
            var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
            const r = try rows.attach(a, &s, opts(root, sync));
            defer {
                s.deinit();
                r.deinit();
            }
            r.verify = true;
            // Save variant A, then variant B (identical tokens, different pixel hash), then geometry variant.
            f.images = &img_a;
            try t.expect(s.keepBornFrame(&tokens, 800, 800, null, &.{}, &.{}, &img_a, 0));
            drainAll(r);
            shared_blocks_before_b = r.blocks.count();
            try t.expectEqual(@as(usize, 1), r.states.items.len);
            f.images = &img_b;
            try t.expect(s.keepBornFrame(&tokens, 800, 800, null, &.{}, &.{}, &img_b, 0));
            drainAll(r);
            try t.expectEqual(@as(usize, 2), r.states.items.len);
            // Block 0 ([0, 256)) precedes the image at 300 and is shared; blocks 1..2 + ring block are distinct.
            try t.expect(r.blocks.count() < 2 * shared_blocks_before_b);
            f.images = &img_geo;
            try t.expect(s.keepBornFrame(&tokens, 800, 800, null, &.{}, &.{}, &img_geo, 0));
            drainAll(r);
            try t.expectEqual(@as(usize, 3), r.states.items.len);
            try t.expectEqual(@as(u64, 3), r.counts.verified);
            try t.expectEqual(@as(u64, 0), r.counts.differed);
            // Evict RAM while keeping disk, then recall variant B -> exact image B bytes, not A.
            for (s.entries.items) |e| e.expired = true;
            s.collectExpired();
            try t.expectEqual(@as(u32, 800), try lookupAndRestore(&s, &f, &tokens, &img_b, 0));
            try t.expect(f.restored_ok);
            try t.expectEqual(@as(usize, 1), s.entries.items.len);
            try t.expectEqual(@as(u64, img_b[0].hash), s.entries.items[0].images[0].hash);
        }
        // Full restart: fresh Store + Rows.open on same directory.
        var f: fake.Fake = .{ .gpa = a, .prompt = &tokens };
        var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, opts(root, sync));
        defer {
            s.deinit();
            r.deinit();
        }
        try t.expectEqual(@as(usize, 3), r.states.items.len);
        try t.expectEqual(@as(u32, 800), try lookupAndRestore(&s, &f, &tokens, &img_a, 0));
        try t.expect(f.restored_ok);
        try t.expectEqual(@as(u64, 1), r.counts.reads);
        // Variant B must not falsely hit variant A in RAM; it recalls B from disk with exact bytes.
        try t.expectEqual(@as(u32, 800), try lookupAndRestore(&s, &f, &tokens, &img_b, 0));
        try t.expect(f.restored_ok);
        try t.expectEqual(@as(u64, 2), r.counts.reads);
        // Geometry variant recalls its own state cleanly.
        try t.expectEqual(@as(u32, 800), try lookupAndRestore(&s, &f, &tokens, &img_geo, 0));
        try t.expect(f.restored_ok);
        try t.expectEqual(@as(u64, 3), r.counts.reads);
        // Unseen pixels miss both RAM and disk.
        try t.expectEqual(@as(u32, 0), try lookupAndRestore(&s, &f, &tokens, &img_other, 0));
        try t.expectEqual(@as(u64, 3), r.counts.reads);
        // Repeat lookup for A hits RAM with zero extra disk reads.
        try t.expectEqual(@as(u32, 800), try lookupAndRestore(&s, &f, &tokens, &img_a, 0));
        try t.expect(f.restored_ok);
        try t.expectEqual(@as(u64, 3), r.counts.reads);
    }
}

test "image disk: lookahead boundaries and pre-image image_frame survive restart with exact isolation" {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-img-boundary-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer rmTree(root);
    var tokens: [600]u32 = undefined;
    makePrompt(&tokens, 256, 4); // exact block-0 lookahead boundary at position 256
    const at_la_a = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 256, .tokens = 4, .hash = 10 }};
    const at_la_b = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 256, .tokens = 4, .hash = 20 }};
    const past_la = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 257, .tokens = 4, .hash = 30 }};
    var plain_tokens: [600]u32 = undefined;
    for (&plain_tokens, 0..) |*w, i| w.* = @intCast(500 + i);
    {
        var f: fake.Fake = .{ .gpa = a, .prompt = &tokens, .at = 256, .images = &at_la_a };
        var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, opts(root, false));
        defer {
            s.deinit();
            r.deinit();
        }
        // State at 256 with lookahead=1 covers token 256 (image at 256 enters prefix!).
        try t.expect(s.keepBornFrame(&tokens, 256, 256, null, &.{}, &.{}, &at_la_a, 0));
        // Pre-image state with image_frame=42 (no image yet in [0, 257)) also persists.
        f.prompt = &plain_tokens;
        f.images = &past_la;
        f.image_frame = 42;
        try t.expect(s.keepBornFrame(&plain_tokens, 256, 256, null, &.{}, &.{}, &past_la, 42));
        drainAll(r);
        try t.expectEqual(@as(usize, 2), r.states.items.len);
    }
    var f: fake.Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
    const r = try rows.attach(a, &s, opts(root, false));
    defer {
        s.deinit();
        r.deinit();
    }
    try t.expectEqual(@as(usize, 2), r.states.items.len);
    // Boundary image A hits; different image B at lookahead position 256 misses!
    try t.expectEqual(@as(u32, 0), try lookupAndRestore(&s, &f, &tokens, &at_la_b, 0));
    try t.expectEqual(@as(u32, 256), try lookupAndRestore(&s, &f, &tokens, &at_la_a, 0));
    try t.expect(f.restored_ok);
    // Framed pre-image state hits same frame even if future image past lookahead (at 257) changes, misses other frame.
    const past_la_changed = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 257, .tokens = 4, .hash = 99 }};
    try t.expectEqual(@as(u32, 0), try lookupAndRestore(&s, &f, &plain_tokens, &past_la_changed, 43));
    try t.expectEqual(@as(u32, 0), try lookupAndRestore(&s, &f, &plain_tokens, &past_la_changed, 0));
    try t.expectEqual(@as(u32, 256), try lookupAndRestore(&s, &f, &plain_tokens, &past_la_changed, 42));
    try t.expect(f.restored_ok);
}

fn writeRawState(dir: []const u8, name: rows.chain.Name, at: u32, block: u32, flags: u32, meta_bytes: []const u8) !void {
    const stem = rows.chain.hex(name);
    var w = try rows.file.Writer.open(dir, &stem, ".st");
    errdefer w.abort();
    try w.append("own");
    _ = try w.finish(.{ .kind = @backingInt(rows.file.Kind.state), .payload = 0, .payload_hash = 0, .name = name, .at = at, .block = block, .flags = flags }, &.{meta_bytes}, false);
}

fn buildMeta(tokens: []const u32, tail: []const u8) ![]u8 {
    const counts = [3]u32{ @intCast(tokens.len), 0, 0 };
    return std.mem.concat(a, u8, &.{ std.mem.asBytes(&counts), std.mem.sliceAsBytes(tokens), tail });
}

fn freeParsed(m: rows.images.Meta) void {
    a.free(m.tokens);
    a.free(m.spans);
    a.free(m.starts);
    a.free(m.images);
}

fn makeEntry(tokens: []u32, imgs: []rows.images.Identity, frame: u32) pc.Entry {
    return .{ .at = 31, .saved = undefined, .bytes = 1, .born = 31, .used = 1, .last = &.{}, .tokens = tokens, .images = imgs, .image_frame = frame };
}

test "image disk: reserved-bit attribution rejects rogue high-bit tokens in diskEligible and parseMeta" {
    var base_toks: [32]u32 = undefined;
    for (&base_toks, 0..) |*w, i| w.* = @intCast(100 + i);
    const h1: u64 = 0x1234_5678_9abc_def0 & ((@as(u64, 1) << 62) - 1);
    const h2: u64 = 0x0fed_cba9_8765_4321 & ((@as(u64, 1) << 62) - 1);
    var id1 = [_]rows.images.Identity{.{ .at = 10, .tokens = 2, .gh = 2, .gw = 2, .hash = h1 }};
    // Valid wrapped image: exact open at at-1, rows in [at, at+tokens), exact close at at+tokens.
    var wrapped = base_toks;
    wrapped[9] = rows.images.imageOpen(h1);
    wrapped[10] = 0x8000_0001;
    wrapped[11] = 0x8000_0002;
    wrapped[12] = rows.images.imageClose(h1);
    var e = makeEntry(&wrapped, &id1, 0);
    try t.expect(rows.images.diskEligible(&e));
    const enc1 = try rows.images.encode(a, &id1, 0);
    defer a.free(enc1);
    const raw_ok = try buildMeta(&wrapped, enc1);
    defer a.free(raw_ok);
    const parsed_ok = try rows.images.parseMeta(a, raw_ok, rows.images.flag);
    freeParsed(parsed_ok);
    // Adjacent images and single shared marker between two images are attributed cleanly.
    var id_adj = [_]rows.images.Identity{
        .{ .at = 10, .tokens = 2, .gh = 2, .gw = 2, .hash = h1 },
        .{ .at = 12, .tokens = 2, .gh = 2, .gw = 2, .hash = h2 },
    };
    var adj = base_toks;
    adj[10] = 0x8000_0001;
    adj[11] = 0x8000_0002;
    adj[12] = 0x8000_0003;
    adj[13] = 0x8000_0004;
    e = makeEntry(&adj, &id_adj, 0);
    try t.expect(rows.images.diskEligible(&e));
    var id_sep = [_]rows.images.Identity{
        .{ .at = 10, .tokens = 2, .gh = 2, .gw = 2, .hash = h1 },
        .{ .at = 13, .tokens = 2, .gh = 2, .gw = 2, .hash = h2 },
    };
    var sep = base_toks;
    sep[10] = 0x8000_0001;
    sep[12] = rows.images.imageOpen(h2);
    sep[13] = 0x8000_0003;
    e = makeEntry(&sep, &id_sep, 0);
    try t.expect(rows.images.diskEligible(&e));
    // Valid frame-only prefix allows bit 31 only at index 0.
    var framed = base_toks;
    framed[0] |= 0x8000_0000;
    e = makeEntry(&framed, &.{}, 42);
    try t.expect(rows.images.diskEligible(&e));
    const enc_f = try rows.images.encode(a, &.{}, 42);
    defer a.free(enc_f);
    const raw_f = try buildMeta(&framed, enc_f);
    defer a.free(raw_f);
    freeParsed(try rows.images.parseMeta(a, raw_f, rows.images.flag));
    // Rogue high-bit tokens beside a valid image (before, after, wrong open/close) fail diskEligible and parseMeta.
    const bad_img_cases = [_]struct { idx: usize, val: u32 }{
        .{ .idx = 2, .val = 0x8000_1234 },
        .{ .idx = 20, .val = 0x8000_5678 },
        .{ .idx = 9, .val = rows.images.imageOpen(h2) },
        .{ .idx = 12, .val = rows.images.imageClose(h2) },
    };
    for (bad_img_cases) |bad| {
        var toks = wrapped;
        toks[bad.idx] = bad.val;
        e = makeEntry(&toks, &id1, 0);
        try t.expect(!rows.images.diskEligible(&e));
        const raw = try buildMeta(&toks, enc1);
        defer a.free(raw);
        try t.expectError(error.RowsRead, rows.images.parseMeta(a, raw, rows.images.flag));
    }
    // Rogue high-bit token at i > 0 alongside a frame-only (image_frame != 0, images.len == 0) prefix fails both.
    var bad_frame = framed;
    bad_frame[5] |= 0x8000_0000;
    e = makeEntry(&bad_frame, &.{}, 42);
    try t.expect(!rows.images.diskEligible(&e));
    const raw_bf = try buildMeta(&bad_frame, enc_f);
    defer a.free(raw_bf);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, raw_bf, rows.images.flag));
}

test "image disk: parseMeta rejects malformed records, count > nt, and invalid flags directly" {
    var toks: [16]u32 = undefined;
    for (&toks, 0..) |*w, i| w.* = @intCast(50 + i);
    toks[4] = 0x8000_0001;
    toks[5] = 0x8000_0002;
    const good_id = [_]rows.images.Identity{.{ .at = 4, .tokens = 2, .gh = 2, .gw = 2, .hash = 77 }};
    const good_enc = try rows.images.encode(a, &good_id, 0);
    defer a.free(good_enc);
    // 1. Unknown flag bits.
    const m_good = try buildMeta(&toks, good_enc);
    defer a.free(m_good);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_good, 4));
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_good, 6));
    // 2. Unsupported format version (0 and 2).
    for ([_]u32{ 0, 2 }) |ver| {
        const bad = try a.dupe(u8, good_enc);
        defer a.free(bad);
        std.mem.writeInt(u32, bad[0..4], ver, .little);
        const m = try buildMeta(&toks, bad);
        defer a.free(m);
        try t.expectError(error.RowsRead, rows.images.parseMeta(a, m, rows.images.flag));
    }
    // 3. Flag set with image_frame == 0 and count == 0.
    var empty_hdr: [rows.images.header_bytes]u8 = undefined;
    std.mem.writeInt(u32, empty_hdr[0..4], rows.images.format_version, .little);
    std.mem.writeInt(u32, empty_hdr[4..8], 0, .little);
    std.mem.writeInt(u32, empty_hdr[8..12], 0, .little);
    const m_empty = try buildMeta(&toks, &empty_hdr);
    defer a.free(m_empty);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_empty, rows.images.flag));
    // 4. Truncated header, truncated record, and trailing extra byte.
    const m_shorthdr = try buildMeta(&toks, good_enc[0..8]);
    defer a.free(m_shorthdr);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_shorthdr, rows.images.flag));
    const m_shortrec = try buildMeta(&toks, good_enc[0..16]);
    defer a.free(m_shortrec);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_shortrec, rows.images.flag));
    const extra = try std.mem.concat(a, u8, &.{ good_enc, &[_]u8{0} });
    defer a.free(extra);
    const m_extra = try buildMeta(&toks, extra);
    defer a.free(m_extra);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_extra, rows.images.flag));
    // 5. count > nt (both nt + 1 with matching tail bytes and huge count rejected before Identity allocation).
    var many_ids: [17]rows.images.Identity = undefined;
    for (&many_ids, 0..) |*id, i| id.* = .{ .at = @intCast(i % 16), .tokens = 1, .gh = 1, .gw = 1, .hash = i + 1 };
    const many_enc = try rows.images.encode(a, &many_ids, 0);
    defer a.free(many_enc);
    const m_many = try buildMeta(&toks, many_enc);
    defer a.free(m_many);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_many, rows.images.flag));
    var huge_hdr = empty_hdr;
    std.mem.writeInt(u32, huge_hdr[8..12], 1_000_000, .little);
    const m_huge = try buildMeta(&toks, &huge_hdr);
    defer a.free(m_huge);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_huge, rows.images.flag));
    // 6. Invalid record fields: oob at, zero tokens/gh/gw, u32 overflow, overlapping, unsorted.
    const bad_recs = [_][]const rows.images.Identity{
        &.{.{ .at = 16, .tokens = 1, .gh = 2, .gw = 2, .hash = 1 }},
        &.{.{ .at = 4, .tokens = 0, .gh = 2, .gw = 2, .hash = 1 }},
        &.{.{ .at = 4, .tokens = 2, .gh = 0, .gw = 2, .hash = 1 }},
        &.{.{ .at = 4, .tokens = 2, .gh = 2, .gw = 0, .hash = 1 }},
        &.{.{ .at = 4, .tokens = std.math.maxInt(u32), .gh = 2, .gw = 2, .hash = 1 }},
        &.{ .{ .at = 4, .tokens = 3, .gh = 2, .gw = 2, .hash = 1 }, .{ .at = 6, .tokens = 1, .gh = 2, .gw = 2, .hash = 2 } },
        &.{ .{ .at = 8, .tokens = 1, .gh = 2, .gw = 2, .hash = 1 }, .{ .at = 4, .tokens = 2, .gh = 2, .gw = 2, .hash = 2 } },
    };
    for (bad_recs) |recs| {
        const enc = try rows.images.encode(a, recs, 0);
        defer a.free(enc);
        const m = try buildMeta(&toks, enc);
        defer a.free(m);
        try t.expectError(error.RowsRead, rows.images.parseMeta(a, m, rows.images.flag));
    }
    // 7. Flag clear with reserved high-bit token or non-empty tail bytes.
    const m_noflag_tok = try buildMeta(&toks, &.{});
    defer a.free(m_noflag_tok);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_noflag_tok, 0));
    var plain: [16]u32 = undefined;
    for (&plain, 0..) |*w, i| w.* = @intCast(50 + i);
    const m_noflag_tail = try buildMeta(&plain, &[_]u8{1});
    defer a.free(m_noflag_tail);
    try t.expectError(error.RowsRead, rows.images.parseMeta(a, m_noflag_tail, 0));
}

test "image disk: store open drops oversized meta, bad trailer fields, and hash-mismatched states" {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-img-corrupt-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer rmTree(root);
    var tokens: [320]u32 = undefined;
    makePrompt(&tokens, 40, 4);
    const valid_img = [_]Image{.{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 40, .tokens = 4, .hash = 77 }};
    var dir_owned: []u8 = &.{};
    defer a.free(dir_owned);
    var bad_names: [6]rows.chain.Name = undefined;
    {
        var f: fake.Fake = .{ .gpa = a, .prompt = &tokens, .at = 100, .images = &valid_img };
        var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, opts(root, true));
        defer {
            s.deinit();
            r.deinit();
        }
        dir_owned = try a.dupe(u8, r.dir);
        try t.expect(s.keepBornFrame(&tokens, 100, 100, null, &.{}, &.{}, &valid_img, 0));
        drainAll(r);
        try t.expectEqual(@as(usize, 1), r.states.items.len);
        const valid_id = [_]rows.images.Identity{.{ .at = 40, .tokens = 4, .gh = 2, .gw = 2, .hash = 77 }};
        const enc_ok = try rows.images.encode(a, &valid_id, 0);
        defer a.free(enc_ok);
        const m_ok = try buildMeta(tokens[0..101], enc_ok);
        defer a.free(m_ok);
        // 1. Structurally valid metadata whose filename hash disagrees with chain.stateImages.
        bad_names[0] = @splat(0x11);
        try writeRawState(r.dir, bad_names[0], 100, rows.block_tokens, rows.images.flag, m_ok);
        // 2. Oversized trailer.meta (> maxMetaBytes) rejected before file.meta allocates.
        const too_big = try a.alloc(u8, @intCast(rows.images.maxMetaBytes(100, 1, rows.images.flag) + 64));
        defer a.free(too_big);
        @memset(too_big, 0);
        bad_names[1] = @splat(0x22);
        try writeRawState(r.dir, bad_names[1], 100, rows.block_tokens, rows.images.flag, too_big);
        // 3. Unknown trailer flag bit rejected before file.meta allocates.
        bad_names[2] = @splat(0x33);
        try writeRawState(r.dir, bad_names[2], 100, rows.block_tokens, 4, m_ok);
        // 4. Zero trailer.at rejected before file.meta allocates.
        bad_names[3] = @splat(0x44);
        try writeRawState(r.dir, bad_names[3], 0, rows.block_tokens, rows.images.flag, m_ok);
        // 5. Wrong trailer.block rejected before file.meta allocates.
        bad_names[4] = @splat(0x55);
        try writeRawState(r.dir, bad_names[4], 100, rows.block_tokens / 2, rows.images.flag, m_ok);
        // 6. Matching SHA-256 state name with an unattributed rogue high-bit token rejected by parseMeta.
        var rogue_toks = tokens;
        rogue_toks[5] |= 0x8000_0000;
        const m_rogue = try buildMeta(rogue_toks[0..101], enc_ok);
        defer a.free(m_rogue);
        bad_names[5] = rows.chain.stateImages(r.keying, rogue_toks[0..101], &.{}, &.{}, &valid_id, 0, 100);
        try writeRawState(r.dir, bad_names[5], 100, rows.block_tokens, rows.images.flag, m_rogue);
    }
    var f: fake.Fake = .{ .gpa = a };
    var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
    const r = try rows.attach(a, &s, opts(root, true));
    defer {
        s.deinit();
        r.deinit();
    }
    try t.expectEqual(@as(usize, 1), r.states.items.len);
    for (bad_names) |name| {
        var pbuf: [1200]u8 = undefined;
        const path = try rows.file.path(&pbuf, dir_owned, &rows.chain.hex(name), ".st", false);
        try t.expectError(error.RowsMissing, rows.file.open(path, .state, null));
    }
    try t.expectEqual(@as(u32, 100), try lookupAndRestore(&s, &f, &tokens, &valid_img, 0));
    try t.expect(f.restored_ok);
}

test "image disk: staging charge accounts for img_ids and stays within coldCharge for image-heavy states" {
    const Hold = struct {
        entered: std.atomic.Value(bool) = .init(false),
        release: std.atomic.Value(bool) = .init(false),
        fn write(ctx: ?*anyopaque, fd: c_int, bytes: []const u8) anyerror!usize {
            const h: *@This() = @ptrCast(@alignCast(ctx.?));
            h.entered.store(true, .release);
            while (!h.release.load(.acquire)) rows.write.io.nap();
            const n = platform.c.write(fd, bytes.ptr, bytes.len);
            if (n < 0) return error.RowsWrite;
            return @intCast(n);
        }
    };
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root_img = try std.fmt.allocPrint(a, "{s}/tf-img-charge-a-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root_img);
    defer rmTree(root_img);
    const root_txt = try std.fmt.allocPrint(a, "{s}/tf-img-charge-b-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root_txt);
    defer rmTree(root_txt);
    var plain_toks: [300]u32 = undefined;
    for (&plain_toks, 0..) |*w, i| w.* = @intCast(100 + i);
    var img_toks = plain_toks;
    var imgs: [64]Image = undefined;
    var ids: [64]rows.images.Identity = undefined;
    for (&imgs, &ids, 0..) |*im, *id, k| {
        const pos: u32 = @intCast(k * 4);
        img_toks[pos] = 0x8000_0200 + @as(u32, @intCast(k));
        im.* = .{ .pixels = &.{}, .gh = 1, .gw = 1, .at = pos, .tokens = 1, .hash = 1000 + k };
        id.* = .{ .at = pos, .tokens = 1, .gh = 1, .gw = 1, .hash = 1000 + k };
    }
    var staged_img: u64 = 0;
    var staged_plain: u64 = 0;
    {
        var f: fake.Fake = .{ .gpa = a, .prompt = &img_toks, .at = 256, .images = &imgs };
        var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, opts(root_img, false));
        defer {
            s.deinit();
            r.deinit();
        }
        const cold = try rows.write.coldCharge(r, 256);
        r.staging_bytes = cold;
        var hold: Hold = .{};
        defer hold.release.store(true, .release);
        r.file_io = .{ .ctx = &hold, .write = Hold.write };
        try t.expect(s.keepBornFrame(&img_toks, 256, 256, null, &.{}, &.{}, &imgs, 0));
        const dl_a = rows.write.nowNs() + 5 * std.time.ns_per_s;
        while (!hold.entered.load(.acquire) and rows.write.nowNs() < dl_a) {
            if (!rows.write.step(r)) rows.write.io.nap();
        }
        try t.expect(hold.entered.load(.acquire));
        staged_img = r.worker.?.bytes();
        try t.expectEqual(@as(u64, 0), r.counts.skipped);
        try t.expect(staged_img <= cold);
        hold.release.store(true, .release);
        drainAll(r);
        try t.expectEqual(@as(usize, 1), r.states.items.len);
    }
    {
        var f: fake.Fake = .{ .gpa = a, .prompt = &plain_toks, .at = 256 };
        var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
        const r = try rows.attach(a, &s, opts(root_txt, false));
        defer {
            s.deinit();
            r.deinit();
        }
        r.staging_bytes = try rows.write.coldCharge(r, 256);
        var hold: Hold = .{};
        defer hold.release.store(true, .release);
        r.file_io = .{ .ctx = &hold, .write = Hold.write };
        try t.expect(s.keepBornFrame(&plain_toks, 256, 256, null, &.{}, &.{}, &.{}, 0));
        const dl_b = rows.write.nowNs() + 5 * std.time.ns_per_s;
        while (!hold.entered.load(.acquire) and rows.write.nowNs() < dl_b) {
            if (!rows.write.step(r)) rows.write.io.nap();
        }
        try t.expect(hold.entered.load(.acquire));
        staged_plain = r.worker.?.bytes();
        hold.release.store(true, .release);
        drainAll(r);
    }
    const want_delta = rows.images.metaBytes(&ids, 0) + ids.len * @sizeOf(rows.images.Identity);
    try t.expectEqual(staged_plain + want_delta, staged_img);
}

test "image disk: salted image entries and unattributed reserved keys stay RAM-only" {
    const tmp = if (platform.c.getenv("TMPDIR")) |p| std.mem.span(p) else ".";
    const root = try std.fmt.allocPrint(a, "{s}/tf-img-salt-{d}", .{ tmp, platform.c.getpid() });
    defer a.free(root);
    defer rmTree(root);
    var tokens: [64]u32 = undefined;
    makePrompt(&tokens, 10, 2);
    const img = [_]Image{.{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 10, .tokens = 2, .hash = 55 }};
    var f: fake.Fake = .{ .gpa = a, .prompt = &tokens, .at = 20, .images = &img };
    var s = pc.Store.init(a, f.snapshots(), .{ .lookahead = 1, .min_prompt = 0 }, 1 << 30);
    const r = try rows.attach(a, &s, opts(root, true));
    defer {
        s.deinit();
        r.deinit();
    }
    // Salted image state stays RAM-only.
    s.salt = 0x5a17;
    try t.expect(s.keepBornFrame(&tokens, 20, 20, null, &.{}, &.{}, &img, 0));
    drainAll(r);
    try t.expectEqual(@as(usize, 0), r.states.items.len);
    // Unattributed reserved high-bit token (no images, no frame) stays RAM-only.
    s.salt = null;
    f.images = &.{};
    try t.expect(s.keepBornFrame(&tokens, 20, 20, null, &.{}, &.{}, &.{}, 0));
    drainAll(r);
    try t.expectEqual(@as(usize, 0), r.states.items.len);
    // Valid image alongside an unrelated rogue high-bit token also stays RAM-only.
    var rogue = tokens;
    rogue[3] |= 0x8000_0000;
    f.prompt = &rogue;
    f.images = &img;
    try t.expect(s.keepBornFrame(&rogue, 20, 20, null, &.{}, &.{}, &img, 0));
    drainAll(r);
    try t.expectEqual(@as(usize, 0), r.states.items.len);
}
