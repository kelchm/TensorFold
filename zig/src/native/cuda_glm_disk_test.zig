//! GLM-5.3-Flash disk staging floors (310K..1M, TP=2 tight budgets), async concurrent saves, and marks-mode reload.
const std = @import("std");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const glm_disk = @import("cuda_glm_disk.zig");
const img = glm.disk_image;
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;
const t = std.testing;

const page_bytes: u64 = 2 << 20;

const Gate = struct {
    entered: std.atomic.Value(bool) = .init(false),
    release: std.atomic.Value(bool) = .init(false),
    default_io: rows.file.IO = .{},

    fn io(g: *Gate) rows.file.IO {
        return .{ .ctx = g, .write = write, .sync = sync };
    }

    fn write(ctx: ?*anyopaque, fd: c_int, bytes: []const u8) !usize {
        const g: *Gate = @ptrCast(@alignCast(ctx.?));
        g.entered.store(true, .release);
        while (!g.release.load(.acquire)) std.Thread.yield() catch {};
        return g.default_io.write(null, fd, bytes);
    }

    fn sync(ctx: ?*anyopaque, fd: c_int) !void {
        const g: *Gate = @ptrCast(@alignCast(ctx.?));
        return g.default_io.sync(null, fd);
    }

    fn waitEntered(g: *const Gate) !void {
        var n: usize = 0;
        while (!g.entered.load(.acquire)) : (n += 1) {
            if (n > 50_000_000) return error.GateTimeout;
            std.Thread.yield() catch {};
        }
    }
};

test "CUDA disk open: GLM's store takes the shared attach's staging floor, so a 125K-token state is staged, not StagingFull" {
    const gpa = t.allocator;
    const T = struct {
        const token = 18_944; // about 18.5 KiB a token, as a served state's files measure (683 MiB at 36,864 tokens)
        fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
            try out.append(a, .{ .bytes = @as(u64, at) * token, .row = token });
        }
        fn bytes(_: *anyopaque, _: u32) u64 {
            return 0;
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
    };
    const sized: rows.Image = .{ .parts = T.parts, .gather = glm_disk.image.gather, .blank = glm_disk.image.blank, .scatter = glm_disk.image.scatter, .whole = glm_disk.image.whole };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
    defer gpa.free(root);
    var family: u8 = 0;
    const vt: api.prompt_cache.Snapshots.VTable = .{ .bytes = T.bytes, .save = T.save, .restore = T.restore, .drop = T.drop, .rows = &sized };
    const default = (rows.Options{ .root = "", .identity = 0, .cap = 0, .floor = 0, .daily = 0 }).staging_bytes;
    const window = 131_072;
    // a RAM store that keeps the state, and one too small for it: the floor is the state's charge within that budget
    for ([_]u64{ 8 << 30, 1 << 29 }) |budget| {
        var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &family, .vtable = &vt }, .{ .lookahead = 1 }, budget);
        const r = glm_disk.open(&store, .{ .dir = "d", .model_type = "glm", .learn = root, .learn_conversations = true }, root, 7, window) catch |e| {
            store.deinit();
            return e;
        };
        defer r.deinit();
        defer store.deinit(); // the store before its disk, as the rows fixtures close them
        try t.expectEqual(r, store.disk.?);
        try t.expectEqual(rows.Spill.off, r.spill);
        const long = try rows.write.coldCharge(r, 124_928); // the conversation the default skipped (StagingFull)
        try t.expect(long > default);
        if (budget > long) {
            try t.expect(r.staging_bytes >= long); // the permit a save's stage step asks `fits` of
            try t.expectEqual(try rows.write.coldCharge(r, window), r.staging_bytes); // the longest, whole
        } else try t.expectEqual(default, r.staging_bytes);
    }
}

test "CUDA disk open: GLM TP=2 staging floor covers 310K..1M and prevents StagingFull when TP prefixes exceed RAM budget" {
    const gpa = t.allocator;
    const Model = struct {
        sh: img.Shape,
        prefixes: [2]u64,
        fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
            const m: *const @This() = @ptrCast(@alignCast(ptr));
            var list: std.ArrayList(img.Part) = .empty;
            defer list.deinit(a);
            try img.storeParts(m.sh, &m.prefixes, at, &list, a);
            for (list.items) |p| try out.append(a, .{ .bytes = p.bytes, .row = p.row, .per = p.per });
        }
        fn bytes(ptr: *anyopaque, at: u32) u64 {
            const m: *const @This() = @ptrCast(@alignCast(ptr));
            const raw = img.imageBytes(m.sh, at) - img.Words.bytes;
            return std.mem.alignForward(u64, raw, page_bytes);
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
            return error.Unused;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: api.prompt_cache.Saved) void {}
    };
    const img_vt: rows.Image = .{ .parts = Model.parts, .gather = glm_disk.image.gather, .blank = glm_disk.image.blank, .scatter = glm_disk.image.scatter, .whole = glm_disk.image.whole };
    const vt: api.prompt_cache.Snapshots.VTable = .{ .bytes = Model.bytes, .save = Model.save, .restore = Model.restore, .drop = Model.drop, .rows = &img_vt };
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
    defer gpa.free(root);

    // GLM-5.3-Flash TP=2 FP8 KV: 11 MLA layers, kv_lora=512, head=false, ring_rows=8192, capacity=1,048,640.
    const prefix: u64 = 217_182_464;
    var fp8: Model = .{
        .sh = .{ .prefix = prefix, .tail = 8448, .layers = 11, .latent = 512, .scales = true, .ring_rows = 8192, .ring_row = 256, .pk_rows = 1_048_640 / 4 + 2, .pooled_row = 256, .capacity = 1_048_640, .head = false, .window = prefix - 16_384, .draft = 4 },
        .prefixes = .{ prefix, prefix },
    };
    const longest_1m = img.longest(fp8.sh);
    {
        var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &fp8, .vtable = &vt }, .{ .lookahead = 1 }, 16 << 30);
        const r = try glm_disk.open(&store, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true }, root, 101, longest_1m);
        defer r.deinit();
        defer store.deinit();
        const full = try rows.write.coldCharge(r, longest_1m);
        try t.expectEqual(full, r.staging_bytes);
        try t.expectEqual(@as(u64, 6914), r.staging_bytes >> 20);
        for ([_]u32{ 310_000, 481_000, 768_000, longest_1m }) |at| {
            try t.expect(try rows.write.coldCharge(r, at) <= r.staging_bytes);
        }
    }
    // Tight 4 GiB RAM budget at TP=2 FP8: 600K fits per-rank RAM (<= 4 GiB) while disk coldCharge > 4 GiB.
    {
        const budget: u64 = 4 << 30;
        var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &fp8, .vtable = &vt }, .{ .lookahead = 1 }, budget);
        const r = try glm_disk.open(&store, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true }, root, 102, longest_1m);
        defer r.deinit();
        defer store.deinit();
        const at: u32 = 600_000;
        try t.expect(Model.bytes(&fp8, at) <= budget);
        const cold = try rows.write.coldCharge(r, at);
        try t.expect(cold > budget); // old rows.attach clamp to store.budget would fail StagingFull here
        try t.expect(r.staging_bytes >= cold);
    }
    // Tight 6 GiB RAM budget at TP=2 BF16: 510K fits per-rank RAM while TP=2 disk coldCharge exceeds 6 GiB.
    var bf16: Model = .{
        .sh = .{ .prefix = prefix, .tail = 8448, .layers = 11, .latent = 1024, .scales = false, .ring_rows = 8192, .ring_row = 256, .pk_rows = 524_288 / 4 + 2, .pooled_row = 256, .capacity = 524_288, .head = false, .window = prefix - 16_384, .draft = 4 },
        .prefixes = .{ prefix, prefix },
    };
    const longest_512k = img.longest(bf16.sh);
    {
        const budget: u64 = 6 << 30;
        var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &bf16, .vtable = &vt }, .{ .lookahead = 1 }, budget);
        const r = try glm_disk.open(&store, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true }, root, 103, longest_512k);
        defer r.deinit();
        defer store.deinit();
        const at: u32 = 510_000;
        try t.expect(Model.bytes(&bf16, at) <= budget);
        const cold = try rows.write.coldCharge(r, at);
        try t.expect(cold > budget);
        try t.expect(r.staging_bytes >= cold);
    }
}

/// Proportionally scaled TP=2 GLM image fixture for fast live-worker async staging tests.
const AsyncFixture = struct {
    const prefix: u64 = 16 * 1024;
    const row_bytes: u32 = 16;
    const block_rows: u32 = 256;

    const Snap = struct {
        at: u32,
        tag: u8,
        data: []u8,
    };

    gpa: Allocator,
    tmp: t.TmpDir,
    root: []u8,
    restored_ok: bool = false,

    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
        try out.append(a, .{ .bytes = 2 * prefix, .row = 0 }); // TP=2 rank prefixes (state's own)
        try out.append(a, .{ .bytes = @as(u64, at) * row_bytes, .row = row_bytes, .per = 1 });
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return prefix + @as(u64, at) * row_bytes; // single-rank RAM size (missing rank 1's prefix + disk metadata!)
    }

    fn gather(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
        const s: *const Snap = @ptrCast(@alignCast(saved));
        var o: usize = 0;
        for (ranges) |g| {
            const n: usize = @intCast(g.len);
            @memcpy(out[o..][0..n], s.data[@intCast(g.off)..][0..n]);
            o += n;
        }
    }

    fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        const f: *AsyncFixture = @ptrCast(@alignCast(ptr));
        const s = try f.gpa.create(Snap);
        errdefer f.gpa.destroy(s);
        s.* = .{ .at = at, .tag = 0, .data = try f.gpa.alloc(u8, @intCast(2 * prefix + @as(u64, at) * row_bytes)) };
        @memset(s.data, 0);
        return @ptrCast(s);
    }

    fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
        const s: *Snap = @ptrCast(@alignCast(saved));
        var o: usize = 0;
        for (ranges) |g| {
            const n: usize = @intCast(g.len);
            @memcpy(s.data[@intCast(g.off)..][0..n], in[o..][0..n]);
            o += n;
        }
    }

    fn whole(_: *anyopaque, _: api.prompt_cache.Saved) bool {
        return true;
    }

    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
        return error.Unused;
    }

    fn restore(ptr: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {
        const f: *AsyncFixture = @ptrCast(@alignCast(ptr));
        f.restored_ok = true;
    }

    fn drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const f: *AsyncFixture = @ptrCast(@alignCast(ptr));
        const s: *Snap = @ptrCast(@alignCast(saved));
        f.gpa.free(s.data);
        f.gpa.destroy(s);
    }

    fn makeSnap(f: *AsyncFixture, at: u32, tag: u8) !*Snap {
        const s = try f.gpa.create(Snap);
        errdefer f.gpa.destroy(s);
        const data = try f.gpa.alloc(u8, @intCast(2 * prefix + @as(u64, at) * row_bytes));
        for (data, 0..) |*b, i| b.* = @truncate(i *% 13 +% tag);
        s.* = .{ .at = at, .tag = tag, .data = data };
        return s;
    }

    const img_vt: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole };
    const snap_vt: api.prompt_cache.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &img_vt };
};

fn makePrompt(a: Allocator, len: usize, base: u32) ![]u32 {
    const p = try a.alloc(u32, len);
    for (p, 0..) |*x, i| x.* = base +% @as(u32, @intCast(i));
    return p;
}

test "CUDA disk open: back-to-back extending saves deduplicate blocks and independent concurrent saves wait on StagingBusy" {
    const gpa = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
    defer gpa.free(root);
    var fx: AsyncFixture = .{ .gpa = gpa, .tmp = tmp, .root = root };

    // Longest=16,384 tokens (~300 KiB payload); budget fits 16,384 in RAM and staging_bytes takes 1 cold 16K save.
    const longest: u32 = 16_384;
    var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &AsyncFixture.snap_vt }, .{ .lookahead = 1 }, 64 << 20);
    const r = try glm_disk.open(&store, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 201, longest);
    defer r.deinit();
    defer store.deinit();
    // Shrink permit to exactly 1 cold 12K save so 8K+12K exceeds it together (testing real StagingBusy!).
    const c_12k = try rows.write.coldCharge(r, 12_288);
    r.staging_bytes = c_12k;

    var gate: Gate = .{};
    defer gate.release.store(true, .release);
    r.file_io = gate.io();

    // 1) Extending conversation (8K -> 8.5K while 8K is held): deduplicates promised blocks and stages immediately.
    const p1 = try makePrompt(gpa, 9_000, 100);
    defer gpa.free(p1);
    const s1 = try fx.makeSnap(8_192, 1);
    defer AsyncFixture.drop(&fx, @ptrCast(s1));
    const s2 = try fx.makeSnap(8_704, 1);
    defer AsyncFixture.drop(&fx, @ptrCast(s2));
    var e1: api.prompt_cache.Entry = .{ .tokens = p1[0..8_193], .at = 8_192, .saved = @ptrCast(s1), .bytes = 1, .born = 9_000, .used = 1, .last = p1 };
    var e2: api.prompt_cache.Entry = .{ .tokens = p1[0..8_705], .at = 8_704, .saved = @ptrCast(s2), .bytes = 1, .born = 9_000, .used = 2, .last = p1 };
    rows.write.queued(r, &e1, &.{});
    try t.expect(rows.write.step(r));
    try gate.waitEntered();
    try t.expectEqual(@as(usize, 1), r.worker.?.pending());
    rows.write.queued(r, &e2, &.{});
    try t.expect(rows.write.step(r)); // deduplicated against promised blocks: stages in parallel!
    try t.expectEqual(@as(usize, 2), r.worker.?.pending());
    try t.expect(r.waiting == null);
    try t.expectEqual(@as(u64, 0), r.counts.skipped);
    gate.release.store(true, .release);
    try t.expectEqual(@as(usize, 0), rows.write.drain(r, rows.write.nowNs() + 10 * std.time.ns_per_s));
    try t.expectEqual(@as(u64, 2), r.counts.writes);
    try t.expectEqual(@as(u64, 0), r.counts.skipped);
    try t.expectEqual(@as(u64, 0), r.counts.failed);

    // 2) Two independent cold saves exceeding the permit together (8K + 12K): 12K hits StagingBusy, waits, publishes.
    var store2 = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &AsyncFixture.snap_vt }, .{ .lookahead = 1 }, 64 << 20);
    const r2 = try glm_disk.open(&store2, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 202, longest);
    defer r2.deinit();
    defer store2.deinit();
    var gate2: Gate = .{};
    defer gate2.release.store(true, .release);
    r2.file_io = gate2.io();
    const pa = try makePrompt(gpa, 9_000, 100_000);
    defer gpa.free(pa);
    const pb = try makePrompt(gpa, 13_000, 500_000);
    defer gpa.free(pb);
    const sa = try fx.makeSnap(8_192, 2);
    defer AsyncFixture.drop(&fx, @ptrCast(sa));
    const sb = try fx.makeSnap(12_288, 3);
    defer AsyncFixture.drop(&fx, @ptrCast(sb));
    var ea: api.prompt_cache.Entry = .{ .tokens = pa[0..8_193], .at = 8_192, .saved = @ptrCast(sa), .bytes = 1, .born = 9_000, .used = 1, .last = pa };
    var eb: api.prompt_cache.Entry = .{ .tokens = pb[0..12_289], .at = 12_288, .saved = @ptrCast(sb), .bytes = 1, .born = 13_000, .used = 2, .last = pb };
    rows.write.queued(r2, &ea, &.{});
    try t.expect(rows.write.step(r2));
    try gate2.waitEntered();
    // Bound worker limit to 1.5x the in-flight 8K job (~320 KiB): 12K fits alone (~310 KiB) but not alongside 8K!
    const used_a = r2.worker.?.bytes();
    r2.worker.?.limit = used_a + used_a / 2;
    r2.staging_bytes = r2.worker.?.limit;
    rows.write.queued(r2, &eb, &.{});
    try t.expect(!rows.write.step(r2)); // StagingBusy: held in queue, NOT skipped as StagingFull!
    try t.expectEqual(&eb, r2.waiting.?);
    try t.expectEqual(@as(usize, 1), r2.queue.items.len);
    try t.expectEqual(@as(u64, 0), r2.counts.skipped);
    gate2.release.store(true, .release);
    try t.expectEqual(@as(usize, 0), rows.write.drain(r2, rows.write.nowNs() + 10 * std.time.ns_per_s));
    try t.expectEqual(@as(u64, 2), r2.counts.writes);
    try t.expectEqual(@as(u64, 0), r2.counts.skipped);
    try t.expectEqual(@as(u64, 0), r2.counts.failed);

    // 3) Tight RAM budget where bytes(12K) <= budget < coldCharge(12K): open() raises staging_bytes so 12K writes!
    const ram_12k = AsyncFixture.bytes(&fx, 12_288);
    var store3 = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &AsyncFixture.snap_vt }, .{ .lookahead = 1 }, ram_12k);
    const r3 = try glm_disk.open(&store3, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 203, longest);
    defer r3.deinit();
    defer store3.deinit();
    try t.expect(try rows.write.coldCharge(r3, 12_288) > ram_12k);
    try t.expect(r3.staging_bytes >= try rows.write.coldCharge(r3, 12_288));
    rows.write.queued(r3, &eb, &.{});
    try t.expectEqual(@as(usize, 0), rows.write.drain(r3, rows.write.nowNs() + 10 * std.time.ns_per_s));
    try t.expectEqual(@as(u64, 1), r3.counts.writes);
    try t.expectEqual(@as(u64, 0), r3.counts.skipped);
}

test "CUDA disk open: TF_GLM_REQUEST_STARTS=marks off-grid state persists, recalls after restart, and resumes" {
    const gpa = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
    defer gpa.free(root);
    var fx: AsyncFixture = .{ .gpa = gpa, .tmp = tmp, .root = root };

    const prompt = try makePrompt(gpa, 1600, 42);
    defer gpa.free(prompt);
    const at: u32 = 1027;
    const chunks = [_]u32{ 300, 650, at, 1500 };

    // Turn 1 save (marks mode saved state at 1027 with engine below = 0 while server chunks have 300, 650, 1027).
    {
        const s = try fx.makeSnap(at, 9);
        defer AsyncFixture.drop(&fx, @ptrCast(s));
        const r = try rows.Rows.open(gpa, .{ .ptr = &fx, .vtable = &AsyncFixture.snap_vt }, .{ .root = root, .identity = 303, .cap = 1 << 26, .floor = 0, .daily = 1 << 26, .lookahead = 1, .planned = true, .grid = 2048, .sync_writes = true });
        defer r.deinit();
        var entry: api.prompt_cache.Entry = .{ .tokens = prompt[0 .. at + 1], .at = at, .saved = @ptrCast(s), .bytes = 1, .born = 1600, .used = 1, .last = prompt };
        rows.write.queued(r, &entry, &chunks);
        while (rows.write.pending(r)) _ = rows.write.step(r);
        try t.expectEqual(@as(usize, 1), r.states.items.len);
    }
    // Simulated restart: reopen store from disk, recall for Turn 2, and check Lanes.resumeFrom in marks vs all mode.
    const r = try rows.Rows.open(gpa, .{ .ptr = &fx, .vtable = &AsyncFixture.snap_vt }, .{ .root = root, .identity = 303, .cap = 1 << 26, .floor = 0, .daily = 1 << 26, .lookahead = 1, .planned = true, .grid = 2048, .sync_writes = true });
    defer r.deinit();
    const idx = r.best(prompt, &chunks, &.{}, 0) orelse return error.DiskStateMissing;
    const restored_saved = try rows.read.read(r, idx);
    defer AsyncFixture.drop(&fx, restored_saved);
    var e: glm.engine.Engine = undefined;
    e.gpa = gpa;
    e.snaps = .empty;
    defer e.snaps.deinit(gpa);
    try e.snaps.put(gpa, 19, undefined);
    var k: glm.engine.Kept = .{ .id = 19, .at = at, .below = 0 };
    var s = try lanes.Stream.init(gpa, .{ .id = "reload", .prompt = prompt, .max_new = 16, .drafts = false, .chunks = &chunks, .reuse = .{ .saved = &k, .at = at } });
    defer s.deinit(gpa);
    var l_all: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined }, .request_starts = .all };
    try t.expectEqual(@as(?glm.engine.Resume, null), l_all.resumeFrom(&s, prompt.len));
    try t.expectEqual(@as(?anyerror, error.SnapshotOutOfStep), s.reuse_error);
    s.reuse_failed = false;
    s.reuse_error = null;
    var l_marks: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = undefined }, .request_starts = .marks };
    const resumed = l_marks.resumeFrom(&s, prompt.len) orelse return error.ExpectedMarksResume;
    try t.expectEqual(@as(usize, at), resumed.at);
    try t.expectEqual(at, s.cached);
}
