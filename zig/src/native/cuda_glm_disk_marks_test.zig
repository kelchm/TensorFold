//! Disk block-dedup cut-history isolation, multi-turn dedup, and real-handle restart reload for marks mode.
const std = @import("std");
const api = @import("engine_api");
const glm = @import("glm_cuda");
const lanes = @import("lanes");
const glm_disk = @import("cuda_glm_disk.zig");
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;
const t = std.testing;

/// Real GLM Kept handles + initialized Engine.snaps + per-state byte buffers for async rows_store tests.
const MarksFixture = struct {
    const prefix: u64 = 16; // Words-like header: [0..8) = below digest, [8..16) = at
    const row_bytes: u32 = 16;

    gpa: Allocator,
    e: glm.engine.Engine,
    next_id: u64 = 1,
    bufs: std.AutoHashMapUnmanaged(u64, []u8) = .empty,

    fn init(gpa: Allocator) MarksFixture {
        var e: glm.engine.Engine = undefined;
        e.gpa = gpa;
        e.snaps = .empty;
        return .{ .gpa = gpa, .e = e };
    }

    fn deinit(f: *MarksFixture) void {
        var it = f.bufs.valueIterator();
        while (it.next()) |b| f.gpa.free(b.*);
        f.bufs.deinit(f.gpa);
        f.e.snaps.deinit(f.gpa);
    }

    fn imgLen(at: u32) usize {
        return @intCast(prefix + @as(u64, at) * row_bytes);
    }

    fn createKept(f: *MarksFixture, at: u32, executed: []const u32, fill: *const fn (usize) u8) !*glm.engine.Kept {
        const id = f.next_id;
        f.next_id += 1;
        const exec_dup = try f.gpa.dupe(u32, executed);
        errdefer f.gpa.free(exec_dup);
        const below = glm.engine.prompt_cuts.below(exec_dup, at);
        const data = try f.gpa.alloc(u8, imgLen(at));
        errdefer f.gpa.free(data);
        std.mem.writeInt(u64, data[0..8], below, .little);
        std.mem.writeInt(u64, data[8..16], at, .little);
        for (0..at) |row| @memset(data[prefix + row * row_bytes ..][0..row_bytes], fill(row));
        try f.bufs.put(f.gpa, id, data);
        errdefer _ = f.bufs.remove(id);
        try f.e.snaps.put(f.gpa, id, .{ .lease = .{}, .len = data.len, .pos = at, .mtp_len = 0, .cur = &.{}, .bases = .{ 0, 0 }, .below = below });
        errdefer _ = f.e.snaps.remove(id);
        const k = try f.gpa.create(glm.engine.Kept);
        k.* = .{ .id = id, .at = at, .below = below, .executed = exec_dup, .has_executed = true };
        return k;
    }

    fn parts(_: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
        try out.append(a, .{ .bytes = prefix, .row = 0 });
        try out.append(a, .{ .bytes = @as(u64, at) * row_bytes, .row = row_bytes, .per = 1 });
    }

    fn bytes(_: *anyopaque, at: u32) u64 {
        return prefix + @as(u64, at) * row_bytes;
    }

    fn gather(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
        const f: *MarksFixture = @ptrCast(@alignCast(ptr));
        const k: *const glm.engine.Kept = @ptrCast(@alignCast(saved));
        const data = f.bufs.get(k.id) orelse return error.NoKeptState;
        var o: usize = 0;
        for (ranges) |g| {
            const n: usize = @intCast(g.len);
            @memcpy(out[o..][0..n], data[@intCast(g.off)..][0..n]);
            o += n;
        }
    }

    fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        const f: *MarksFixture = @ptrCast(@alignCast(ptr));
        const id = f.next_id;
        f.next_id += 1;
        const data = try f.gpa.alloc(u8, imgLen(at));
        errdefer f.gpa.free(data);
        @memset(data, 0);
        try f.bufs.put(f.gpa, id, data);
        errdefer _ = f.bufs.remove(id);
        try f.e.snaps.put(f.gpa, id, .{ .lease = .{}, .len = 0, .pos = at, .mtp_len = 0, .cur = &.{}, .bases = .{ 0, 0 }, .below = 0 });
        errdefer _ = f.e.snaps.remove(id);
        const k = try f.gpa.create(glm.engine.Kept);
        k.* = .{ .id = id, .at = at };
        return @ptrCast(k);
    }

    fn scatter(ptr: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
        const f: *MarksFixture = @ptrCast(@alignCast(ptr));
        const k: *glm.engine.Kept = @ptrCast(@alignCast(saved));
        const data = f.bufs.get(k.id) orelse return error.NoKeptState;
        var o: usize = 0;
        for (ranges) |g| {
            const n: usize = @intCast(g.len);
            @memcpy(data[@intCast(g.off)..][0..n], in[o..][0..n]);
            o += n;
        }
        if (ranges.len > 0 and ranges[0].off == 0) {
            const below = std.mem.readInt(u64, data[0..8], .little);
            k.below = below;
            if (f.e.snaps.getPtr(k.id)) |s| {
                s.len = data.len;
                s.below = below;
            }
        }
    }

    fn whole(ptr: *anyopaque, saved: api.prompt_cache.Saved) bool {
        const f: *MarksFixture = @ptrCast(@alignCast(ptr));
        const k: *const glm.engine.Kept = @ptrCast(@alignCast(saved));
        const s = f.e.snaps.get(k.id) orelse return false;
        return s.len != 0;
    }

    fn startsOf(_: *anyopaque, saved: api.prompt_cache.Saved) ?[]const u32 {
        const k: *const glm.engine.Kept = @ptrCast(@alignCast(saved));
        return if (k.has_executed) k.executed else null;
    }

    fn setStarts(ptr: *anyopaque, saved: api.prompt_cache.Saved, executed: []const u32) anyerror!void {
        const f: *MarksFixture = @ptrCast(@alignCast(ptr));
        const k: *glm.engine.Kept = @ptrCast(@alignCast(saved));
        const dup = try f.gpa.dupe(u32, executed);
        f.gpa.free(k.executed);
        k.executed = dup;
        k.has_executed = true;
    }

    fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!api.prompt_cache.Saved {
        return error.Unused;
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: api.prompt_cache.Saved) anyerror!void {}

    fn drop(ptr: *anyopaque, saved: api.prompt_cache.Saved) void {
        const f: *MarksFixture = @ptrCast(@alignCast(ptr));
        const k: *glm.engine.Kept = @ptrCast(@alignCast(saved));
        if (f.bufs.fetchRemove(k.id)) |kv| f.gpa.free(kv.value);
        _ = f.e.snaps.remove(k.id);
        f.gpa.free(k.executed);
        f.gpa.destroy(k);
    }

    const img_vt: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter, .whole = whole, .starts = startsOf, .set_starts = setStarts };
    const snap_vt: api.prompt_cache.Snapshots.VTable = .{ .bytes = bytes, .save = save, .restore = restore, .drop = drop, .rows = &img_vt };
};

fn settle(r: *rows.Rows) !void {
    const deadline = rows.write.nowNs() + 10 * std.time.ns_per_s;
    while (rows.write.pending(r)) {
        if (rows.write.nowNs() >= deadline) return error.SettleTimeout;
        if (!rows.write.step(r)) rows.write.io.nap();
    }
    rows.write.reap(r);
}

fn makePrompt(a: Allocator, len: usize, base: u32) ![]u32 {
    const p = try a.alloc(u32, len);
    for (p, 0..) |*x, i| x.* = base +% @as(u32, @intCast(i));
    return p;
}

test "CUDA disk open: marks mode isolates async block stems across cold 1027 vs 1500 under shared server chunks" {
    const gpa = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
    defer gpa.free(root);
    var fx = MarksFixture.init(gpa);
    defer fx.deinit();

    const prompt = try makePrompt(gpa, 1600, 42);
    defer gpa.free(prompt);
    const server_chunks = [_]u32{ 300, 650, 1027, 1500 };
    const FillA = struct {
        fn byte(row: usize) u8 {
            return if (row == 700) 0xA1 else @truncate(row +% 11);
        }
    };
    const FillB = struct {
        fn byte(row: usize) u8 {
            return if (row == 700) 0xB2 else @truncate(row +% 11);
        }
    };
    const k_a = try fx.createKept(1027, &.{}, FillA.byte);
    defer MarksFixture.drop(&fx, @ptrCast(k_a));
    const k_b = try fx.createKept(1500, &.{}, FillB.byte);
    defer MarksFixture.drop(&fx, @ptrCast(k_b));
    const orig_a = try gpa.dupe(u8, fx.bufs.get(k_a.id).?);
    defer gpa.free(orig_a);
    const orig_b = try gpa.dupe(u8, fx.bufs.get(k_b.id).?);
    defer gpa.free(orig_b);

    // Real async writer (sync_writes=false, mark_starts=true) with identical server chunks on both cold passes.
    {
        var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &MarksFixture.snap_vt }, .{ .lookahead = 1, .planned = true, .grid = 2048 }, 64 << 20);
        const r = try glm_disk.openWithFloor(&store, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 401, 2048, 64 << 10, true);
        defer r.deinit();
        defer store.deinit();
        try t.expect(r.mark_starts and !r.sync_writes);
        var e_a: api.prompt_cache.Entry = .{ .tokens = prompt[0..1028], .at = 1027, .saved = @ptrCast(k_a), .bytes = 1, .born = 1600, .used = 1, .last = prompt };
        var e_b: api.prompt_cache.Entry = .{ .tokens = prompt[0..1501], .at = 1500, .saved = @ptrCast(k_b), .bytes = 1, .born = 1600, .used = 2, .last = prompt };
        rows.write.queued(r, &e_a, &server_chunks);
        rows.write.queued(r, &e_b, &server_chunks);
        try t.expectEqual(@as(usize, 0), rows.write.drain(r, rows.write.nowNs() + 10 * std.time.ns_per_s));
        try t.expectEqual(@as(u64, 2), r.counts.writes);
    }

    // Reopen after simulated restart: both survive, block [512, 768) stems differ, and 1500 reads back 0xB2!
    var store2 = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &MarksFixture.snap_vt }, .{ .lookahead = 1, .planned = true, .grid = 2048 }, 64 << 20);
    const r2 = try glm_disk.openWithFloor(&store2, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 401, 2048, 64 << 10, true);
    defer r2.deinit();
    defer store2.deinit();
    try t.expectEqual(@as(usize, 2), r2.states.items.len);

    var idx_a: ?usize = null;
    var idx_b: ?usize = null;
    for (r2.states.items, 0..) |st, i| {
        if (st.at == 1027) idx_a = i;
        if (st.at == 1500) idx_b = i;
    }
    var wants_a: std.ArrayList(rows.write.Want) = .empty;
    defer wants_a.deinit(gpa);
    var wants_b: std.ArrayList(rows.write.Want) = .empty;
    defer wants_b.deinit(gpa);
    try r2.wantsOf(gpa, &r2.states.items[idx_a.?], &wants_a);
    try r2.wantsOf(gpa, &r2.states.items[idx_b.?], &wants_b);
    // Block k=2 covers rows [512, 768) containing row 700: stems MUST be distinct!
    try t.expect(!std.mem.eql(u8, &wants_a.items[2].name, &wants_b.items[2].name));

    const best_b = r2.best(prompt, &server_chunks, &.{}, 0) orelse return error.DiskStateMissing;
    try t.expectEqual(@as(u32, 1500), r2.states.items[best_b].at);
    const rest_b_saved = try rows.read.read(r2, best_b);
    defer MarksFixture.drop(&fx, rest_b_saved);
    const rest_b: *glm.engine.Kept = @ptrCast(@alignCast(rest_b_saved));
    try t.expect(rest_b.has_executed and rest_b.executed.len == 0);
    try t.expectEqualSlices(u8, orig_b, fx.bufs.get(rest_b.id).?);
    try t.expectEqual(@as(u8, 0xB2), fx.bufs.get(rest_b.id).?[MarksFixture.prefix + 700 * MarksFixture.row_bytes]);

    // Real restored_saved handle passes through Lanes.resumeFrom in marks mode (and is rejected in all mode).
    var s = try lanes.Stream.init(gpa, .{ .id = "reload", .prompt = prompt, .max_new = 16, .drafts = false, .chunks = &server_chunks, .reuse = .{ .saved = rest_b_saved, .at = 1500 } });
    defer s.deinit(gpa);
    var l_all: glm.lanes.Lanes = .{ .x = .{ .e = &fx.e, .link = undefined }, .request_starts = .all };
    try t.expectEqual(@as(?glm.engine.Resume, null), l_all.resumeFrom(&s, prompt.len));
    try t.expectEqual(@as(?anyerror, error.SnapshotOutOfStep), s.reuse_error);
    s.reuse_failed = false;
    s.reuse_error = null;
    var l_marks: glm.lanes.Lanes = .{ .x = .{ .e = &fx.e, .link = undefined }, .request_starts = .marks };
    const resumed = l_marks.resumeFrom(&s, prompt.len) orelse return error.ExpectedMarksResume;
    try t.expectEqual(@as(usize, 1500), resumed.at);
    try t.expectEqual(rest_b.id, resumed.id);
    try t.expectEqual(@as(u32, 1500), s.cached);

    // Reopening 1,027 reads back 0xA1 byte-for-byte intact.
    const rest_a_saved = try rows.read.read(r2, idx_a.?);
    defer MarksFixture.drop(&fx, rest_a_saved);
    const rest_a: *glm.engine.Kept = @ptrCast(@alignCast(rest_a_saved));
    try t.expectEqualSlices(u8, orig_a, fx.bufs.get(rest_a.id).?);
    try t.expectEqual(@as(u8, 0xA1), fx.bufs.get(rest_a.id).?[MarksFixture.prefix + 700 * MarksFixture.row_bytes]);
}

test "CUDA disk open: marks mode multi-turn dedup shares 100% of prefix blocks across turns and restart" {
    const gpa = t.allocator;
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", gpa);
    defer gpa.free(root);
    var fx = MarksFixture.init(gpa);
    defer fx.deinit();

    const prompt = try makePrompt(gpa, 5200, 100);
    defer gpa.free(prompt);
    const server_chunks = [_]u32{ 300, 650, 1200, 2500, 4100, 4650, 4900 };
    const Fill = struct {
        fn byte(row: usize) u8 {
            return @truncate(row *% 17 +% 3);
        }
    };

    const ex1 = try glm.engine.prompt_cuts.executedBelow(gpa, &.{}, 0, &.{4100}, 4100);
    defer gpa.free(ex1);
    try t.expectEqualSlices(u32, &.{ 2048, 4096 }, ex1);
    const ex2 = try glm.engine.prompt_cuts.executedBelow(gpa, ex1, 4100, &.{ 4100, 4650 }, 4650);
    defer gpa.free(ex2);
    try t.expectEqualSlices(u32, &.{ 2048, 4096, 4100 }, ex2);

    const k1 = try fx.createKept(4100, ex1, Fill.byte);
    defer MarksFixture.drop(&fx, @ptrCast(k1));
    const k2 = try fx.createKept(4650, ex2, Fill.byte);
    defer MarksFixture.drop(&fx, @ptrCast(k2));
    const orig_k2 = try gpa.dupe(u8, fx.bufs.get(k2.id).?);
    defer gpa.free(orig_k2);

    // Session 1: Turn 1 (4100 -> 16 blocks) + Turn 2 (4650 -> only 2 new blocks; all 16 in [0, 4096) deduplicated!).
    {
        var store = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &MarksFixture.snap_vt }, .{ .lookahead = 1, .planned = true, .grid = 2048 }, 64 << 20);
        const r = try glm_disk.openWithFloor(&store, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 402, 8192, 64 << 10, true);
        defer r.deinit();
        defer store.deinit();
        var e1: api.prompt_cache.Entry = .{ .tokens = prompt[0..4101], .at = 4100, .saved = @ptrCast(k1), .bytes = 1, .born = 5200, .used = 1, .last = prompt };
        rows.write.queued(r, &e1, &server_chunks);
        try settle(r);
        try t.expectEqual(@as(usize, 16), r.blocks.count()); // 4096 / 256 = 16 blocks

        var e2: api.prompt_cache.Entry = .{ .tokens = prompt[0..4651], .at = 4650, .saved = @ptrCast(k2), .bytes = 1, .born = 5200, .used = 2, .last = prompt };
        rows.write.queued(r, &e2, &server_chunks);
        try settle(r);
        try t.expectEqual(@as(usize, 18), r.blocks.count()); // +2 blocks ([4096,4352), [4352,4608)); [0,4096) 100% shared
    }

    // Session 2 (restart): restore Turn 2 via rows.read.read, verify set_starts, save Turn 3 (+1 block = 19) & Seen.id.
    var store2 = api.prompt_cache.Store.init(gpa, .{ .ptr = &fx, .vtable = &MarksFixture.snap_vt }, .{ .lookahead = 1, .planned = true, .grid = 2048 }, 64 << 20);
    const r2 = try glm_disk.openWithFloor(&store2, .{ .dir = "d", .model_type = "glm5_next", .learn = root, .learn_conversations = true, .learn_gib = 1, .learn_min_free_gib = 0, .learn_daily_gib = 1 }, root, 402, 8192, 64 << 10, true);
    defer r2.deinit();
    defer store2.deinit();
    try t.expectEqual(@as(usize, 18), r2.blocks.count());
    const idx2 = r2.best(prompt, &server_chunks, &.{}, 0) orelse return error.DiskStateMissing;
    try t.expectEqual(@as(u32, 4650), r2.states.items[idx2].at);
    const rest2_saved = try rows.read.read(r2, idx2);
    defer MarksFixture.drop(&fx, rest2_saved);
    const rest2: *glm.engine.Kept = @ptrCast(@alignCast(rest2_saved));
    try t.expect(rest2.has_executed);
    try t.expectEqualSlices(u32, &.{ 2048, 4096, 4100 }, rest2.executed);
    try t.expectEqualSlices(u8, orig_k2, fx.bufs.get(rest2.id).?);

    // Turn 3 inherits rest2.executed after restart.
    const ex3 = try glm.engine.prompt_cuts.executedBelow(gpa, rest2.executed, rest2.at, &.{ 4650, 4900 }, 4900);
    defer gpa.free(ex3);
    try t.expectEqualSlices(u32, &.{ 2048, 4096, 4100, 4650 }, ex3);
    const k3 = try fx.createKept(4900, ex3, Fill.byte);
    defer MarksFixture.drop(&fx, @ptrCast(k3));
    const prev_named = api.prompt_cache.status.named.swap(true, .acq_rel);
    defer api.prompt_cache.status.named.store(prev_named, .release);
    const seen3 = api.prompt_cache.status.Seen.ofSaved(&store2, @ptrCast(k3), prompt[0..4901], &.{}, &server_chunks, &.{}, 0, 4900);
    var e3: api.prompt_cache.Entry = .{ .tokens = prompt[0..4901], .at = 4900, .saved = @ptrCast(k3), .bytes = 1, .born = 5200, .used = 3, .last = prompt };
    rows.write.queued(r2, &e3, &server_chunks);
    try t.expectEqual(@as(usize, 0), rows.write.drain(r2, rows.write.nowNs() + 10 * std.time.ns_per_s));
    try t.expectEqual(@as(usize, 19), r2.blocks.count()); // only 1 new block [4608, 4864); [0, 4608) 100% deduplicated
    const idx3 = r2.best(prompt, &server_chunks, &.{}, 0) orelse return error.DiskStateMissing;
    try t.expectEqual(@as(u32, 4900), r2.states.items[idx3].at);
    try t.expectEqual(rows.chain.stem(r2.states.items[idx3].name), seen3.id);
}
