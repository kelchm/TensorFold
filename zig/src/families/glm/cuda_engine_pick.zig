//! The rank-gathered greedy pick, a row at a time or a window's rows at once, kept apart from engine lifetime.
const cuda = @import("cuda");
const std = @import("std");
const ranks = @import("cuda_ranks.zig");
const chance = @import("cuda_mtp_chance.zig");
const engine = @import("cuda_engine.zig");
const fw = @import("cuda_forward.zig");
const Engine = engine.Engine;
const pair_bytes = 16;
/// The rows one gather of greedy picks holds (Engine.pairs, a rank's): a shared window's.
pub const pick_rows = engine.shared_rows;

pub fn greedy(e: *Engine, logits: u64) !u32 {
    return (try pick(e, logits, false)).token;
}

/// greedy with the pick's chance from every rank's softmax sum (tf_glm_top1_sums), for --mtp-confidence's drafts.
pub fn greedyChance(e: *Engine, logits: u64) !chance.Pick {
    return pick(e, logits, true);
}

/// `rows` rows' greedy tokens: one top-1 launch, gather, sync and read (TF_GLM_GREEDY_ROWS), else `greedy` a row.
pub fn greedyRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
    const V = e.share.vocab.len();
    if (rows < 2 or !e.settings.greedy_rows) {
        for (0..rows) |r| out[r] = try greedy(e, logits + r * V * 2);
        return;
    }
    if (rows > pick_rows) return error.WindowTooWide;
    const world = e.share.world;
    const mine = e.pairs.ptr + e.share.rank * rows * pair_bytes;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(V)));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(mine);
    try cuda.launch.launch(e.m.glue.top1_rows, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, e.stream, &a);
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, rows * pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, rows * pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world * pick_rows][4]u32 = undefined; // Engine.init refuses a larger world
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.sliceAsBytes(all[0 .. world * rows]));
    for (out[0..rows], 0..) |*o, r| o.* = rowPick(all[0 .. world * rows], world, rows, r);
}

/// Row `r`'s token from one gather of `rows` rows' words in rank order ([world][rows]): `greedy`'s pick of that row.
pub fn rowPick(all: []const [4]u32, world: usize, rows: usize, r: usize) u32 {
    var words: [ranks.max_world][4]u32 = undefined;
    for (words[0..world], 0..) |*w, q| w.* = all[q * rows + r];
    return chance.pick(words[0..world], false).token;
}

/// A shared window's rows in one greedyRows call: the switch on, no grammar, no sampled part (alike on every rank).
pub fn oneDraw(e: *const Engine, parts: []const engine.Part) bool {
    if (!e.settings.greedy_rows or e.structured) return false;
    for (parts) |p| if ((e.seqs.items[p.slot] orelse return false).sampling != null) return false;
    return true;
}

/// Engine.verifyShared's draws under `oneDraw`: each part bound with its rows_at, then every row in one greedyRows.
pub fn drawShared(e: *Engine, b: *const fw.Buffers, parts: []const engine.Part, segs: []const fw.Seg) !void {
    var outs: [pick_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, segs) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        rows = g.row0 + g.rows;
    }
    if (rows > pick_rows) return error.WindowTooWide;
    try greedyRows(e, b.logits, rows, outs[0..rows]);
    for (parts, segs) |p, g| @memcpy(p.out[0..g.rows], outs[g.row0..][0..g.rows]);
}

fn pick(e: *Engine, logits: u64, sums: bool) !chance.Pick {
    const world = e.share.world;
    var a: cuda.Args = .{};
    a.add(logits);
    a.add(@as(i64, @intCast(e.share.vocab.len())));
    a.add(@as(u32, @intCast(e.share.vocab.lo)));
    a.add(e.pairs.ptr + e.share.rank * pair_bytes);
    try cuda.launch.launch(if (sums) e.m.glue.top1_sums else e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
    const mine = e.pairs.ptr + e.share.rank * pair_bytes;
    const stamp = @import("cuda_prof.zig").span(e.profiling_prompt, e.stream, .token_gather);
    if (e.group) |g| try g.allGather(e.stream, mine, e.pairs.ptr, pair_bytes) else try e.lib.check(e.lib.api.ncclAllGather(mine, e.pairs.ptr, pair_bytes, .u8, e.comm, e.stream.handle), "ncclAllGather");
    stamp.end();
    try e.stream.synchronize();
    try e.checkGathers();
    var all: [ranks.max_world][4]u32 = undefined; // Engine.init refuses a larger world
    try cuda.memory.download(e.d, e.pairs.ptr, std.mem.asBytes(&all)[0 .. world * pair_bytes]);
    return chance.pick(all[0..world], sums);
}
