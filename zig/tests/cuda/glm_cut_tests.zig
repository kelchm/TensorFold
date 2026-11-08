//! GLM-5.3-Flash prompt passes cut at request starts (cuda_prompt_cuts) against the 2,048-row grid, on one Engine.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const tape_windows = 8; // teacher-forced verify windows after the prompt, each kept whole
const tape_rows = 4;

const Pattern = struct { name: []const u8, cuts: []const u32 };

/// Whether every cut is a multiple of 64 ("64"), of 4 ("4"), or neither ("any"); the grid's own run is "control".
fn class(cuts: []const u32) []const u8 {
    if (cuts.len == 0) return "control";
    for (cuts) |c| if (c % 4 != 0) return "any";
    for (cuts) |c| if (c % 64 != 0) return "4";
    return "64";
}

/// Cuts from `first` every `step` rows below `len`, at most `most` of them.
fn every(a: std.mem.Allocator, len: usize, first: usize, step: usize, most: usize) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    var c = first;
    while (c < len and out.items.len < most) : (c += step) try out.append(a, @intCast(c));
    return out.items;
}

/// Message-like cuts: gaps of 256 to 3,000 rows from a fixed seed, each rounded down to a multiple of `align`.
fn messages(a: std.mem.Allocator, len: usize, alignment: usize, seed: u64) ![]const u32 {
    var out: std.ArrayList(u32) = .empty;
    var rng = std.Random.DefaultPrng.init(seed);
    var c: usize = 0;
    while (true) {
        c += 256 + rng.random().uintLessThan(usize, 2745);
        const at = c - c % alignment;
        if (at >= len) break;
        if (at > 0 and (out.items.len == 0 or at > out.items[out.items.len - 1])) try out.append(a, @intCast(at));
    }
    return out.items;
}

/// The patterns a `len`-row prompt is cut in: single cuts, regular cuts, short runs and message-like starts.
fn patterns(a: std.mem.Allocator, len: usize) ![]const Pattern {
    var out: std.ArrayList(Pattern) = .empty;
    try out.append(a, .{ .name = "grid-again", .cuts = &.{} });
    const singles = [_]u32{ 1, 3, 4, 7, 17, 33, 63, 64, 65, 100, 128, 129, 256, 1000, 1023, 1024, 1025, 2047, 2049, 3000 };
    for (singles) |s| if (s < len) try out.append(a, .{ .name = try std.fmt.allocPrint(a, "at-{d}", .{s}), .cuts = try a.dupe(u32, &.{s}) });
    try out.append(a, .{ .name = "every-300", .cuts = try every(a, len, 300, 300, 1 << 20) });
    try out.append(a, .{ .name = "every-512", .cuts = try every(a, len, 512, 512, 1 << 20) });
    try out.append(a, .{ .name = "every-576", .cuts = try every(a, len, 576, 576, 1 << 20) });
    try out.append(a, .{ .name = "run-of-4s", .cuts = try every(a, len, 1000, 4, 17) });
    try out.append(a, .{ .name = "run-of-64s", .cuts = try every(a, len, 1024, 64, 17) });
    try out.append(a, .{ .name = "last-row", .cuts = try a.dupe(u32, &.{@intCast(len - 1)}) });
    for ([_]usize{ 64, 4, 1 }) |al| for ([_]u64{ 1, 2 }) |seed| {
        const cuts = try messages(a, len, al, seed * 7919 + al);
        if (cuts.len > 0) try out.append(a, .{ .name = try std.fmt.allocPrint(a, "messages-{d}-seed{d}", .{ al, seed }), .cuts = cuts });
    };
    return out.items;
}

fn bf16(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

/// The KDA records' copy a state reads next, both cleared in `bytes` (glm_solo_tests: a pass flips them).
fn takeRecords(a: std.mem.Allocator, st: anytype, bytes: []u8) ![]const u8 {
    const at = st.rec - st.arena.buf.ptr;
    const one = st.kda_layers * st.rec_layer;
    const cur: usize = if (st.kda_layers > 0) st.cur[0] else 0;
    const out = try a.dupe(u8, bytes[at + cur * one ..][0..one]);
    @memset(bytes[at..][0 .. 2 * one], 0);
    return out;
}

/// One run of a prompt: its first token, the prompt's last logits row, the state after it, then the tape's windows.
const Run = struct {
    first: u32,
    prompt: []u16,
    state: []u8,
    records: []const u8,
    base: usize,
    mtp_base: usize,
    steps: usize,
    tape: []u16,
    outs: []u32,
};

fn run(a: std.mem.Allocator, gpu: Gpu, x: anytype, e: anytype, slot: u32, ids: []const u32, tape: []const u32, cuts: []const u32) !Run {
    const V = e.share.vocab.len();
    try x.bind(slot);
    try x.begin(null);
    if (cuts.len > 0) try x.cuts(cuts);
    var at: usize = 0;
    var steps: usize = 0;
    const first = while (true) {
        const r = try x.chunk(ids, at, false, null, null, null);
        steps += 1;
        if (r.first) |f| break f;
        at += r.rows;
    };
    var out: Run = .{ .first = first, .prompt = try a.alloc(u16, V), .state = undefined, .records = undefined, .base = 0, .mtp_base = 0, .steps = steps, .tape = try a.alloc(u16, tape.len * V), .outs = try a.alloc(u32, tape.len) };
    try cuda.memory.download(gpu.d, e.bp.logits, std.mem.sliceAsBytes(out.prompt));
    try e.stream.synchronize();
    const st = &e.cur.st;
    out.state = try a.alloc(u8, st.arena.buf.len);
    try st.arena.buf.download(0, out.state);
    out.records = try takeRecords(a, st, out.state);
    out.base = st.base;
    out.mtp_base = st.mtp_base;
    var w: usize = 0;
    while (w < tape.len) : (w += tape_rows) {
        const win = tape[w..][0..tape_rows];
        try x.verify(win, out.outs[w..][0..tape_rows], 0);
        try cuda.memory.download(gpu.d, e.bd.logits, std.mem.sliceAsBytes(out.tape[w * V ..][0 .. tape_rows * V]));
        try x.keep(tape_rows);
    }
    return out;
}

/// The largest |difference| between two bf16 logit runs and how many differ.
fn compare(got: []const u16, want: []const u16) struct { n: usize, max: f32 } {
    var n: usize = 0;
    var max: f32 = 0;
    for (got, want) |g, w| if (g != w) {
        n += 1;
        max = @max(max, @abs(bf16(g) - bf16(w)));
    };
    return .{ .n = n, .max = max };
}

/// `glm-cut-check`: prompts cut at request starts against the grid's run: first token, the prompt's last logits,
/// every state byte, then teacher-forced windows' logits and tokens. Reports each pattern; fails only on an error.
/// `tokens`: a file of whitespace- or comma-separated ids (the prompt, then the tape), or `synthetic`.
/// TF_GLM_CUT_KV8=1 runs the sequences with fp8 KV (--kv-cache fp8).
pub fn cutCheck(gpu: Gpu, checkpoint: []const u8, tokens: []const u8, lens_arg: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lens: std.ArrayList(usize) = .empty;
    var it = std.mem.tokenizeScalar(u8, lens_arg, ',');
    while (it.next()) |s| try lens.append(a, try std.fmt.parseInt(usize, s, 10));
    try check.expect(lens.items.len > 0, "glm-cut-check: no prompt lengths in {s}", .{lens_arg});
    const most = std.mem.max(usize, lens.items);
    const need = most + tape_windows * tape_rows;
    var pool: []u32 = try a.alloc(u32, need);
    if (std.mem.eql(u8, tokens, "synthetic")) {
        for (pool, 0..) |*u, j| u.* = @intCast(100 + (j * 7919) % 5000);
    } else {
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, tokens, a, .limited(1 << 28));
        var n: usize = 0;
        var t = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
        while (t.next()) |s| : (n += 1) {
            if (n == need) break;
            pool[n] = try std.fmt.parseInt(u32, s, 10);
        }
        try check.expect(n == need, "glm-cut-check: {s} holds {d} ids, the longest prompt and its tape need {d}", .{ tokens, n, need });
    }
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, rank, world, id_path);
    defer _ = lib.api.ncclCommDestroy(comm);
    var problem: []const u8 = "";
    var set = glm.ranks.Settings.read(&glm.ranks.environment, &problem) catch |err| {
        std.debug.print("REFUSE {s}\n", .{problem});
        return err;
    };
    if (std.c.getenv("TF_GLM_CUT_KV8")) |v| set.kv8 = std.mem.eql(u8, std.mem.span(v), "1");
    const capacity = need + 4 * glm.engine.prompt_rows;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    if (rank != 0) {
        try glm.follow.follow(e, &link);
        check.pass("rank {d} followed rank 0's cut prompt passes to the end", .{rank});
        return;
    }
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    std.debug.print("CUT setup world={d} kv8={} dense_bits={d} head_bits={d} split={} grid={d} tokens={s}\n", .{ world, set.kv8, set.dense_bits, set.head_bits, set.split, glm.engine.prompt_rows, tokens });
    var equal: usize = 0;
    var total: usize = 0;
    for (lens.items) |len| {
        var la = std.heap.ArenaAllocator.init(gpu.gpa);
        defer la.deinit();
        const l = la.allocator();
        const ids = pool[0..len];
        const tape = pool[len..][0 .. tape_windows * tape_rows];
        const want = try run(l, gpu, x, e, 0, ids, tape, &.{});
        for (try patterns(l, len)) |p| {
            const got = try run(l, gpu, x, e, 1, ids, tape, p.cuts);
            const pr = compare(got.prompt, want.prompt);
            const tp = compare(got.tape, want.tape);
            const layout = got.base == want.base and got.mtp_base == want.mtp_base and got.state.len == want.state.len;
            const state: []const u8 = if (!layout) "layout" else if (std.mem.indexOfDiff(u8, got.state, want.state) == null and std.mem.eql(u8, got.records, want.records)) "equal" else "differs";
            const same = got.first == want.first and pr.n == 0 and tp.n == 0 and std.mem.eql(u32, got.outs, want.outs) and std.mem.eql(u8, state, "equal");
            equal += @intFromBool(same);
            total += 1;
            std.debug.print("CUT len={d} pattern={s} class={s} cuts={d} steps={d}/{d} first={s} prompt_logits={d} prompt_max={d:.6} state={s} tape_logits={d} tape_max={d:.6} tape_tokens={s} verdict={s}\n", .{
                len,                                                               p.name,                        class(p.cuts), p.cuts.len, got.steps, want.steps,
                if (got.first == want.first) "equal" else "differs",               pr.n,                          pr.max,        state,      tp.n,      tp.max,
                if (std.mem.eql(u32, got.outs, want.outs)) "equal" else "differs", if (same) "EQUAL" else "DIFF",
            });
        }
    }
    check.pass("glm-cut-check at TP={d}: {d} of {d} cut patterns equal the grid's run byte for byte (rank 0's logits and state)", .{ world, equal, total });
}
