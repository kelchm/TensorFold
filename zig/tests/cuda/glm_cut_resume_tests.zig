//! GLM-5.3-Flash prompt passes cut at request starts, kept and resumed, against fresh grid runs, on every rank.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

const Hash = std.hash.Wyhash;
const tape_len = 48;
const widths = [_]usize{ 1, 3, 8, 2, 5, 4, 7, 6 }; // verify windows, each kept in part or whole
const poisons = [_]u8{ 0x41, 0x5a, 0x77 }; // distinct finite fill bytes for bf16, fp32 and fp8 alike

/// One cache's canonical rows at `n` positions: hashes of the latent rows, their scales and the completed pooled rows
/// (tf_glm_pool_keys writes no incomplete pool), and one hash per index-ring row from the ring's base.
const Cache = struct { latent: u64, scales: u64, pooled: u64, base: usize, n: usize, ik: []u64, ig: []u64 };

/// A rank's committed state, canonical: rows by position, the KDA records' current copy, no scratch.
const Print = struct { conv: u64, rec: []u64, caches: []Cache, pos: usize, mtp_len: usize, cursors: u64, held: u64 };

/// Everything one run observes on this rank.
const Run = struct { first: u32, prompt: []u16, print: Print, after: Print, logits: []u16, mtp: []u16, outs: []u32, drafts: []u32 };

fn hashDev(gpu: Gpu, buf: []u8, ptr: u64, len: usize) !u64 {
    var h = Hash.init(0);
    var done: usize = 0;
    while (done < len) {
        const n = @min(buf.len, len - done);
        try cuda.memory.download(gpu.d, ptr + done, buf[0..n]);
        h.update(buf[0..n]);
        done += n;
    }
    return h.final();
}

fn ringRows(a: std.mem.Allocator, gpu: Gpu, buf: []u8, raw: u64, base: usize, n: usize, row: usize) ![]u64 {
    const held = if (n > base) n - base else 0;
    const out = try a.alloc(u64, held);
    var done: usize = 0;
    while (done < held) {
        const k = @min(buf.len / row, held - done);
        try cuda.memory.download(gpu.d, raw + done * row, buf[0 .. k * row]);
        for (0..k) |i| out[done + i] = Hash.hash(0, buf[i * row ..][0..row]);
        done += k;
    }
    return out;
}

fn cachePrint(a: std.mem.Allocator, gpu: Gpu, buf: []u8, e: anytype, x: anytype, rings: [2]u64, base: usize, n: usize) !Cache {
    const sparse = @import("glm_cuda").sparse;
    const st = &e.cur.st;
    const latent_row: usize = @as(usize, e.c.kv_lora) * @as(usize, if (x.ls == 0) 2 else 1);
    const pooled_row: usize = @as(usize, e.c.i_dim) * 2;
    const full = n / sparse.pool;
    return .{
        .latent = try hashDev(gpu, buf, x.lc, n * latent_row),
        .scales = if (x.ls == 0) 0 else try hashDev(gpu, buf, x.ls, n * @sizeOf(f32)),
        .pooled = try hashDev(gpu, buf, x.pk, full * pooled_row),
        .base = base,
        .n = n,
        .ik = try ringRows(a, gpu, buf, rings[0], base, n, st.ring_row),
        .ig = try ringRows(a, gpu, buf, rings[1], base, n, st.ring_row),
    };
}

fn print(a: std.mem.Allocator, gpu: Gpu, buf: []u8, e: anytype) !Print {
    const st = &e.cur.st;
    try check.expect(st.table == 0, "glm-cut-resume compares contiguous sequences only", .{});
    try e.stream.synchronize();
    const rec = try a.alloc(u64, st.kda_layers);
    for (rec, 0..) |*r, l| r.* = try hashDev(gpu, buf, st.recAt(st.cur[l], l), st.rec_layer);
    const caches = try a.alloc(Cache, st.caches.len + 1);
    for (st.caches, 0..) |x, j| caches[j] = try cachePrint(a, gpu, buf, e, x, .{ st.rings[2 * j], st.rings[2 * j + 1] }, st.base, st.pos);
    const j = st.caches.len;
    caches[j] = try cachePrint(a, gpu, buf, e, st.mtp_caches, .{ st.rings[2 * j], st.rings[2 * j + 1] }, st.mtp_base, st.mtp_len);
    const cursors = (try hashDev(gpu, buf, st.pos_dev, 4)) ^ (try hashDev(gpu, buf, st.mtp_pos_dev, 4)) *% 31;
    const held = try hashDev(gpu, buf, st.prompt_hidden, @as(usize, e.c.hidden) * 2);
    return .{ .conv = try hashDev(gpu, buf, st.conv, st.kda_layers * 3 * st.conv_c * 2), .rec = rec, .caches = caches, .pos = st.pos, .mtp_len = st.mtp_len, .cursors = cursors, .held = held };
}

/// What differs between two prints: layers or rows, by kind.
const Diff = struct { pos: bool = false, conv: bool = false, cursors: bool = false, held: bool = false, rec: usize = 0, latent: usize = 0, scales: usize = 0, pooled: usize = 0, ring: usize = 0, overlap: usize = 0, bases: usize = 0 };

fn diff(g: Print, w: Print) Diff {
    var d: Diff = .{ .pos = g.pos != w.pos or g.mtp_len != w.mtp_len, .conv = g.conv != w.conv, .cursors = g.cursors != w.cursors, .held = g.held != w.held };
    for (g.rec, w.rec) |x, y| d.rec += @intFromBool(x != y);
    for (g.caches, w.caches) |x, y| {
        d.latent += @intFromBool(x.latent != y.latent);
        d.scales += @intFromBool(x.scales != y.scales);
        d.pooled += @intFromBool(x.pooled != y.pooled);
        d.bases += @intFromBool(x.base != y.base);
        if (x.n != y.n) continue;
        const lo = @max(x.base, y.base); // positions both rings still hold
        for (lo..x.n) |p| {
            if (p - x.base >= x.ik.len or p - y.base >= y.ik.len) continue;
            d.overlap += 1;
            d.ring += @intFromBool(x.ik[p - x.base] != y.ik[p - y.base] or x.ig[p - x.base] != y.ig[p - y.base]);
        }
    }
    return d;
}

fn quiet(d: Diff) bool {
    return !d.pos and !d.conv and !d.cursors and !d.held and d.rec == 0 and d.latent == 0 and d.scales == 0 and d.pooled == 0 and d.ring == 0;
}

fn bf16(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

fn logitDiff(got: []const u16, want: []const u16) struct { n: usize, max: f32 } {
    var n: usize = 0;
    var max: f32 = 0;
    for (got, want) |g, w| if (g != w) {
        n += 1;
        max = @max(max, @abs(bf16(g) - bf16(w)));
    };
    return .{ .n = n, .max = max };
}

/// A fresh sequence in `slot` (an old one is released first), every arena byte set to `fill`, then begun with `cuts`.
fn fresh(e: anytype, slot: u32, fill: u8, cuts: []const u32) !void {
    if (slot != 0 and slot < e.seqs.items.len and e.seqs.items[slot] != null) e.release(slot);
    try e.bind(slot);
    const st = &e.cur.st;
    try e.d.check(e.d.api.cuMemsetD8Async(st.arena.buf.ptr, fill, st.arena.buf.len, e.stream.handle), "cuMemsetD8Async");
    try e.begin(null);
    for (cuts, 0..) |c, i| try check.expect(c > 0 and (i == 0 or c > cuts[i - 1]), "cuts must ascend after 0", .{});
    try e.cur.cuts.appendSlice(e.gpa, cuts);
}

/// Bits of a bf16 value whose exponent is all ones: an infinity or a NaN.
fn nonfinite(xs: []const u16) usize {
    var n: usize = 0;
    for (xs) |x| n += @intFromBool(x & 0x7f80 == 0x7f80);
    return n;
}

/// The prompt's first token and last logits row, the canonical state after it, then the teacher-forced tape:
/// windows of varied widths, each kept in part (KDA replay) or whole, with MTP drafts after every keep whose follow
/// ids are each kept row's successor; then the canonical state again after the last keep.
fn observe(a: std.mem.Allocator, gpu: Gpu, buf: []u8, e: anytype, first: u32, tape: []const u32) !Run {
    const glm = @import("glm_cuda");
    const V = e.share.vocab.len();
    var r: Run = .{ .first = first, .prompt = try a.alloc(u16, V), .print = undefined, .after = undefined, .logits = &.{}, .mtp = &.{}, .outs = &.{}, .drafts = &.{} };
    try cuda.memory.download(gpu.d, e.bp.logits, std.mem.sliceAsBytes(r.prompt));
    r.print = try print(a, gpu, buf, e);
    var logits: std.ArrayList(u16) = .empty;
    var mtp: std.ArrayList(u16) = .empty;
    var outs: std.ArrayList(u32) = .empty;
    var drafts: std.ArrayList(u32) = .empty;
    var d: [glm.engine.max_drafts]u32 = undefined;
    try e.draft(true, tape[0..1], 2, &d, null); // the prompt's last row is followed by the tape's first id
    try drafts.appendSlice(a, d[0..2]);
    try cuda.memory.download(gpu.d, e.bm.logits, std.mem.sliceAsBytes(try mtp.addManyAsSlice(a, V)));
    var off: usize = 0;
    var i: usize = 0;
    while (off + widths[i % widths.len] + 1 <= tape.len) : (i += 1) { // a kept row's successor stays on the tape
        const w = widths[i % widths.len];
        const win = tape[off..][0..w];
        var o: [8]u32 = undefined;
        try e.verify(win, o[0..w], 0);
        try cuda.memory.download(gpu.d, e.bd.logits, std.mem.sliceAsBytes(try logits.addManyAsSlice(a, w * V)));
        try outs.appendSlice(a, o[0..w]);
        const keep = if (i % 3 == 2) w else @max(1, w - 1 - i % 2); // mostly partial keeps
        try e.keep(keep);
        try e.draft(false, tape[off + 1 ..][0..keep], 2, &d, null);
        try drafts.appendSlice(a, d[0..2]);
        try cuda.memory.download(gpu.d, e.bm.logits, std.mem.sliceAsBytes(try mtp.addManyAsSlice(a, V)));
        off += keep;
    }
    r.after = try print(a, gpu, buf, e);
    r.logits = logits.items;
    r.mtp = mtp.items;
    r.outs = outs.items;
    r.drafts = drafts.items;
    return r;
}

fn report(rank: c_int, case: []const u8, against: []const u8, got: Run, want: Run) bool {
    const p = logitDiff(got.prompt, want.prompt);
    const t = logitDiff(got.logits, want.logits);
    const m = logitDiff(got.mtp, want.mtp);
    const d = diff(want.print, got.print);
    const after = diff(want.after, got.after);
    const bad = nonfinite(got.prompt) + nonfinite(got.logits) + nonfinite(got.mtp);
    const tokens = got.first == want.first and std.mem.eql(u32, got.outs, want.outs);
    const drafts = std.mem.eql(u32, got.drafts, want.drafts);
    const exact = tokens and drafts and bad == 0 and p.n == 0 and t.n == 0 and m.n == 0 and quiet(d) and quiet(after);
    std.debug.print("CUT2 rank={d} {s} vs={s} tokens={s} drafts={s} nonfinite={d} prompt_logits={d}@{d:.6} tape_logits={d}@{d:.6} mtp_logits={d}@{d:.6} state[pos={s} conv={s} cursors={s} held={s} rec={d} latent={d} scales={d} pooled={d} ring={d}/{d} bases={d}] after[conv={s} rec={d} latent={d} pooled={d} ring={d}/{d}] verdict={s}\n", .{
        rank,                               case,                               against,
        if (tokens) "equal" else "differs", if (drafts) "equal" else "differs", bad,
        p.n,                                p.max,                              t.n,
        t.max,                              m.n,                                m.max,
        if (d.pos) "differs" else "equal",  if (d.conv) "differs" else "equal", if (d.cursors) "differs" else "equal",
        if (d.held) "differs" else "equal", d.rec,                              d.latent,
        d.scales,                           d.pooled,                           d.ring,
        d.overlap,                          d.bases,                            if (after.conv or after.pos or after.cursors) "differs" else "equal",
        after.rec,                          after.latent,                       after.pooled,
        after.ring,                         after.overlap,                      if (exact) "EXACT" else "DIFF",
    });
    return exact;
}

/// Message-like starts below `end`: gaps of 256 to 3,000 rows from a seed, each rounded down to `alignment`.
fn messages(a: std.mem.Allocator, end: usize, alignment: usize, seed: u64) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    var rng = std.Random.DefaultPrng.init(seed);
    var c: usize = 0;
    while (true) {
        c += 256 + rng.random().uintLessThan(usize, 2745);
        const at = c - c % alignment;
        if (at >= end) break;
        if (at > 0 and (out.items.len == 0 or at > out.items[out.items.len - 1])) try out.append(a, @intCast(at));
    }
    return out.items;
}

/// The served prompt pass a chunk at a time, as Leader.chunk steps it on every rank without pairing or solo passes:
/// to the next cut or prefill_rows on, the indexer split decided by the chunk's start, `from` on the first step only.
/// (Paired and solo steps against single chunks are glm-pair-check's and glm-solo-check's.)
fn pass(e: anytype, ids: []const u32, from: anytype, marks: anytype) !u32 {
    const glm = @import("glm_cuda");
    var at: usize = if (from) |r| r.at else 0;
    var first_step = true;
    while (true) {
        var end = @min(ids.len, at + glm.engine.prompt_rows);
        for (e.cur.cuts.items) |c| if (c > at) {
            end = @min(end, c);
            break;
        };
        e.index_split = glm.sparse.splitAt(at, e.settings.index_split, e.share.world);
        const part = ids[at..@min(ids.len, end + 1)];
        if (try e.prefillChunk(part, at, ids.len, if (first_step) from else null, null, marks)) |t| return t;
        first_step = false;
        at = end;
    }
}

/// A mark's call: keep the state at the mark, on this rank, under the case's id.
const Keeper = struct {
    e: *@import("glm_cuda").engine.Engine,
    id: u64,
    kept: bool = false,
    failed: ?anyerror = null,
    fn call(ptr: *anyopaque, _: u32) void {
        const k: *Keeper = @ptrCast(@alignCast(ptr));
        k.e.keepAs(k.id) catch |err| {
            k.failed = err;
            return;
        };
        k.kept = true;
    }
};

/// `glm-cut-resume`: every rank runs the same steps on its own engine and compares its own logits and state.
/// For each (history, extension): fresh grid passes under two fill bytes (G1, G2); a fresh pass cut at message starts
/// and at the history's end, keeping its state at that mark as the server does (F); another conversation over a
/// fresh sequence (X); and the extended prompt resumed from F's kept state in a third fresh sequence (R). R must
/// equal F (a resume equals a fresh pass with the same cuts), F must equal G1 (cuts change nothing), G2 must equal G1
/// (nothing reads an unwritten byte). Prints every comparison, then fails if any rank saw a difference.
/// `ids`: whitespace-separated token ids.
pub fn cutResume(gpu: Gpu, checkpoint: []const u8, ids_path: []const u8, rank: c_int, world: c_int, id_path: []const u8) !void {
    const glm = @import("glm_cuda");
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { history: usize, ext: usize };
    const cases = [_]Case{ .{ .history = 5001, .ext = 203 }, .{ .history = 12003, .ext = 602 }, .{ .history = 30002, .ext = 1101 } };
    var most: usize = 0;
    for (cases) |c| most = @max(most, c.history + c.ext);
    const need = most + tape_len + 4096; // and another conversation's ids past them
    const pool = try a.alloc(u32, need);
    {
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, a, .limited(1 << 28));
        var n: usize = 0;
        var t = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
        while (t.next()) |s| : (n += 1) {
            if (n == need) break;
            pool[n] = try std.fmt.parseInt(u32, s, 10);
        }
        try check.expect(n == need, "glm-cut-resume: {s} holds {d} ids, {d} needed", .{ ids_path, n, need });
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
    const capacity = most + tape_len + 2 * glm.engine.prompt_rows;
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, @intCast(rank), @intCast(world), .{ .capacity = capacity, .mtp = true, .settings = set });
    defer e.deinit();
    for (pool) |id| try check.expect(id < e.c.vocab, "glm-cut-resume: id {d} is past the vocabulary ({d})", .{ id, e.c.vocab });
    try glm.snapshots.agree(e, 8 << 30);
    try e.bind(0); // release() binds slot 0 again, so it must exist
    const buf = try a.alloc(u8, 64 << 20);
    std.debug.print("CUT2 setup rank={d} world={d} kv8={} dense_bits={d} head_bits={d} grid={d} index_split={d} capacity={d}\n", .{ rank, world, set.kv8, set.dense_bits, set.head_bits, glm.engine.prompt_rows, set.index_split, capacity });
    var exact: usize = 0;
    var total: usize = 0;
    for (cases, 0..) |c, ci| {
        var la = std.heap.ArenaAllocator.init(gpu.gpa);
        defer la.deinit();
        const l = la.allocator();
        const len = c.history + c.ext;
        const ids = pool[0..len];
        const tape = pool[len..][0..tape_len];
        const other = pool[most + tape_len ..][0..4096];
        try fresh(e, 1, poisons[0], &.{});
        const g1 = try observe(l, gpu, buf, e, try pass(e, ids, @as(?glm.engine.Resume, null), @as(?glm.engine.Marks, null)), tape);
        try fresh(e, 2, poisons[1], &.{});
        const g2 = try observe(l, gpu, buf, e, try pass(e, ids, @as(?glm.engine.Resume, null), @as(?glm.engine.Marks, null)), tape);
        exact += @intFromBool(report(rank, try std.fmt.allocPrint(l, "history={d} ext={d} len%4={d} run=G2", .{ c.history, c.ext, len % 4 }), "G1", g2, g1));
        total += 1;
        const patterns = [_]struct { name: []const u8, below: []u32 }{
            .{ .name = "end-only", .below = &.{} },
            .{ .name = "messages-any", .below = try messages(l, c.history, 1, 7 * ci + 1) },
            .{ .name = "messages-64", .below = try messages(l, c.history, 64, 7 * ci + 2) },
        };
        for (patterns, 0..) |pat, pi| {
            const with_end = try std.mem.concat(l, u32, &.{ pat.below, &.{@as(u32, @intCast(c.history))} });
            var keeper: Keeper = .{ .e = e, .id = 1000 * ci + pi + 1 };
            const marks: glm.engine.Marks = .{ .at = &.{@as(u32, @intCast(c.history))}, .ctx = &keeper, .call = Keeper.call };
            try fresh(e, 3, poisons[pi % 3], with_end);
            const f = try observe(l, gpu, buf, e, try pass(e, ids, @as(?glm.engine.Resume, null), @as(?glm.engine.Marks, marks)), tape);
            if (keeper.failed) |err| return err;
            try check.expect(keeper.kept, "glm-cut-resume: the mark at {d} kept no state", .{c.history});
            try fresh(e, 4, poisons[(pi + 1) % 3], try messages(l, other.len, 1, 99 + pi)); // another conversation meanwhile
            _ = try pass(e, other, @as(?glm.engine.Resume, null), @as(?glm.engine.Marks, null));
            try fresh(e, 5, poisons[(pi + 2) % 3], with_end);
            const from: glm.engine.Resume = .{ .id = keeper.id, .at = c.history };
            const r = try observe(l, gpu, buf, e, try pass(e, ids, @as(?glm.engine.Resume, from), @as(?glm.engine.Marks, null)), tape);
            e.dropState(keeper.id);
            const base = try std.fmt.allocPrint(l, "history={d} ext={d} len%4={d} cuts={d} pattern={s}", .{ c.history, c.ext, len % 4, with_end.len, pat.name });
            exact += @intFromBool(report(rank, try std.fmt.allocPrint(l, "{s} run=F", .{base}), "G1", f, g1));
            exact += @intFromBool(report(rank, try std.fmt.allocPrint(l, "{s} run=R", .{base}), "F", r, f));
            exact += @intFromBool(report(rank, try std.fmt.allocPrint(l, "{s} run=R", .{base}), "G1", r, g1));
            total += 3;
        }
        for ([_]u32{ 1, 2, 3, 4, 5 }) |s| e.release(s);
    }
    std.debug.print("CUT2 summary rank={d}: {d} of {d} comparisons exact\n", .{ rank, exact, total });
    const worst = try e.shared(@floatFromInt(total - exact)); // every rank's count of differing comparisons, the most
    if (worst > 0) {
        std.debug.print("FAIL glm-cut-resume: a rank saw {d} differing comparisons\n", .{@as(usize, @intFromFloat(worst))});
        return error.TestFailed;
    }
    check.pass("glm-cut-resume at TP={d} rank {d}: all {d} comparisons exact in tokens, drafts, logits, MTP logits and canonical state, on every rank", .{ world, rank, total });
}
