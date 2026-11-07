//! One TP rank's GLM-5.3-Flash CUDA engine (the Python family's decode.Engine); all ranks draw the same tokens.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const config = @import("config.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const sampler = @import("cuda_sampler.zig");
const prompt_lp = @import("cuda_prompt_logprobs.zig");
const heat = @import("heat");
const modelopt = @import("core").modelopt;
const dflash = @import("dflash.zig");
const prof_mod = @import("cuda_prof.zig");
const prompt_pair = @import("cuda_prompt_pair.zig");
const kv_attn = @import("cuda_kv_shard_attn.zig");
const replay_timing = @import("cuda_replay_timing.zig");
const replay_audit = @import("cuda_replay_audit.zig");
const snapshots = @import("cuda_snapshots.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Options = @import("cuda_engine_types.zig").Options;
const Resume = @import("cuda_engine_types.zig").Resume;
const Marks = @import("cuda_engine_types.zig").Marks;
const Part = @import("cuda_engine_types.zig").Part;
const Cancel = @import("cuda_engine.zig").Cancel;
const max_shared = 16;
const shared_rows = 32;
const max_drafts = 3;

/// Prefill in ordinary chunks, restoring a kept state when supplied.
pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
    var start = try e.prefillStart(ids.len, from);
    while (true) : (start += fw.prefill_rows) {
        if (try e.prefillChunk(ids[start..@min(ids.len, start + fw.prefill_rows + 1)], start, ids.len, null, cancel, marks)) |first| return first;
    }
}

/// A prompt of `len` ids and the verify window after it inside a sequence of `capacity` rows (equal fits).
pub fn promptFits(len: usize, capacity: usize) bool {
    return len > 0 and len + fw.window_rows <= capacity;
}

/// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
pub fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
    if (!promptFits(len, e.cur.st.capacity)) return error.PromptTooLong;
    const r = from orelse return 0;
    if (r.at == 0 or r.at >= len or r.at % fw.prefill_rows != 0) return error.BadResume;
    try e.restoreState(r.id); // DFlash2's context too: its kept rows, or from r.at when it kept none
    return r.at;
}

/// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
    const split = e.index_split; // this step's alone
    e.index_split = false;
    if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
    if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
    const rows = @min(len - start, fw.prefill_rows);
    if (part.len > rows + 1) return prompt_pair.prefill(e, part, start, len, cancel, marks, split);
    if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
    var one: [1]fw.Rank = undefined;
    var f = e.view(&e.bp, &one);
    f.index_split = split;
    if (kv_attn.wanted(&f, one[0])) { // TF_GLM_KV_SHARD, no twin: its exchanges on the main stream, whole
        f.pieces = null;
        f.ce = null;
        if (e.kv_ctx.counts) |c| c.owner_prompts += 1;
    }
    if (start + rows <= len -| @as(usize, @intCast(dflash.WIN))) f.taps = null; // none read (dflash.promptRun)
    const as_i32: []const i32 = @ptrCast(part);
    if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
    f.prof = e.prompt_profs[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3];
    const labels = [_][]const u8{ "prompt chunk to 64 rows", "prompt chunk to 256 rows", "prompt chunk to 1024 rows", "prompt chunk to 2048 rows" };
    if (f.prof) |p| try p.beginPrompt(e.stream, labels[if (rows <= 64) 0 else if (rows <= 256) 1 else if (rows <= 1024) 2 else 3], .{ .rank = e.share.rank, .slot = e.slot, .start = start, .rows = rows }, e.c.run, e.pieces.count);
    if (f.prof) |p| e.profiling_prompt = p;
    defer if (f.prof != null) {
        e.profiling_prompt = null;
    };
    defer if (f.prof) |p| p.end();
    errdefer if (f.prof) |p| {
        p.prompt.?.invalid = true;
    };
    var stamp = prof_mod.span(f.prof, e.stream, .gate);
    e.gate(cancel) catch |err| {
        if (err == error.Cancelled) try snapshots.interrupted(e, marks, @intCast(start));
        return err;
    };
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .gate);
    try f.run(as_i32[0..rows]);
    e.index_splits += @intFromBool(split);
    stamp = prof_mod.span(f.prof, e.stream, .prompt_score);
    if (e.prompt_logprobs) |p| if (part.len > 1) try prompt_lp.run(e, p, part[1..]);
    stamp.end();
    stamp = prof_mod.span(f.prof, e.stream, .drafter);
    if (e.drafter) |dr| try dr.prompt(e, e.slot, start, rows, len);
    if (e.drafter) |dr| if (start + rows == len) try dr.ringDigest(e, e.slot, len);
    stamp.end();
    stamp = prof_mod.span(f.prof, e.stream, .draw);
    const last = start + rows == len;
    var first: ?u32 = null;
    if (last) {
        var t: [1]u32 = undefined;
        try e.draw(e.bp.logits, 1, len, 0, &t);
        first = t[0];
    }
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .draw);
    if (last) try e.holdPrompt(e.bp.fnormed + (rows - 1) * e.c.hidden * 2);
    stamp = prof_mod.span(f.prof, e.stream, .mtp);
    // With a drafter (DFlash2) every draft is its block's (cuda_lanes draftFn); the target MTP layer's caches are never
    if (e.mtp and e.drafter == null) {
        const nxt = as_i32[1..];
        if (nxt.len > 0) { // decode._absorb_rows
            const mp = f.prof;
            defer f.prof = mp;
            f.prof = null; // the head's pass as one section
            try f.mtp(nxt, &.{e.bp.fnormed});
            f.prof = mp;
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
        }
    }
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .mtp);
    stamp = prof_mod.span(f.prof, e.stream, .commit);
    try f.commit(rows);
    stamp.end();
    if (f.prof) |p| p.mark(e.stream, .commit);
    return first;
}

/// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
pub fn verify(e: *Engine, ids: []const u32, out: []u32, bits: u64) !void {
    if (ids.len == 0 or ids.len > fw.window_rows) return error.WindowTooWide;
    if (e.replay) |r| try r.audit.start(e, &.{.{ .slot = e.slot, .pos = e.pos(), .rows = ids.len }});
    try e.giveTwin(e.slot);
    var one: [1]fw.Rank = undefined;
    var f = e.view(&e.bd, &one);
    const at = e.cur.st.pos;
    f.prof = e.profs[ids.len - 1];
    if (f.prof) |p| try p.begin(e.stream, verifyLabel(ids.len));
    f.replay_timing = if (e.replay) |r| r.audit.trace else null;
    try f.run(@ptrCast(ids));
    try replay_timing.mark(f.replay_timing, e.stream, .bookkeeping);
    if (e.drafter) |dr| try dr.noteVerify(e.slot, 0, ids.len, at, true);
    e.cur.rows_at = e.bd.fnormed;
    try replay_timing.mark(f.replay_timing, e.stream, .selection);
    try e.draw(e.bd.logits, ids.len, at + 1, bits, out);
    try replay_timing.mark(f.replay_timing, e.stream, .target_commit);
    if (f.prof) |p| p.end();
}

const verifyLabel = @import("cuda_engine_prof.zig").verifyLabel;
const sharedLabel = @import("cuda_engine_prof.zig").sharedLabel;

/// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
pub fn verifyShared(e: *Engine, parts: []const Part) !void {
    const b = if (e.bs) |*x| x else return error.NoSharedWindows;
    if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
    if (e.replay) |r| {
        var audit_parts: [4]replay_audit.Part = undefined;
        if (parts.len > audit_parts.len) return error.ReplayCapacity;
        for (parts, audit_parts[0..parts.len]) |p, *a| {
            const seq = e.seqs.items[p.slot] orelse return error.UnknownSlot;
            a.* = .{ .slot = p.slot, .pos = seq.st.pos, .rows = p.ids.len };
        }
        try r.audit.start(e, audit_parts[0..parts.len]);
    }
    var segs: [max_shared]fw.Seg = undefined;
    var ids: [shared_rows]u32 = undefined;
    var rows: usize = 0;
    for (parts, 0..) |p, i| {
        if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
        for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
        try e.bind(p.slot);
        try e.giveTwin(p.slot);
        segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
        @memcpy(ids[rows..][0..p.ids.len], p.ids);
        rows += p.ids.len;
    }
    var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
    var f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .ring = e.settings.ring, .kda_wide = e.settings.kda_wide, .absorb_mma = e.settings.numerics.absorb_mma, .taps = if (e.drafter) |dr| dr.taps(false) else null, .side = e.side, .moe_fuse = e.moe_fuse, .kv = e.kvCtx() };
    f.prof = e.shared_profs[rows - 1];
    if (f.prof) |p| try p.begin(e.stream, sharedLabel(rows));
    defer if (f.prof) |p| p.end();
    f.replay_timing = if (e.replay) |r| r.audit.trace else null;
    try f.run(@ptrCast(ids[0..rows]));
    try replay_timing.mark(f.replay_timing, e.stream, .bookkeeping);
    if (e.drafter) |dr| for (parts, segs[0..parts.len], 0..) |p, g, i| try dr.noteVerify(p.slot, g.row0, g.rows, g.st.pos, i == 0);
    // every part greedy (no grammar, no sampled stream; the same on every rank): all rows in one pick, one launch,
    var all_greedy = !e.structured;
    for (parts) |p| all_greedy = all_greedy and e.seqs.items[p.slot].?.sampling == null;
    try replay_timing.mark(f.replay_timing, e.stream, .selection);
    if (all_greedy) {
        var outs: [shared_rows]u32 = undefined;
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        }
        try @import("cuda_engine_pick.zig").greedyRows(e, b.logits, rows, outs[0..rows]);
        for (parts, segs[0..parts.len]) |p, g| @memcpy(p.out[0..g.rows], outs[g.row0..][0..g.rows]);
        try replay_timing.mark(f.replay_timing, e.stream, .target_commit);
        return;
    }
    for (parts, segs[0..parts.len]) |p, g| {
        try e.bind(p.slot);
        e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
        try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.bits, p.out);
    }
    try replay_timing.mark(f.replay_timing, e.stream, .target_commit);
}

/// DFlash2's round (every rank, the same `reqs`): each stream's kept rows into its context, then the block pass.
pub fn draftBlock(e: *Engine, reqs: []dflash.Req) !void {
    const dr = e.drafter orelse return error.NoDrafter;
    for (reqs) |*q| q.pos = @intCast((e.seqs.items[q.slot] orelse return error.UnknownSlot).st.pos);
    try dr.draft(e, reqs);
}

/// The kept rows into the contexts of streams a copy drafts next, with no block pass (every rank).
pub fn ingestBlock(e: *Engine, reqs: []dflash.Req) !void {
    const dr = e.drafter orelse return error.NoDrafter;
    for (reqs) |*q| q.pos = @intCast((e.seqs.items[q.slot] orelse return error.UnknownSlot).st.pos);
    if (e.replay) |r| try r.audit.validate(e, reqs);
    try replay_timing.mark(if (e.replay) |r| r.audit.trace else null, e.stream, .context_commit);
    try dr.ingestOnly(e, reqs);
    if (e.replay) |r| try r.audit.ingested(e);
}

/// forward.commit of the last window's first `n` rows (the KDA replay when fewer than it ran).
pub fn keep(e: *Engine, n: usize) !void {
    if (e.replay) |r| try r.audit.keep(e.slot, n);
    var one: [1]fw.Rank = undefined;
    const f = e.view(&e.bd, &one);
    try f.commit(n);
}

/// decode.draft: the MTP head absorbs the kept rows (committed first) with `follow`, then chains `depth` drafts.
pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32, chances: ?[]f64) !void {
    if (!e.mtp) return error.NoDraftHead;
    if (depth > max_drafts) return error.TooDeep;
    const bm = if (e.bm) |*b| b else return error.NoDraftHead;
    var one: [1]fw.Rank = undefined;
    const f = e.view(bm, &one);
    const D = e.c.hidden;
    const hidden = if (from_prompt) @import("cuda_prompt_hold.zig").input(e) else e.cur.rows_at;
    try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len - e.cur.in_head);
    e.cur.in_head = 0;
    const next: []const i32 = @ptrCast(follow);
    var s0: usize = 0;
    while (s0 < next.len) : (s0 += fw.window_rows) {
        const part = next[s0..@min(next.len, s0 + fw.window_rows)];
        try f.mtp(part, &.{hidden + s0 * D * 2});
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + part.len);
    }
    for (0..depth) |j| {
        if (chances != null and !e.structured and e.cur.sampling == null) {
            const p = try @import("cuda_engine_pick.zig").greedyChance(e, bm.logits);
            out[j] = p.token;
            chances.?[j] = p.chance;
        } else {
            try e.draw(bm.logits, 1, e.cur.st.pos + 1 + j, 0, out[j..][0..1]);
            if (chances) |c| c[j] = 1; // a sampled or structured draft keeps chance 1: its window runs whole
        }
        if (j + 1 < depth) {
            try f.mtp(@ptrCast(out[j..][0..1]), &.{bm.fnormed});
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + 1);
            e.cur.in_head += 1;
        }
    }
}

/// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
pub fn draw(e: *Engine, logits: u64, rows: usize, first: u64, bits: u64, out: []u32) !void {
    if (e.structured) return e.drawWhole(logits, rows, first, bits, out);
    if (e.cur.sampling) |s| {
        if (e.discard_draws) { // a follower's verify rows: their columns to every rank, the draw itself rank 0's
            @memset(out[0..rows], 0);
            return e.smp.gather(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, e.smp.full.ptr);
        }
        try e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
        return e.checkGathers(); // the draw read rows the forward's gathers made
    }
    try @import("cuda_engine_pick.zig").greedyRows(e, logits, rows, out);
}
