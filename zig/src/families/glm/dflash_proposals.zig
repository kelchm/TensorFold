//! Proposal membership and folded per-stream commits travel identically in Kind.block's flags to every CUDA rank.
const std = @import("std");
const dflash = @import("dflash.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Req = dflash.Req;

pub fn active(reqs: []const Req, out: []Req) []const Req {
    var n: usize = 0;
    for (reqs) |q| if (q.propose) {
        out[n] = q;
        n += 1;
    };
    return out[0..n];
}

pub fn mask(reqs: []const Req) u32 {
    var bits: u32 = 0;
    for (reqs, 0..) |q, i| if (q.propose) {
        bits |= @as(u32, 1) << @intCast(i);
    };
    return bits;
}

pub fn enabled(bits: u32, i: usize) bool {
    return bits & (@as(u32, 1) << @intCast(i)) != 0;
}

/// Pre-send validation matching Forward.commit so a bad commit fails on rank 0 before any wire broadcast.
pub fn checkCommit(e: *const Engine, slot: u32, keep: usize) !void {
    if (slot >= e.seqs.items.len) return error.UnknownSlot;
    const st = &(e.seqs.items[slot] orelse return error.UnknownSlot).st;
    if (keep == 0 or keep > st.pending or (st.prompt and keep != st.pending)) return error.BadCommit;
    if (!st.prompt and st.kda_layers > 0) {
        const cur = st.cur[0];
        for (st.cur) |c| if (c != cur) return error.KdaLayersOutOfStep;
    }
}

/// Validates every stream and folded commit precondition on rank 0, encodes `buf`, and returns Kind.block's flags.
pub fn prepareBlock(e: *const Engine, reqs: []const Req, commit_mask: u16, buf: *[3 * dflash.MAX_S]u32) !u32 {
    comptime std.debug.assert(dflash.MAX_S <= 16);
    if (reqs.len == 0 or reqs.len > dflash.MAX_S) return error.TooManyStreams;
    if (reqs.len < 16 and (commit_mask >> @intCast(reqs.len)) != 0) return error.BadStep;
    if (e.drafter == null) return error.NoDrafter;
    for (reqs, 0..) |q, i| {
        if (q.slot >= e.seqs.items.len or e.seqs.items[q.slot] == null) return error.UnknownSlot;
        for (reqs[0..i]) |prev| if (prev.slot == q.slot) return error.SlotTwice;
        if (q.propose and q.anchor >= e.c.vocab) return error.BadTokenId;
        const n32 = std.math.cast(u32, q.n) orelse return error.BadStep;
        if ((commit_mask & (@as(u16, 1) << @intCast(i))) != 0) try checkCommit(e, q.slot, q.n);
        buf[3 * i ..][0..3].* = .{ q.slot, n32, q.anchor };
    }
    return mask(reqs) | (@as(u32, commit_mask) << 16);
}

/// Follower's Kind.block handler: decodes commit_mask/propose bits, commits flagged slots, then runs draftBlock.
pub fn applyBlock(e: *Engine, depth: u32, flags: u32, ids: []const u32) !void {
    comptime std.debug.assert(dflash.MAX_S <= 16);
    const k = depth;
    const commit_mask: u16 = @intCast(flags >> 16);
    if (k == 0 or k > dflash.MAX_S or ids.len != 3 * k) return error.BadStep;
    if (k < 16 and (commit_mask >> @intCast(k)) != 0) return error.BadStep;
    var reqs: [dflash.MAX_S]Req = undefined;
    for (reqs[0..k], 0..) |*q, i| {
        q.* = .{ .slot = ids[3 * i], .pos = 0, .n = ids[3 * i + 1], .anchor = ids[3 * i + 2], .propose = enabled(flags, i) };
        for (reqs[0..i]) |prev| if (prev.slot == q.slot) return error.SlotTwice;
        if (q.propose and q.anchor >= e.c.vocab) return error.BadTokenId;
    }
    try keepAndBlock(e, reqs[0..k], commit_mask);
}

pub fn keepAndBlock(e: *Engine, reqs: []Req, commit_mask: u16) !void {
    for (reqs, 0..) |q, i| if ((commit_mask & (@as(u16, 1) << @intCast(i))) != 0) {
        try e.bind(q.slot);
        try e.keep(q.n);
    };
    try e.draftBlock(reqs);
}
