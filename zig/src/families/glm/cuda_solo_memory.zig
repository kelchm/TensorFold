//! The wider solo workspace is admitted before its allocations, from the same plans that allocate it.
const std = @import("std");
const state = @import("cuda_state.zig");
const types = @import("cuda_engine_types.zig");
const dflash = @import("dflash.zig");

/// Remaining engine allocations after weights, packs and pieces (a pair's twin too); DFlash2's taps precede its load.
pub fn bytes(gpa: std.mem.Allocator, w: anytype, o: types.Options) !usize {
    const width: usize = o.settings.solo;
    const cap = o.capacity;
    const table: ?usize = if (o.settings.pages) (cap + state.page_rows - 1) / state.page_rows else null;
    const head = o.mtp and !o.settings.dflash_only;
    const pair = o.settings.pair and (width <= state.prefill_rows or o.settings.pair_solo) and w.share.world > 1;
    const joint = pair and o.settings.pair_experts;
    var n = try state.State.bytesFor(gpa, w, cap, o.settings.kv8, table, head, o.settings.ringWidth());
    n += state.Buffers.bytesOf(w, width, true, cap, width, true, joint);
    if (pair) n += state.Buffers.twinBytes(w, width, cap, joint);
    const decode = state.Buffers.bytes(w, state.window_rows, false, cap, state.window_rows);
    n += if (o.settings.dflash_only) decode else 2 * decode;
    if (o.mtp) n += state.Buffers.bytesOf(w, types.shared_rows, false, cap, state.window_rows, true, false);
    if (o.drafter) n += dflash.tapRows(width) * dflash.TAP_ROW;
    return n;
}

pub fn require(need: usize, room: ?u64) !void {
    if (need > (room orelse return error.GlmSoloMemory)) return error.GlmSoloMemory;
}

pub fn check(e: anytype, ctx: anytype, o: types.Options) !void {
    if (o.settings.solo <= 4096 and !o.settings.pair_solo) return;
    var effective = o;
    effective.settings.pages = e.settings.pages;
    const need = try bytes(e.gpa, &e.w, effective);
    const room = if (o.room) |r| r.read(r.ctx) else (try ctx.memInfo()).free;
    require(need, room) catch |err| {
        std.log.err("TF_GLM_CHUNK_SOLO={d}{s}: remaining prompt, sequence and decode buffers need {d} bytes; startup budget has {?d} bytes", .{ o.settings.solo, if (o.settings.pair_solo) " with TF_GLM_PROMPT_PAIR=solo" else "", need, room });
        return err;
    };
}

test "a solo pass: insufficient or unreadable startup room is refused" {
    try require(8192, 8192);
    try std.testing.expectError(error.GlmSoloMemory, require(8192, 8191));
    try std.testing.expectError(error.GlmSoloMemory, require(8192, null));
}
