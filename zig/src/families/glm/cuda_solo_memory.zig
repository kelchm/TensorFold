//! The wider solo workspace is admitted before its allocations, from the same plans that allocate it.
const std = @import("std");
const state = @import("cuda_state.zig");
const types = @import("cuda_engine_types.zig");
const dflash = @import("dflash.zig");

/// Remaining engine allocations after weights, packs and pieces (a pair's twin too); DFlash2's taps precede its load.
pub fn bytes(gpa: std.mem.Allocator, w: anytype, o: types.Options) !usize {
    const width: usize = o.settings.solo;
    const cap = o.capacity;
    const table: ?usize = if (o.settings.pages) @import("cuda_pages.zig").tableLen(cap) else null;
    var n = try state.State.bytesFor(gpa, w, cap, o.settings.kv8, table, o.settings.ringWidth());
    n += state.Buffers.bytes(w, width, true, cap, width);
    if (o.settings.pair_solo and w.share.world > 1) n += state.Buffers.twinBytes(w, width, cap); // prompt_pair.wanted
    n += 2 * state.Buffers.bytes(w, state.window_rows, false, cap, state.window_rows);
    if (o.mtp) n += state.Buffers.bytes(w, @import("cuda_engine.zig").shared_rows, false, cap, state.window_rows);
    if (o.drafter) n += dflash.tapRows(width) * dflash.TAP_ROW;
    return n;
}

pub fn require(need: usize, room: ?u64) !void {
    if (need > (room orelse return error.GlmSoloMemory)) return error.GlmSoloMemory;
}

pub fn check(e: anytype, ctx: anytype, o: types.Options) !void {
    if (o.settings.solo <= 4096 and !o.settings.pair_solo) return;
    const need = try bytes(e.gpa, &e.w, o);
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
