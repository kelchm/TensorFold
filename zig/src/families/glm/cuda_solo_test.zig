//! Wider solo geometry and sizing without executing CUDA kernels.
const std = @import("std");
const solo = @import("cuda_prompt_solo.zig");
const state = @import("cuda_state.zig");
const engine = @import("cuda_engine.zig");
const cuts = @import("cuda_prompt_cuts.zig");
const memory = @import("cuda_solo_memory.zig");
const t = std.testing;
const C = state.prefill_rows;

test "a solo pass: four chunks preserve every interior keep mark and the final tail" {
    try t.expect(solo.takesChunks(0, 6 * C, 4 * C, &.{}));
    for ([_]u32{ C, 2 * C, 3 * C }) |mark| try t.expect(!solo.takesChunks(0, 6 * C, 4 * C, &.{mark}));
    try t.expect(solo.takesChunks(0, 6 * C, 4 * C, &.{ 4 * C, 5 * C }));
    try t.expect(solo.takesChunks(C, 6 * C, 4 * C, &.{C}));
    try t.expect(!solo.takesChunks(C + 1, 6 * C, 4 * C, &.{}));
    try t.expect(!solo.takesChunks(0, C + 1, 4 * C, &.{}));
    try t.expect(solo.takesChunks(0, C + 2, 4 * C, &.{}));
    try t.expectEqual(@as(usize, 4 * C), solo.rowsAt(C, 8 * C, 4 * C));
    try t.expectEqual(@as(usize, 3 * C + 1), solo.rowsAt(C, 4 * C + 1, 4 * C));
    try t.expectEqual(@as(usize, 4 * C), try cuts.stepRows(8 * C, &.{}, C, 4 * C + 1, 4 * C));
    try t.expectEqual(@as(usize, 2 * C), try cuts.stepRows(8 * C, &.{}, C, 2 * C + 1, 2 * C));
    try t.expectError(error.BadChunk, cuts.stepRows(8 * C, &.{2 * C}, 0, 4 * C + 1, 4 * C));
}

test "a solo pass: four chunks rebase at the same point as a chunk at a time" {
    for ([_]usize{ 2 * C, 4 * C }) |ring| for (0..9) |chunk| {
        const start = chunk * C;
        for ([_]usize{ C + 2, 2 * C, 3 * C, 4 * C }) |rows| {
            var base: usize = 0;
            var at = start;
            while (at < start + rows) : (at += C) {
                if (at + @min(C, start + rows - at) > base + ring) base = at;
            }
            const wide_base = if (start + rows > ring) start else 0;
            try t.expectEqual(rows <= ring and base == wide_base, solo.ringsAgree(0, ring, start, rows));
        }
    };
    try t.expect(solo.ringsAgree(0, 4 * C, 4 * C, 4 * C));
    try t.expect(!solo.ringsAgree(0, 4 * C, C, 4 * C));
    try t.expect(!solo.ringsAgree(C, 4 * C, 0, 4 * C));
}

test "a solo pass: followers refuse wide rows beyond buffers, rings or a request cut" {
    var seq: @import("cuda_engine_types.zig").Seq = undefined;
    seq.st.base = 0;
    seq.st.ring_rows = 4 * C;
    seq.cuts = .empty;
    var e: engine.Engine = undefined;
    e.cur = &seq;
    e.bp.rows = 4 * C;
    e.prompt_logprobs = null;
    try t.expect(solo.allowed(&e, 0, 4 * C));
    try t.expect(!solo.allowed(&e, C, 4 * C));
    try t.expect(!solo.allowed(&e, 1, 4 * C));
    try t.expect(!solo.allowed(&e, 0, 4 * C + 1));
    e.bp.rows = 2 * C;
    try t.expect(!solo.allowed(&e, 0, 4 * C));
    e.bp.rows = 4 * C;
    seq.st.ring_rows = 2 * C;
    try t.expect(!solo.allowed(&e, 0, 4 * C));
    seq.st.ring_rows = 4 * C;
    try seq.cuts.append(t.allocator, 2 * C);
    defer seq.cuts.deinit(t.allocator);
    try t.expect(!solo.allowed(&e, 0, 4 * C));
}

test "a solo pass: the leader uses the restored ring base and keeps marks inside four chunks" {
    var seq: @import("cuda_engine_types.zig").Seq = undefined;
    seq.st.base = 0;
    seq.st.ring_rows = 4 * C;
    seq.cuts = .empty;
    var e: engine.Engine = undefined;
    e.gpa = t.allocator;
    e.cur = &seq;
    e.bp.rows = 4 * C;
    e.prompt_logprobs = null;
    e.prompt_profs = @splat(null);
    e.images = null;
    e.snaps = .empty;
    defer e.snaps.deinit(t.allocator);
    try t.expect(solo.at(&e, 0, 8 * C, null, null));
    try t.expect(!solo.at(&e, 2 * C, 8 * C, null, null));
    try e.snaps.put(t.allocator, 1, .{ .lease = undefined, .len = 0, .pos = 2 * C, .mtp_len = 2 * C, .cur = &.{}, .bases = .{ 2 * C, 0 } });
    const from: engine.Resume = .{ .id = 1, .at = 2 * C };
    try t.expect(solo.at(&e, 2 * C, 8 * C, from, null));
    const Mark = struct {
        fn call(_: *anyopaque, _: u32) void {}
    };
    var ctx: u8 = 0;
    for ([_]u32{ 3 * C, 4 * C, 5 * C }) |mark| {
        try t.expect(!solo.at(&e, 2 * C, 8 * C, from, .{ .at = &.{mark}, .ctx = &ctx, .call = Mark.call }));
    }
}

test "a solo pass: exact startup plans charge EXL3 scratch, every ring and DFlash2 taps" {
    const weights = @import("cuda_weights.zig");
    var layers: [45]weights.Layer = undefined;
    var w: weights.Weights = .{ .gpa = t.allocator, .config = .{ .exl3 = true, .head_unit = 128 }, .share = undefined, .layers = &layers };
    w.share = try @import("cuda_shard.zig").Share.of(&w.config, 0, 2);
    for (&layers, 0..) |*l, i| {
        l.mixer = if (w.config.kind(@intCast(i)) == .mla) .{ .mla = undefined } else .{ .kda = undefined };
        if (l.mixer == .kda) l.mixer.kda.proj = .{ .bf16 = 0 };
    }
    const cap = 1 << 20;
    var prompt: [3]usize = undefined;
    var sequence: [3]usize = undefined;
    for ([_]usize{ C, 2 * C, 4 * C }, 0..) |width, i| {
        prompt[i] = state.Buffers.bytes(&w, width, true, cap, width);
        sequence[i] = try state.State.bytesFor(t.allocator, &w, cap, true, null, width);
        const o: engine.Options = .{ .capacity = cap, .mtp = true, .settings = .{ .solo = @intCast(width), .kv8 = true } };
        const n = try memory.bytes(t.allocator, &w, o);
        var drafted = o;
        drafted.drafter = true;
        try t.expectEqual(@import("dflash.zig").tapRows(width) * @import("dflash.zig").TAP_ROW, try memory.bytes(t.allocator, &w, drafted) - n);
        try memory.require(n, n);
        try t.expectError(error.GlmSoloMemory, memory.require(n, n - 1));
    }
    for ([_]bool{ false, true }) |shared_input| {
        const exl3 = @import("cuda_exl3.zig");
        const small = exl3.scratchBytes(4096, 4096, 1024, true, shared_input);
        const wide = exl3.scratchBytes(8192, 4096, 1024, true, shared_input);
        try t.expectEqual(2 * small[0], wide[0]);
    }
    try t.expectEqual(sequence[0], sequence[1]);
    try t.expectEqual(@as(usize, (state.mlaLayers(&w) + 1) * 2 * 4096 * 128 * 2), sequence[2] - sequence[1]);
    try t.expect(prompt[2] > prompt[1] and prompt[1] > prompt[0]);
}

test "a solo pass pairs (TF_GLM_PROMPT_PAIR=solo): startup plans charge the twin and index rings of both chunks" {
    const weights = @import("cuda_weights.zig");
    var layers: [45]weights.Layer = undefined;
    var w: weights.Weights = .{ .gpa = t.allocator, .config = .{ .exl3 = true, .head_unit = 128 }, .share = undefined, .layers = &layers };
    w.share = try @import("cuda_shard.zig").Share.of(&w.config, 0, 2);
    for (&layers, 0..) |*l, i| {
        l.mixer = if (w.config.kind(@intCast(i)) == .mla) .{ .mla = undefined } else .{ .kda = undefined };
        if (l.mixer == .kda) l.mixer.kda.proj = .{ .bf16 = 0 };
    }
    const cap = 1 << 20;
    const alone: engine.Options = .{ .capacity = cap, .mtp = true, .settings = .{ .solo = 2 * C } };
    var paired = alone;
    paired.settings.pair_solo = true;
    const twin = state.Buffers.twinBytes(&w, 2 * C, cap);
    const rings = (state.mlaLayers(&w) + 1) * 2 * 4096 * 128 * 2; // 4,096 more rows in every ring, the MTP head's too
    try t.expectEqual(twin + rings, try memory.bytes(t.allocator, &w, paired) - try memory.bytes(t.allocator, &w, alone));
    try t.expectEqual(try state.State.bytesFor(t.allocator, &w, cap, false, null, 4 * C), try state.State.bytesFor(t.allocator, &w, cap, false, null, paired.settings.ringWidth()));
    // the twin holds its own rows only: under the whole buffer, and twice the grid pair's but one logits block
    try t.expect(twin < state.Buffers.bytes(&w, 2 * C, true, cap, 2 * C));
    const logits = std.mem.alignForward(usize, state.prompt_heads * w.share.vocab.len() * 2, 256);
    try t.expectEqual(2 * (state.Buffers.twinBytes(&w, C, cap) - logits), twin - logits);
}
