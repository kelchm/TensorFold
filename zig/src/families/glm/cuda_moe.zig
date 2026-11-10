//! GLM-5.3-Flash's MoE on CUDA, a rank's share of the Python family's moe_block: top-8 plus shared, grouped NVFP4.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const experts = @import("cuda_experts.zig");
const exl3 = @import("cuda_exl3.zig");
const prof_mod = @import("cuda_prof.zig");
const projection = @import("cuda_proj.zig");

/// One MoE layer on a rank: router (bf16 [count][dims]) and fp32 bias, routed experts, shared [gate | up] and down.
pub const Layer = struct {
    router: u64,
    bias: u64,
    routed: cuda.experts.Layer, // NVFP4 experts, or only the shape (width, dims, limit) when `exl3` holds them
    exl3: ?exl3.Experts = null,
    count: usize,
    shared_gu: projection.Proj, // as the checkpoint stores them: bf16, block FP8 or the MTP's MXFP8
    shared_down: projection.Proj,
    shared: usize,
    top_k: usize,
    scale: f32,
    norm: bool,
};

/// The block's scratch for up to `rows` rows (forward.Buffers' MoE part).
pub const Scratch = struct {
    mlog: u64, // fp32 [rows][count]
    router_part: u64, // fp32 [router_ks][rows][count]
    pick: u64, // i32 [rows][top_k + 1]
    wts: u64, // fp32 [rows][top_k + 1]
    plan: cuda.grouped.Plan,
    eact: u64, // bf16 [rows (top_k + 1)][width]
    ey: u64, // [rows][top_k + 1][dims]: fp32, bf16 in a prompt chunk (bf16Slots)
    sgu: u64, // bf16 [rows][2 shared]
    sact: u64, // bf16 [rows][shared]
    sxs: u64, // fp32 [rows][shared / 64]
    sy: u64, // fp32 [rows][dims]
    mm_part: u64, // fp32 split-K partials of the shared expert's matmuls
    x3: exl3.Scratch = .{ .picks = 0, .xg = 0, .xu = 0, .z = 0 }, // EXL3 experts (eact holds their down inputs)
};

/// The kernels the block runs: the captured Triton set, the grouped NVFP4 experts, the plan, and glm_ops.
pub const Kernels = struct { set: glue.Set, experts: *const experts.Kernels, exl3: *const exl3.Kernels, put_slot: cuda.Function, prof: ?*prof_mod.Prof = null, exl3_b16: bool = false };

/// The slot rows' type: bf16 in an NVFP4 prompt chunk, and in an EXL3 one with TF_GLM_EXL3_BF16_SLOTS=1; else fp32.
pub fn bf16Slots(k: Kernels, l: Layer, prompt: bool) bool {
    if (l.exl3 == null) return prompt;
    return prompt and k.exl3_b16 and exl3.fused(l.routed.dims, l.routed.width);
}

/// forward.moe_block: out [rows][dims] fp32 partial; `prompt`: 64-pair items, the prompt kernels, bf16 slot rows.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, x, x_stride, 0, rows, prompt, out);
}

/// The block up to its fill: the router, the top-k pick, the plan and the routed experts into the slot rows.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const shared_id: i32 = @intCast(l.count);
    var stamp = prof_mod.span(k.prof, stream, .router);
    try glue.router(k.set, stream, x, x_stride, l.router, s.mlog, s.router_part, rows, d, l.count);
    try glue.select(k.set, stream, s.mlog, l.bias, s.pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm);
    stamp.end();
    stamp = prof_mod.span(k.prof, stream, .experts);
    defer stamp.end();
    if (k.prof) |p| p.mark(stream, .router);
    defer if (k.prof) |p| p.mark(stream, .routed);
    if (l.exl3) |w| return exl3.run(k.exl3, k.experts, stream, w, s.x3, s.plan, x, x_stride, s.pick, slots, rows, l.count, d, l.routed.width, l.routed.limit, prompt, bf16Slots(k, l, prompt), s.ey, s.eact);
    const plan: experts.Plan = .{ .route = s.plan, .slots = slots, .experts = l.count + 1, .prompt = prompt };
    try experts.route(k.experts, stream, s.pick, rows * slots, plan);
    try experts.run(k.experts, stream, .gate_up, l.routed, plan, x, x_stride, rows, s.eact, shared_id);
    try experts.run(k.experts, stream, if (prompt) .down else .down_f32, l.routed, plan, s.eact, l.routed.width, rows, s.ey, shared_id);
}

/// moe_block's fill of rows lo..hi: shared expert into the last slot, slot sum into out fp32 (any range, same bits).
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    const stamp = prof_mod.span(k.prof, stream, .shared_expert);
    defer stamp.end();
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const n = hi - lo;
    const bf16_slots = bf16Slots(k, l, prompt);
    const eb: usize = if (bf16_slots) 2 else 4;
    try projection.matmul(k.set, stream, l.shared_gu, x + lo * x_stride * 2, x_stride, s.sgu + lo * 2 * l.shared * 2, s.mm_part, n, 2 * l.shared, d, false);
    try glue.swiglu(k.set, stream, s.sgu + lo * 2 * l.shared * 2, s.sact + lo * l.shared * 2, s.sxs + lo * (l.shared / 64) * 4, l.routed.limit, n, l.shared);
    const act = s.sact + lo * l.shared * 2;
    // TF_GLM_SHARED_SLOT: shared-down straight into slot top_k, rounded once as put_slot rounds; else sy, then put_slot
    if (!try projection.matmulStrided(k.set, stream, l.shared_down, act, l.shared, s.ey + (lo * slots + l.top_k) * d * eb, slots * d, n, d, l.shared, !bf16_slots)) {
        try projection.matmul(k.set, stream, l.shared_down, act, l.shared, s.sy + lo * d * 4, s.mm_part, n, d, l.shared, true);
        var a: cuda.Args = .{};
        a.add(s.sy + lo * d * 4);
        a.add(s.ey + lo * slots * d * eb);
        for ([_]usize{ n, d, slots, l.top_k, @intFromBool(bf16_slots) }) |v| a.add(@as(c_int, @intCast(v)));
        try cuda.launch.launch(k.put_slot, .{ .grid = .{ .x = @intCast((n * d + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
    try glue.combine(k.set, stream, s.ey + lo * slots * d * eb, !bf16_slots, s.wts + lo * slots * 4, out, n, d, slots);
}

test "slot rows: bf16 in an NVFP4 prompt chunk, in an EXL3 one only with TF_GLM_EXL3_BF16_SLOTS, fp32 in decode" {
    const t = std.testing;
    var k: Kernels = undefined;
    var l: Layer = undefined;
    for ([_]bool{ false, true }) |opted| {
        k.exl3_b16 = opted;
        l.routed.dims = 4096;
        l.routed.width = 1024;
        l.exl3 = null;
        try t.expect(bf16Slots(k, l, true) and !bf16Slots(k, l, false));
        l.exl3 = std.mem.zeroes(exl3.Experts);
        try t.expectEqual(opted, bf16Slots(k, l, true));
        try t.expect(!bf16Slots(k, l, false));
        l.routed.width = 1000; // not the fused kernels' shape: the copied fp32 rows
        try t.expect(!bf16Slots(k, l, true));
    }
}
