//! TF_GLM_ROUTER_FUSE on one GPU: the K slices and slice_sum against tf_glm_router_fused, A-B-A-B, same bytes.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// (rows, experts): the MoE router and the index gate at a 4,096-row rank share, a TP=1 chunk, a wave's worth of rows.
const shapes = [_][2]usize{ .{ 4096, 288 }, .{ 4096, 128 }, .{ 8192, 288 }, .{ 8192, 128 }, .{ 1024, 288 } };
const d = 4096;
const reps = 50;
/// A paired 8,192-row forward's calls at 4,096 rows a rank: 84 MoE routers, 22 index gates.
const moe_calls = 84;
const gate_calls = 22;

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    var slices = try glm.glue.Kernels.load(gm);
    slices.router_fuse = false;
    var fused = slices;
    fused.router_fuse = true;
    const sets = [2]glm.glue.Set{ .{ .glue = &slices, .prompt = true }, .{ .glue = &fused, .prompt = true } };
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var t0 = try cuda.Event.init(gpu.d, true);
    defer t0.deinit();
    var t1 = try cuda.Event.init(gpu.d, true);
    defer t1.deinit();
    var pair = [2]f64{ 0, 0 };
    for (shapes) |sh| {
        const m = sh[0];
        const ne = sh[1];
        const host_x = try a.alloc(u16, m * d);
        defer a.free(host_x);
        for (host_x) |*v| v.* = @truncate(@as(u32, @bitCast(rng.float(f32) * 2 - 1)) >> 16);
        const host_w = try a.alloc(u16, ne * d);
        defer a.free(host_w);
        for (host_w) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) * 2 - 1) * 0.02)) >> 16);
        var x = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_x));
        defer x.free();
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host_w));
        defer w.free();
        var part = try cuda.DeviceBuffer.alloc(gpu.d, glm.glue.router_ks * m * ne * 4);
        defer part.free();
        var outs: [2]cuda.DeviceBuffer = undefined;
        for (&outs) |*o| o.* = try cuda.DeviceBuffer.alloc(gpu.d, m * ne * 4);
        defer for (&outs) |*o| o.free();
        try check.expect(glm.glue.routerFused(sets[1], x.ptr, d, m, ne), "glm-router-bench {d} x {d}: the fused path takes it", .{ m, ne });
        var ms = [2]f64{ std.math.inf(f64), std.math.inf(f64) };
        for (0..4) |round| for (0..2) |which| {
            try t0.record(s);
            for (0..reps) |_| try glm.glue.router(sets[which], s, x.ptr, d, w.ptr, outs[which].ptr, part.ptr, m, d, ne);
            try t1.record(s);
            try t1.synchronize();
            if (round > 0) ms[which] = @min(ms[which], @as(f64, try cuda.Event.elapsedMs(t0, t1)) / reps);
        };
        const got = try a.alloc(u8, m * ne * 4);
        defer a.free(got);
        const want = try a.alloc(u8, m * ne * 4);
        defer a.free(want);
        try outs[0].download(0, want);
        try outs[1].download(0, got);
        try check.sameBytes("glm-router-bench: fused logits equal the slices'", got, want);
        std.debug.print("RESULT glm-router-bench {d} rows x {d} experts: slices+sum {d:.3} ms, fused {d:.3} ms ({d:.2}x)\n", .{ m, ne, ms[0], ms[1], ms[0] / ms[1] });
        if (m == 4096) for (&pair, ms) |*p, v| {
            p.* += v * @as(f64, if (ne == 288) moe_calls else gate_calls);
        };
    }
    std.debug.print("RESULT glm-router-bench a paired forward (84 x 288, 22 x 128 at 4,096 rows): slices+sum {d:.1} ms, fused {d:.1} ms\n", .{ pair[0], pair[1] });
    check.pass("glm-router-bench: fused router bytes equal glue.router's slices", .{});
}
