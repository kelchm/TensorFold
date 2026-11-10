//! One-GPU packed prompt projections at GLM-5.3-Flash TP=2 shapes against tuned cuBLASLt BF16 (`kernel=`: GLM's pick).
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const timing = @import("roofline_time.zig");
const gemm = @import("roofline_gemm.zig");
const shapes = @import("glm_prompt_mm_shapes.zig");
const data = @import("glm_prompt_mm_data.zig");
const peak = 126.5;

const PackedJob = struct {
    set: glm.glue.Set,
    stream: cuda.Stream,
    shape: shapes.Shape,
    x: u64,
    w: u64,
    out: u64,
    m: usize,

    pub fn enqueue(j: *const PackedJob) !void {
        const sh = j.shape;
        try glm.b16.matmul(j.set, j.stream, j.x, sh.ldx(), j.w, j.out, 0, j.m, sh.n, sh.k, sh.wide);
    }
};

const Sum = struct {
    work: f64 = 0,
    best: [2]f64 = .{ 0, 0 },
    median: [2]f64 = .{ 0, 0 },
    layers: usize = 0,

    fn add(s: *Sum, sh: shapes.Shape, m: usize, quantized: timing.Stats, lt: timing.Stats) void {
        const count: f64 = @floatFromInt(sh.layers);
        s.layers = sh.layers;
        s.work += sh.work(m) * count;
        for ([_]timing.Stats{ quantized, lt }, 0..) |v, i| {
            s.best[i] += v.best_ms * count;
            s.median[i] += v.median_ms * count;
        }
    }

    fn print(s: Sum, name: []const u8, m: usize) void {
        const count: f64 = @floatFromInt(s.layers);
        for ([_][]const u8{ "packed", "cublaslt-bf16" }, 0..) |path, i| {
            std.debug.print("GLM_PROMPT_MM sum={s} M={d} path={s} layers={d} best_ms={d:.6} median_ms={d:.6} per_layer_best_ms={d:.6} per_layer_median_ms={d:.6} best_TFLOPS={d:.3} median_TFLOPS={d:.3} best_peak_pct={d:.2} median_peak_pct={d:.2}\n", .{ name, m, path, s.layers, s.best[i], s.median[i], s.best[i] / count, s.median[i] / count, s.work / s.best[i], s.work / s.median[i], 100 * s.work / s.best[i] / peak, 100 * s.work / s.median[i] / peak });
        }
        std.debug.print("GLM_PROMPT_MM sum-ratio={s} M={d} packed_over_cublaslt_best={d:.3} packed_over_cublaslt_median={d:.3}\n", .{ name, m, s.best[0] / s.best[1], s.median[0] / s.median[1] });
    }
};

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len > 2) return error.GlmPromptMmArguments;
    const bits = if (rest.len >= 1) try std.fmt.parseInt(u8, rest[0], 10) else 4;
    if (bits != 4 and bits != 6 and bits != 8) return error.GlmPromptMmBits;
    const seconds = if (rest.len >= 2) try std.fmt.parseFloat(f64, rest[1]) else 1;
    if (!std.math.isFinite(seconds) or seconds <= 0 or seconds > 3600) return error.GlmPromptMmSeconds;
    const list = try shapes.list();
    shapes.print(&list);
    try @import("roofline_facts.zig").print(gpu);
    var library = try cuda.cublaslt.Library.open();
    defer library.close();
    var handle: cuda.cublaslt.Handle = null;
    try library.check(library.api.cublasLtCreate(&handle), "glm-prompt-mm cuBLASLt handle");
    defer _ = library.api.cublasLtDestroy(handle);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var mm_mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
    defer mm_mod.unload();
    const mk = try glm.mm.Kernels.load(mm_mod);
    var ws = try cuda.DeviceBuffer.alloc(gpu.d, gemm.workspace_bytes);
    defer ws.free();
    var total: [shapes.rows.len]Sum = @splat(.{});
    var groups: [shapes.rows.len][4]Sum = @splat(@splat(.{}));
    const kernel = mk.prompt_symbols[(bits - 4) / 2];
    std.debug.print("GLM_PROMPT_MM config bits={d} kernel={s} seconds_per_path={d:.3} rows=2048,4096,8192 device=0 tp_shape=2 warmup=3 min_repeats=7 graph_batch_target_ms=10 peak_bf16_TFLOPS={d:.1} cublaslt_version={d} workspace_limit={d} pdl={d}\n", .{ bits, kernel, seconds, peak, library.api.cublasLtGetVersion(), gemm.workspace_bytes, @intFromBool(mk.pdl) });
    std.debug.print("GLM_PROMPT_MM scope=packed-backbone-linears inputs=bf16 activation_conversion=none output=served-type accumulate=fp32 fixture=dyadic-qpack weights_and_input_setup=untimed cache=warm-reused-buffers\n", .{});
    std.debug.print("GLM_PROMPT_MM excluded=head-last-row,mtp,routed-experts,routers,mla-absorb-expand,nonlinearities,communication indexer_qb=enabled sums=modeled-serial-45-layer-backbone ratio=packed-ms/cublaslt-ms lower_ratio_is_faster=1 fp8=not-measured\n", .{});
    for (list) |sh| {
        var weights = try data.Weights.init(gpu, &mk, stream, sh, bits);
        defer weights.deinit();
        const ptrs = [_]u64{weights.quantized.ptr};
        const widths = [_]u8{bits};
        const packs: glm.mm.Packed = .{ .ptrs = &ptrs, .bits = &widths };
        // b16.matmul takes this branch before reading glue; match Forward.set's tiled packed prompt selection.
        const set: glm.glue.Set = .{ .glue = undefined, .mm = &mk, .packs = &packs, .prompt = true };
        for (shapes.rows, 0..) |m, ri| {
            const pair = try bench(gpu, &library, handle, stream, ws, sh, m, seconds, bits, kernel, set, weights);
            total[ri].add(sh, m, pair[0], pair[1]);
            groups[ri][@backingInt(sh.group)].add(sh, m, pair[0], pair[1]);
        }
    }
    for (shapes.rows, 0..) |m, ri| {
        inline for (std.meta.tags(shapes.Group)) |group| groups[ri][@backingInt(group)].print(@tagName(group), m);
        total[ri].layers = (glm.config.Config{}).layers;
        total[ri].print("backbone", m);
    }
    check.pass("glm-prompt-mm-bench: all 39 packed prompt cases agree with dequantized cuBLASLt", .{});
}

fn bench(gpu: check.Gpu, library: *const cuda.cublaslt.Library, handle: cuda.cublaslt.Handle, stream: cuda.Stream, ws: cuda.DeviceBuffer, sh: shapes.Shape, m: usize, seconds: f64, bits: u8, kernel: []const u8, set: glm.glue.Set, weights: data.Weights) ![2]timing.Stats {
    var x = try data.input(gpu, sh, m);
    defer x.free();
    const bytes = m * sh.n * @as(usize, if (sh.wide) 4 else 2);
    var out = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer out.free();
    var reference = try cuda.DeviceBuffer.alloc(gpu.d, bytes);
    defer reference.free();
    defer stream.synchronize() catch @panic("glm-prompt-mm retains buffers until GPU completion");
    var plan = try gemm.initLinear(library, handle, .{ m, sh.n, sh.k }, false, sh.ldx(), sh.wide);
    defer plan.deinit();
    const input = x.ptr + sh.offset * 2;
    var lt_job: gemm.Job = .{ .plan = &plan, .stream = stream, .x = input, .w = weights.bf16.ptr, .out = reference.ptr, .ws = ws.ptr };
    var packed_job: PackedJob = .{ .set = set, .stream = stream, .shape = sh, .x = input, .w = weights.quantized.ptr, .out = out.ptr, .m = m };
    const selected = try gemm.tune(gpu, &lt_job);
    try out.fill8(0xff, stream.handle);
    try reference.fill8(0xff, stream.handle);
    try lt_job.enqueue();
    try packed_job.enqueue();
    try stream.synchronize();
    try data.validate(sh, m, bits, out, reference);
    const quantized = try timing.measure(gpu, stream, seconds, &packed_job);
    const lt = try timing.measure(gpu, stream, seconds, &lt_job);
    try data.validate(sh, m, bits, out, reference);
    for ([_]timing.Stats{ quantized, lt }, [_][]const u8{ "packed", "cublaslt-bf16" }) |s, path| {
        const work = sh.work(m);
        std.debug.print("GLM_PROMPT_MM result={s} M={d} N={d} K={d} layers={d} bits={d} path={s} output={s} best_ms={d:.6} median_ms={d:.6} best_TFLOPS={d:.3} median_TFLOPS={d:.3} best_peak_pct={d:.2} median_peak_pct={d:.2} packed_over_cublaslt_best={d:.3} packed_over_cublaslt_median={d:.3} repeats={d} batch={d} gpu_ms={d:.1} workspace={d} heuristic={d}/{d} kernel={s}\n", .{ sh.name, m, sh.n, sh.k, sh.layers, bits, path, if (sh.wide) "fp32" else "bf16", s.best_ms, s.median_ms, s.best(work), s.middle(work), 100 * s.best(work) / peak, 100 * s.middle(work) / peak, quantized.best_ms / lt.best_ms, quantized.median_ms / lt.median_ms, s.repeats, s.batch, s.gpu_ms, if (std.mem.eql(u8, path, "packed")) @as(usize, 0) else plan.workspace_need, selected.index, selected.count, if (std.mem.eql(u8, path, "packed")) kernel else "cublaslt" });
    }
    return .{ quantized, lt };
}
