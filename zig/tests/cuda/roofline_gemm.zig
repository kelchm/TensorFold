//! Achievable BF16 and tensorwide E4M3 GEMMs using the runtime's cuBLASLt loader and ABI.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const timing = @import("roofline_time.zig");
const lt = cuda.cublaslt;
const workspace_bytes = 64 << 20;

const Job = struct {
    plan: *const lt.Gemm,
    stream: cuda.Stream,
    x: u64,
    w: u64,
    out: u64,
    ws: u64,

    pub fn enqueue(j: *const Job) !void {
        const g = j.plan;
        const one: f32 = 1;
        const zero: f32 = 0;
        try supported(g.lt, g.lt.api.cublasLtMatmul(g.handle, g.desc, &one, j.w, g.a, j.x, g.b, &zero, j.out, g.c, j.out, g.c, &g.algo, j.ws, workspace_bytes, j.stream.handle));
    }
};

pub fn run(gpu: check.Gpu, seconds: f64, report: *timing.Report) !void {
    var library = lt.Library.open() catch |err| {
        report.skip("gemm", @errorName(err));
        return;
    };
    defer library.close();
    std.debug.print("ROOFLINE cublaslt-version {d} version workspace={d} heuristic_candidates=32 output=bf16 accumulate=fp32\n", .{ library.api.cublasLtGetVersion(), workspace_bytes });
    var handle: lt.Handle = null;
    try library.check(library.api.cublasLtCreate(&handle), "roofline cublasLtCreate");
    defer _ = library.api.cublasLtDestroy(handle);
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var ws = try cuda.DeviceBuffer.alloc(gpu.d, workspace_bytes);
    defer ws.free();
    for ([_]bool{ false, true }) |fp8| {
        const name = if (fp8) "gemm-fp8-e4m3" else "gemm-bf16";
        if (fp8 and try gpu.ctx.capability() < 89) {
            report.skip(name, "FP8 requires sm_89 or later");
            continue;
        }
        for ([_][3]u64{ .{ 4096, 4096, 4096 }, .{ 8192, 8192, 8192 }, .{ 2048, 4096, 4096 }, .{ 2048, 8192, 4096 }, .{ 2048, 16384, 4096 } }) |shape| {
            bench(gpu, &library, handle, stream, ws, shape, fp8, seconds, report) catch |err| switch (err) {
                error.RooflineUnsupported, error.NoAlgorithm => {
                    var buf: [96]u8 = undefined;
                    report.skip(name, try std.fmt.bufPrint(&buf, "M={d} N={d} K={d} reason={s}", .{ shape[0], shape[1], shape[2], @errorName(err) }));
                },
                else => return err,
            };
        }
    }
}

fn bench(gpu: check.Gpu, library: *const lt.Library, handle: lt.Handle, stream: cuda.Stream, ws: cuda.DeviceBuffer, shape: [3]u64, fp8: bool, seconds: f64, report: *timing.Report) !void {
    const m = shape[0];
    const n = shape[1];
    const k = shape[2];
    var x = try input(gpu, m * k, fp8, 17);
    defer x.free();
    var w = try input(gpu, n * k, fp8, 91);
    defer w.free();
    var out = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
    defer out.free();
    var plan = try init(library, handle, shape, fp8);
    defer plan.deinit();
    var job: Job = .{ .plan = &plan, .stream = stream, .x = x.ptr, .w = w.ptr, .out = out.ptr, .ws = ws.ptr };
    var candidates: [32]lt.Heuristic = undefined;
    const count = try algorithms(library, plan, &candidates);
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    var best: f64 = std.math.inf(f64);
    var selected: ?lt.Heuristic = null;
    var selected_index: usize = 0;
    for (candidates[0..count], 0..) |candidate, index| {
        if (candidate.state != 0 or candidate.workspace_size > workspace_bytes) continue;
        plan.algo = candidate.algo;
        plan.workspace_need = candidate.workspace_size;
        job.enqueue() catch |err| switch (err) {
            error.RooflineUnsupported => continue,
            else => return err,
        };
        try stream.synchronize();
        var samples: [3]f64 = undefined;
        for (&samples) |*ms| {
            try start.record(stream);
            for (0..3) |_| try job.enqueue();
            try end.record(stream);
            try end.synchronize();
            ms.* = try start.elapsedMs(end);
        }
        const ms = check.median(&samples);
        if (ms < best) {
            best = ms;
            selected = candidate;
            selected_index = index;
        }
    }
    const chosen = selected orelse return error.NoAlgorithm;
    plan.algo = chosen.algo;
    plan.workspace_need = chosen.workspace_size;
    const stats = try timing.measure(gpu, stream, seconds, &job);
    try validate(out, shape);
    var buf: [256]u8 = undefined;
    const details = try std.fmt.bufPrint(&buf, "M={d} N={d} K={d} output=bf16 accumulate=fp32 scales=1 workspace={d} heuristic={d}/{d} warm_buffers=1", .{ m, n, k, chosen.workspace_size, selected_index, count });
    report.add(if (fp8) "gemm-fp8-e4m3" else "gemm-bf16", "TFLOPS", @as(f64, @floatFromInt(2 * m * n * k)) / 1e9, stats, details);
}

/// TN column-major view D^T[n,m] = W[n,k] . X^T[k,m], with default tensorwide scales of one.
fn init(library: *const lt.Library, handle: lt.Handle, shape: [3]u64, fp8: bool) !lt.Gemm {
    const api = library.api;
    var g: lt.Gemm = undefined;
    g.lt = library;
    g.handle = handle;
    try library.check(api.cublasLtMatmulDescCreate(&g.desc, lt.compute_32f, .f32), "roofline descriptor");
    errdefer _ = api.cublasLtMatmulDescDestroy(g.desc);
    const trans: c_int = 1;
    try library.check(api.cublasLtMatmulDescSetAttribute(g.desc, lt.desc_transa, &trans, @sizeOf(c_int)), "roofline TN");
    // cudaDataType_t CUDA_R_8F_E4M3=28; keep the benchmark's extra format outside the engine's BF16-only API.
    const Layout = *const fn (*lt.Layout, c_int, u64, u64, i64) callconv(.c) lt.Status;
    const layout: Layout = @ptrCast(api.cublasLtMatrixLayoutCreate);
    const dtype: c_int = if (fp8) 28 else 14;
    const m = shape[0];
    const n = shape[1];
    const k = shape[2];
    try supported(library, layout(&g.a, dtype, k, n, @intCast(k)));
    errdefer _ = api.cublasLtMatrixLayoutDestroy(g.a);
    try supported(library, layout(&g.b, dtype, k, m, @intCast(k)));
    errdefer _ = api.cublasLtMatrixLayoutDestroy(g.b);
    try library.check(api.cublasLtMatrixLayoutCreate(&g.c, .bf16, n, m, @intCast(n)), "roofline output layout");
    g.workspace_need = 0;
    return g;
}

fn algorithms(library: *const lt.Library, plan: lt.Gemm, found: []lt.Heuristic) !usize {
    const api = library.api;
    var pref: lt.Preference = null;
    try library.check(api.cublasLtMatmulPreferenceCreate(&pref), "roofline preference");
    defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
    const limit: u64 = workspace_bytes;
    try library.check(api.cublasLtMatmulPreferenceSetAttribute(pref, lt.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "roofline workspace limit");
    var count: c_int = 0;
    try supported(library, api.cublasLtMatmulAlgoGetHeuristic(plan.handle, plan.desc, plan.a, plan.b, plan.c, plan.c, pref, @intCast(found.len), found.ptr, &count));
    if (count <= 0) return error.NoAlgorithm;
    return @intCast(count);
}

fn supported(library: *const lt.Library, status: lt.Status) !void {
    if (status == 7 or status == 8 or status == 15) return error.RooflineUnsupported;
    try library.check(status, "roofline cuBLASLt");
}

fn code(index: u64, seed: u32) u8 {
    var v: u32 = @truncate(index);
    v ^= seed;
    v ^= v >> 16;
    v *%= 0x7feb352d;
    v ^= v >> 15;
    return @as(u8, @truncate(v & 7)) | @as(u8, @truncate((v >> 7) & 0x80));
}

fn input(gpu: check.Gpu, count: u64, fp8: bool, seed: u32) !cuda.DeviceBuffer {
    const host = try gpu.gpa.alloc(u8, count * @as(usize, if (fp8) 1 else 2));
    defer gpu.gpa.free(host);
    for (0..count) |i| {
        const c = code(i, seed);
        if (fp8) host[i] = 0x20 | c else std.mem.writeInt(u16, host[i * 2 ..][0..2], 0x3e00 | (@as(u16, c & 7) << 4) | (@as(u16, c & 0x80) << 8), .little);
    }
    return cuda.DeviceBuffer.fromHost(gpu.d, host);
}

fn value(index: u64, seed: u32) f64 {
    const c = code(index, seed);
    const v = @as(f64, @floatFromInt(8 + @as(u32, c & 7))) / 64;
    return if (c & 0x80 == 0) v else -v;
}

/// Products are multiples of 1/4096 and the exact sums fit FP32; the final BF16 rounding is independently checked.
fn validate(out: cuda.DeviceBuffer, shape: [3]u64) !void {
    const m = shape[0];
    const n = shape[1];
    const k = shape[2];
    for ([_]u64{ 0, m / 2, m - 1 }) |row| for ([_]u64{ 0, n / 2, n - 1 }) |col| {
        var sum: f64 = 0;
        for (0..k) |t| sum += value(row * k + t, 17) * value(col * k + t, 91);
        const bits: u32 = @bitCast(@as(f32, @floatCast(sum)));
        const want: u16 = @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
        var got: u16 = 0;
        try out.download((row * n + col) * 2, std.mem.asBytes(&got));
        try check.expect(got == want, "roofline GEMM [{d},{d}]: {x} != {x} (FP64)", .{ row, col, got, want });
    };
}
