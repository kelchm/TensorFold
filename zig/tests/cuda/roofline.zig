//! Synthetic ceilings on device 0, independent of checkpoints and engine execution.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const timing = @import("roofline_time.zig");
const options = @import("roofline_options");
const Gpu = check.Gpu;
const rounds: u32 = 4096;
const threads = 256;

fn Blob(comptime name: []const u8) type {
    return struct {
        const bytes align(16) = @embedFile(name).*;
    };
}

fn image(comptime name: []const u8) []const u8 {
    return if (options.roofline_images) &Blob(name).bytes else &.{};
}

const KernelJob = struct {
    function: cuda.Function,
    cfg: cuda.Config,
    stream: cuda.Stream,
    args: cuda.Args,

    pub fn enqueue(j: *KernelJob) !void {
        try cuda.launch.launch(j.function, j.cfg, j.stream, &j.args);
    }
};

const CopyJob = struct {
    pub const graph_batches = false;
    stream: cuda.Stream,
    src: u64,
    dst: u64,
    bytes: usize,

    pub fn enqueue(j: *CopyJob) !void {
        try j.stream.d.check(j.stream.d.api.cuMemcpyDtoD_v2(j.dst, j.src, j.bytes), "roofline cuMemcpyDtoD");
    }
};

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len > 1) return error.RooflineArguments;
    const seconds = if (rest.len == 1) try std.fmt.parseFloat(f64, rest[0]) else 1;
    if (!std.math.isFinite(seconds) or seconds <= 0 or seconds > 3600) return error.RooflineSeconds;
    var report: timing.Report = .{};
    const sm = try gpu.ctx.capability();
    const sms: u32 = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const l2: usize = @intCast(try gpu.ctx.attribute(.l2_cache_size));
    try @import("roofline_facts.zig").print(gpu);
    std.debug.print("ROOFLINE config {d:.3} seconds-per-test warmup=3 min_repeats=7 graph_batch_target_ms=10 device=0\n", .{seconds});
    if (image("roofline_image").len == 0 or sm < 80) {
        report.skip("kernels", "build tf-cuda-test with -Dnvcc or -Dfatbins for sm_80 or later");
    } else {
        var module = try cuda.Module.load(gpu.d, image("roofline_image"));
        defer module.unload();
        var stream = try cuda.Stream.init(gpu.d, false);
        defer stream.deinit();
        try memory(gpu, module, stream, sms, l2, seconds, &report);
        try compute(gpu, module, stream, sms, sm, seconds, &report);
        if ((sm == 120 or sm == 121) and image("roofline_low_image").len > 0) {
            var handle: cuda.abi.Module = null;
            const status = gpu.d.api.cuModuleLoadData(&handle, image("roofline_low_image").ptr);
            if (status == 209 or status == 801) {
                report.skip("mma-fp4-fp6", "driver has no compatible architecture-specific image");
                return finish(gpu, seconds, &report);
            }
            try gpu.d.check(status, "roofline low-precision image");
            var low: cuda.Module = .{ .d = gpu.d, .handle = handle };
            defer low.unload();
            try computeLow(gpu, low, stream, sms, seconds, &report);
        } else report.skip("mma-fp4-fp6", "requires a separately built sm_120a or sm_121a image");
    }
    try finish(gpu, seconds, &report);
}

fn finish(gpu: Gpu, seconds: f64, report: *timing.Report) !void {
    try @import("roofline_gemm.zig").run(gpu, seconds, report);
    report.summary();
}

fn memory(gpu: Gpu, module: cuda.Module, stream: cuda.Stream, sms: u32, l2: usize, seconds: f64, report: *timing.Report) !void {
    const cfg: cuda.Config = .{ .grid = .{ .x = sms * 8 }, .block = .{ .x = threads } };
    var sink = try cuda.DeviceBuffer.alloc(gpu.d, cfg.grid.x * threads / 32 * 16);
    defer sink.free();
    const read = try module.function("tf_roof_read");
    const write = try module.function("tf_roof_write");
    const copy = try module.function("tf_roof_copy");
    var sizes: [9]usize = undefined;
    sizes[0] = if (l2 >= 4096) std.math.floorPowerOfTwo(usize, l2 / 4) else 0;
    for (sizes[1..], 0..) |*size, i| size.* = @as(usize, 32 << 20) << @intCast(i);
    for (sizes, 0..) |bytes, si| {
        var skip_buf: [96]u8 = undefined;
        if (bytes == 0) {
            report.skip("l2", "driver reports no L2 capacity");
            continue;
        }
        var src = cuda.DeviceBuffer.alloc(gpu.d, bytes) catch |err| switch (err) {
            error.OutOfMemory, error.OutOfDeviceMemory => {
                report.skip("memory-size", try std.fmt.bufPrint(&skip_buf, "bytes={d} source allocation failed", .{bytes}));
                continue;
            },
            else => return err,
        };
        defer src.free();
        var dst = cuda.DeviceBuffer.alloc(gpu.d, bytes) catch |err| switch (err) {
            error.OutOfMemory, error.OutOfDeviceMemory => {
                report.skip("memory-size", try std.fmt.bufPrint(&skip_buf, "bytes={d} destination allocation failed", .{bytes}));
                continue;
            },
            else => return err,
        };
        defer dst.free();
        var init: KernelJob = .{ .function = write, .cfg = cfg, .stream = stream, .args = .{} };
        init.args.add(src.ptr);
        init.args.add(src.ptr);
        init.args.add(@as(u64, bytes / 16));
        init.args.add(sink.ptr);
        init.args.add(@as(u32, 1));
        try init.enqueue();
        try stream.synchronize();
        const cache = if (si == 0) "L2-warm" else if (bytes > l2 * 4) "beyond-L2" else "cache-influenced";
        const passes: u32 = if (si == 0) 32 else 1;
        var buf: [256]u8 = undefined;
        const details = try std.fmt.bufPrint(&buf, "bytes={d} footprint={d} passes={d} cache={s} load=cg.v4.u32 store=wb.v4.u32 grid={d} block={d}", .{ bytes, bytes * 2 + sink.len, passes, cache, cfg.grid.x, threads });
        const names = if (si == 0) [_][]const u8{ "l2-read", "l2-write", "l2-copy" } else if (bytes <= l2 * 4) [_][]const u8{ "cached-read", "cached-write", "cached-copy" } else [_][]const u8{ "memory-read", "memory-write", "memory-copy" };
        for ([_]cuda.Function{ read, write, copy }, names, 0..) |function, name, mode| {
            var job = init;
            job.function = function;
            job.args = .{};
            job.args.add(src.ptr);
            job.args.add(dst.ptr);
            job.args.add(@as(u64, bytes / 16));
            job.args.add(sink.ptr);
            job.args.add(passes);
            if (mode != 0) try dst.fill8(0xcc, null);
            const stats = try timing.measure(gpu, stream, seconds, &job);
            const traffic = bytes * passes * @as(usize, if (mode == 2) 2 else 1);
            report.add(name, "GB/s", @as(f64, @floatFromInt(traffic)) / 1e6, stats, details);
            if (mode != 0) try sameEdges(src, dst);
        }
        const legacy = cuda.Stream.borrowed(gpu.d, null);
        var job: CopyJob = .{ .stream = legacy, .src = src.ptr, .dst = dst.ptr, .bytes = bytes };
        const stats = try timing.measure(gpu, legacy, seconds, &job);
        const dtod = try std.fmt.bufPrint(&buf, "bytes={d} footprint={d} cache={s} api=cuMemcpyDtoD traffic=2*bytes stream=legacy", .{ bytes, bytes * 2, cache });
        report.add(if (si == 0) "l2-dtod" else if (bytes <= l2 * 4) "cached-dtod" else "memory-dtod", "GB/s", @as(f64, @floatFromInt(bytes * 2)) / 1e6, stats, dtod);
        try sameEdges(src, dst);
    }
}

fn sameEdges(src: cuda.DeviceBuffer, dst: cuda.DeviceBuffer) !void {
    var a: [256]u8 = undefined;
    var b: [256]u8 = undefined;
    for ([_]usize{ 0, src.len / 2, src.len - a.len }) |offset| {
        try src.download(offset, &a);
        try dst.download(offset, &b);
        try check.sameBytes("roofline memory samples", &b, &a);
    }
}

const Compute = struct { name: []const u8, symbol: [:0]const u8, k: u32, operand: u32, min_sm: u32 = 80, optional: bool = false };

fn compute(gpu: Gpu, module: cuda.Module, stream: cuda.Stream, sms: u32, sm: u32, seconds: f64, report: *timing.Report) !void {
    for ([_]Compute{
        .{ .name = "mma-fp16", .symbol = "tf_roof_f16", .k = 16, .operand = 0x3c003c00 },
        .{ .name = "mma-bf16", .symbol = "tf_roof_bf16", .k = 16, .operand = 0x3f803f80 },
        .{ .name = "mma-fp8-e4m3", .symbol = "tf_roof_f8", .k = 32, .operand = 0x38383838, .min_sm = 89, .optional = true },
        .{ .name = "mma-int8", .symbol = "tf_roof_i8", .k = 32, .operand = 0x01010101 },
        .{ .name = "fma-fp32", .symbol = "tf_roof_fma", .k = 0, .operand = 0 },
    }) |c| {
        if (sm < c.min_sm) {
            report.skip(c.name, "instruction unsupported on this SM");
            continue;
        }
        try oneCompute(gpu, module, stream, sms, seconds, report, c);
    }
}

fn computeLow(gpu: Gpu, module: cuda.Module, stream: cuda.Stream, sms: u32, seconds: f64, report: *timing.Report) !void {
    for ([_]Compute{
        .{ .name = "mma-fp4-e2m1", .symbol = "tf_roof_f4", .k = 32, .operand = 0x08080808, .optional = true },
        .{ .name = "mma-fp6-e3m2", .symbol = "tf_roof_f6", .k = 32, .operand = 0x0c0c0c0c, .optional = true },
    }) |c| try oneCompute(gpu, module, stream, sms, seconds, report, c);
}

fn oneCompute(gpu: Gpu, module: cuda.Module, stream: cuda.Stream, sms: u32, seconds: f64, report: *timing.Report, c: Compute) !void {
    var handle: cuda.abi.Function = null;
    const status = gpu.d.api.cuModuleGetFunction(&handle, module.handle, c.symbol.ptr);
    if (status == cuda.abi.error_not_found and c.optional) {
        report.skip(c.name, "loaded image has no compatible instruction entry");
        return;
    }
    try gpu.d.check(status, "roofline kernel lookup");
    const function: cuda.Function = .{ .d = gpu.d, .handle = handle };
    const local_bytes = try function.attribute(.local_size_bytes);
    const regs = try function.attribute(.num_regs);
    try check.expect(local_bytes == 0, "roofline {s}: compiler allocated {d} local bytes; register ceiling needs no spills", .{ c.name, local_bytes });
    const resident = try function.occupancy(threads, 0);
    if (resident == 0) return error.RooflineOccupancy;
    const blocks = sms * resident * 2;
    var out = try cuda.DeviceBuffer.alloc(gpu.d, blocks * threads * 4);
    defer out.free();
    var job: KernelJob = .{ .function = function, .cfg = .{ .grid = .{ .x = blocks }, .block = .{ .x = threads } }, .stream = stream, .args = .{} };
    job.args.add(out.ptr);
    job.args.add(rounds);
    job.args.add(c.operand);
    const stats = try timing.measure(gpu, stream, seconds, &job);
    const host = try gpu.gpa.alloc(f32, out.len / 4);
    defer gpu.gpa.free(host);
    try out.download(0, std.mem.sliceAsBytes(host));
    const expected: f32 = @floatFromInt(if (c.k == 0) 8 * rounds + 36 else 16 * c.k * rounds * 8);
    for (host) |v| try check.expect(v == expected, "roofline {s}: output {d} != {d}", .{ c.name, v, expected });
    const per_thread: f64 = if (c.k == 0) 2 * 8 * 8 * rounds else @as(f64, @floatFromInt(2 * 16 * 8 * c.k * 4 * 8 * rounds)) / 32;
    const work = per_thread * @as(f64, @floatFromInt(blocks * threads)) / 1e9;
    var buf: [320]u8 = undefined;
    const details = try std.fmt.bufPrint(&buf, "grid={d} block={d} warps_per_block=8 resident_blocks_per_sm={d} rounds={d} independent_chains={d} issues_per_round={d} mma_k={d} regs={d} local_bytes={d} loop_memory=0 final_store=1 dense=1", .{ blocks, threads, resident, rounds, @as(u32, if (c.k == 0) 8 else 4), @as(u32, if (c.k == 0) 64 else 32), c.k, regs, local_bytes });
    report.add(c.name, if (c.operand == 0x01010101) "TOPS" else "TFLOPS", work, stats, details);
}
