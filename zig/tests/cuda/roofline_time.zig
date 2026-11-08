//! CUDA-event samples of warmed batches; graph replay for kernels/GEMM, default-stream batches for cuMemcpyDtoD.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");

pub const Stats = struct {
    best_ms: f64,
    median_ms: f64,
    gpu_ms: f64,
    repeats: usize,
    batch: usize,
    graph_batches: bool,

    pub fn best(s: Stats, work: f64) f64 {
        return work / s.best_ms;
    }

    pub fn middle(s: Stats, work: f64) f64 {
        return work / s.median_ms;
    }
};

pub fn measure(gpu: check.Gpu, stream: cuda.Stream, seconds: f64, job: anytype) !Stats {
    for (0..3) |_| try job.enqueue();
    try stream.synchronize();
    var start = try cuda.Event.init(gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(gpu.d, true);
    defer end.deinit();
    try start.record(stream);
    try job.enqueue();
    try end.record(stream);
    try end.synchronize();
    const single: f64 = try start.elapsedMs(end);
    if (!std.math.isFinite(single) or single <= 0) return error.RooflineTimer;
    const batch: usize = @intFromFloat(@min(1024, @max(1, @ceil(10 / single))));
    const Job = @typeInfo(@TypeOf(job)).pointer.child;
    const graph_batches = if (@hasDecl(Job, "graph_batches")) Job.graph_batches else true;
    var graph: ?cuda.graph.Graph = null;
    defer if (graph) |*g| g.deinit();
    var exec: ?cuda.graph.Exec = null;
    defer if (exec) |*e| e.deinit();
    if (graph_batches) {
        try cuda.graph.beginCapture(stream, .thread_local);
        for (0..batch) |_| try job.enqueue();
        graph = try cuda.graph.endCapture(stream);
        exec = try graph.?.instantiate();
        try exec.?.launchOn(stream);
    }
    try stream.synchronize();
    var samples: std.ArrayList(f64) = .empty;
    defer samples.deinit(gpu.gpa);
    var total: f64 = 0;
    var best: f64 = std.math.inf(f64);
    while (total < seconds * 1000 or samples.items.len < 7) {
        try start.record(stream);
        if (exec) |e| try e.launchOn(stream) else for (0..batch) |_| try job.enqueue();
        try end.record(stream);
        try end.synchronize();
        const ms: f64 = try start.elapsedMs(end);
        if (!std.math.isFinite(ms) or ms <= 0) return error.RooflineTimer;
        const per = ms / @as(f64, @floatFromInt(batch));
        try samples.append(gpu.gpa, per);
        best = @min(best, per);
        total += ms;
    }
    return .{ .best_ms = best, .median_ms = check.median(samples.items), .gpu_ms = total, .repeats = samples.items.len, .batch = batch, .graph_batches = graph_batches };
}

pub const Report = struct {
    const Peak = struct { name: []const u8, unit: []const u8, best: f64, median: f64 };
    peaks: [24]Peak = undefined,
    count: usize = 0,
    measurements: usize = 0,
    skipped: usize = 0,

    pub fn skip(r: *Report, name: []const u8, reason: []const u8) void {
        r.skipped += 1;
        std.debug.print("ROOFLINE {s} 0 SKIP {s}\n", .{ name, reason });
    }

    /// Work is bytes/1e6 for GB/s or operations/1e9 for TFLOPS/TOPS; FMA counts as two operations.
    pub fn add(r: *Report, name: []const u8, unit: []const u8, work: f64, s: Stats, details: []const u8) void {
        r.measurements += 1;
        const best = s.best(work);
        const median = s.middle(work);
        std.debug.print("ROOFLINE {s} {d:.3} {s} median={d:.3} best_ms={d:.6} median_ms={d:.6} repeats={d} batch={d} gpu_ms={d:.1} batch_mode={s} {s}\n", .{ name, best, unit, median, s.best_ms, s.median_ms, s.repeats, s.batch, s.gpu_ms, if (s.graph_batches) "graph" else "stream", details });
        for (r.peaks[0..r.count]) |*p| {
            if (!std.mem.eql(u8, p.name, name)) continue;
            if (best > p.best) p.* = .{ .name = name, .unit = unit, .best = best, .median = median };
            return;
        }
        r.peaks[r.count] = .{ .name = name, .unit = unit, .best = best, .median = median };
        r.count += 1;
    }

    pub fn summary(r: *const Report) void {
        std.debug.print("ROOFLINE summary {d} measurements skipped={d} peaks=best-observed median=at-winning-test dense_ops=2*M*N*K bytes=logical-read+write\n", .{ r.measurements, r.skipped });
        for (r.peaks[0..r.count]) |p|
            std.debug.print("ROOFLINE summary-{s} {d:.3} {s} median={d:.3}\n", .{ p.name, p.best, p.unit, p.median });
    }
};
