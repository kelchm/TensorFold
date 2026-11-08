//! Deterministic packed fixtures, exact BF16 dequantization and full-output checks outside timed regions.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Shape = @import("glm_prompt_mm_shapes.zig").Shape;

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn value(u: u16) f32 {
    return @bitCast(@as(u32, u) << 16);
}

fn hash(i: usize, seed: u32) u32 {
    var v: u32 = @truncate(i);
    v ^= seed;
    v ^= v >> 16;
    v *%= 0x7feb352d;
    v ^= v >> 15;
    v *%= 0x846ca68b;
    return v ^ (v >> 16);
}

fn activation(row: usize, j: usize, k: usize) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(hash(row * k + j, 17) % 15)) - 7)) / 32;
}

fn weight(col: usize, j: usize, k: usize, bits: u8) f32 {
    const top = (@as(i32, 1) << @as(u5, @intCast(bits - 1))) - 1;
    const q = if (j % 64 == 0) top else @as(i32, @intCast(hash(col * k + j, 91) % @as(u32, @intCast(2 * top + 1)))) - top;
    const scale = @as(u32, 64) << @as(u5, @intCast((col + j / 64) % 4));
    return @as(f32, @floatFromInt(q)) / @as(f32, @floatFromInt(scale));
}

fn code(words: []const u32, i: usize, bits: u8) i32 {
    const p = words[i / 16 * (bits / 2) ..][0 .. bits / 2];
    const j = i % 16;
    const raw = switch (bits) {
        4 => (p[j / 8] >> @as(u5, @intCast(j % 8 * 4))) & 15,
        8 => (p[j / 4] >> @as(u5, @intCast(j % 4 * 8))) & 255,
        else => if (j < 15) (p[j / 5] >> @as(u5, @intCast(j % 5 * 6))) & 63 else (p[0] >> 30) | ((p[1] >> 30) << 2) | ((p[2] >> 30) << 4),
    };
    return @as(i32, @intCast(raw)) - (@as(i32, 1) << @as(u5, @intCast(bits - 1)));
}

pub const Weights = struct {
    quantized: cuda.DeviceBuffer,
    bf16: cuda.DeviceBuffer,

    pub fn init(gpu: check.Gpu, mk: *const glm.mm.Kernels, stream: cuda.Stream, sh: Shape, bits: u8) !Weights {
        const a = gpu.gpa;
        const host = try a.alloc(u16, sh.n * sh.k);
        defer a.free(host);
        for (host, 0..) |*v, i| v.* = bf(weight(i / sh.k, i % sh.k, sh.k, bits));
        const words = try a.alloc(u32, sh.n * sh.k * bits / 32);
        defer a.free(words);
        const scales = try a.alloc(u16, sh.n * (sh.k / 64));
        defer a.free(scales);
        glm.qpack.quantize(bits, host, sh.n, sh.k, words, scales);
        for (host, 0..) |*v, i| {
            const scale: f32 = @as(f16, @bitCast(scales[i / 64]));
            const decoded = @as(f32, @floatFromInt(code(words, i, bits))) * scale;
            try check.expect(decoded == value(v.*), "glm-prompt-mm: fixture dequantization is exact at {d}", .{i});
            v.* = bf(decoded);
        }
        var w = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
        errdefer w.free();
        const sizes = glm.mm.packedBytes(sh.n, sh.k, bits);
        var source = try cuda.DeviceBuffer.alloc(gpu.d, sizes[0] + sizes[1]);
        defer source.free();
        try source.upload(0, std.mem.sliceAsBytes(words));
        try source.upload(sizes[0], std.mem.sliceAsBytes(scales));
        var quantized = try cuda.DeviceBuffer.alloc(gpu.d, source.len);
        errdefer quantized.free();
        try glm.mm.tilePacked(mk, stream, bits, source.ptr, source.ptr + sizes[0], quantized.ptr, quantized.ptr + sizes[0], sh.n, sh.k);
        try stream.synchronize();
        return .{ .quantized = quantized, .bf16 = w };
    }

    pub fn deinit(w: *Weights) void {
        w.quantized.free();
        w.bf16.free();
    }
};

pub fn input(gpu: check.Gpu, sh: Shape, m: usize) !cuda.DeviceBuffer {
    const host = try gpu.gpa.alloc(u16, m * sh.ldx());
    defer gpu.gpa.free(host);
    @memset(host, 0x7fc1);
    for (0..m) |row| for (0..sh.k) |j| {
        host[row * sh.ldx() + sh.offset + j] = bf(activation(row, j, sh.k));
    };
    return cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
}

fn output(bytes: []const u8, wide: bool) f32 {
    return if (wide) @bitCast(std.mem.readInt(u32, bytes[0..4], .little)) else value(std.mem.readInt(u16, bytes[0..2], .little));
}

pub fn validate(sh: Shape, m: usize, bits: u8, got: cuda.DeviceBuffer, reference: cuda.DeviceBuffer) !void {
    const size: usize = if (sh.wide) 4 else 2;
    const rtol: f32 = if (sh.wide) 1e-4 else 8e-3;
    const atol: f32 = 1e-4;
    var g: [65536]u8 = undefined;
    var r: [65536]u8 = undefined;
    var at: usize = 0;
    var max_abs: f32 = 0;
    var max_scaled: f32 = 0;
    while (at < m * sh.n) {
        const count = @min(g.len / size, m * sh.n - at);
        try got.download(at * size, g[0 .. count * size]);
        try reference.download(at * size, r[0 .. count * size]);
        for (0..count) |i| {
            const gv = output(g[i * size ..], sh.wide);
            const rv = output(r[i * size ..], sh.wide);
            const diff = @abs(gv - rv);
            const bound = atol + rtol * @abs(rv);
            try check.expect(std.math.isFinite(gv) and std.math.isFinite(rv) and diff <= bound, "glm-prompt-mm {s} M={d} [{d},{d}]: {e} != {e} bound={e}", .{ sh.name, m, (at + i) / sh.n, (at + i) % sh.n, gv, rv, bound });
            max_abs = @max(max_abs, diff);
            max_scaled = @max(max_scaled, diff / bound);
        }
        at += count;
    }
    // Independently check nine cuBLASLt outputs in FP64, including final rows and columns and the N=160 tail.
    for ([_]usize{ 0, m / 2, m - 1 }) |row| for ([_]usize{ 0, sh.n / 2, sh.n - 1 }) |col| {
        var sum: f64 = 0;
        for (0..sh.k) |j| sum += @as(f64, activation(row, j, sh.k)) * @as(f64, weight(col, j, sh.k, bits));
        const want = if (sh.wide) @as(f32, @floatCast(sum)) else value(bf(@floatCast(sum)));
        try reference.download((row * sh.n + col) * size, r[0..size]);
        const rv = output(&r, sh.wide);
        try check.expect(std.math.isFinite(rv) and @abs(rv - want) <= atol + rtol * @abs(want), "glm-prompt-mm {s}: cuBLASLt reference against FP64", .{sh.name});
    };
    std.debug.print("GLM_PROMPT_MM check={s} M={d} elements={d} reference=dequantized-bf16-cublaslt fp64_samples=9 atol={e} rtol={e} max_abs={e} max_bound_fraction={d:.4}\n", .{ sh.name, m, m * sh.n, atol, rtol, max_abs, max_scaled });
}
