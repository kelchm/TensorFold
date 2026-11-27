//! GLM-5.3-Flash's affine 4-bit matmul for the draft head (the Python family's qmm.matmul, tile 2) and its tiling.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");

/// qmm_group.cu's tile-2 instantiations (LaneTile<64, 16, 64, 1, 4, 8>, bf16 and fp32 out).
const group_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb0ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const group_f32_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const group_smem: u32 = 35328; // LaneTile<64, 16, 64, 1, 4, 8>::SMEM

/// qmm_group.cu's Part and Parts, passed by value: four projections at most, one used here.
const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
const Parts = extern struct { p: [4]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232);
}

/// A tiled 4-bit matrix [n][k] (shared.Q4): packed words, (k / 64, npad) bf16 scales and biases, npad = n up to 128s.
pub const Q4 = struct { w: u64, s: u64, b: u64, n: usize, k: usize, npad: usize };

pub const Kernels = struct {
    group_f32: ?cuda.Function = null,
    group: cuda.Function,
    pack_dense: cuda.Function,
    transpose: cuda.Function,
    gb10: bool, // qmm_group_cuda's programmatic dependent launch: on for GB10 (sm_121)

    /// From the qmm_group and affine4_pack modules.
    pub fn load(group: cuda.Module, ops: cuda.Module, ctx: *const cuda.Context) !Kernels {
        const k: Kernels = .{
            .group = try group.function(group_symbol),
            .pack_dense = try ops.function("tf_pack_dense"),
            .transpose = try ops.function("tf_transpose_pad16"),
            .gb10 = try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1,
        };
        try k.group.allowDynamicShared(group_smem);
        return k;
    }

    /// Resolve the FP32-output tile-2 kernel only when TF_GLM_DFLASH_Q4_DOWN=1 requests Q4 down projections.
    pub fn prepareWide(k: *Kernels, group: cuda.Module) !void {
        if (k.group_f32 != null) return;
        const f = try group.optionalFunction(group_f32_symbol) orelse {
            if (!@import("builtin").is_test) std.log.err("TF_GLM_DFLASH_Q4_DOWN=1 requires qmm_group FP32-output kernel ({s}); rebuild fatbins or unset TF_GLM_DFLASH_Q4_DOWN", .{group_f32_symbol});
            return error.MissingWideQ4Kernel;
        };
        try f.allowDynamicShared(group_smem);
        k.group_f32 = f;
    }
};

pub fn npadOf(n: usize) usize {
    return (n + 127) / 128 * 128;
}

/// Device bytes of a Q4's words, scales and biases.
pub fn bytes(n: usize, k: usize) [3]usize {
    const p = npadOf(n);
    return .{ p * k / 2, k / 64 * p * 2, k / 64 * p * 2 };
}

/// qmm.split_k: K slices fixed by the weight's shape, never by the row count.
pub fn splitK(n: usize, k: usize) usize {
    const tiles = (n + 63) / 64;
    const groups = k / 64;
    var sk: usize = 1;
    while (sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups / (sk * 2) >= 8) sk *= 2;
    return sk;
}

/// shared.pack: MLX words i32 [n][k/8] and bf16 scales and biases [n][k/64] -> `q` (caller's buffers, `bytes` long).
pub fn pack(ks: *const Kernels, stream: cuda.Stream, words: u64, scales: u64, biases: u64, q: Q4) !void {
    const kg = q.k / 64;
    const total: u64 = @as(u64, q.npad / 64) * kg * 512;
    var a: cuda.Args = .{};
    a.add(words);
    for ([_]usize{ q.n, q.k / 8, kg }) |v| a.add(@as(c_int, @intCast(v)));
    a.add(q.w);
    a.add(@as(c_longlong, @intCast(total)));
    try cuda.launch.launch(ks.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    for ([_]u64{ scales, biases }, [_]u64{ q.s, q.b }) |src, dst| {
        var t: cuda.Args = .{};
        t.add(src);
        for ([_]usize{ q.n, kg, q.npad }) |v| t.add(@as(c_int, @intCast(v)));
        t.add(dst);
        const cells = @as(u64, kg) * q.npad;
        try cuda.launch.launch(ks.transpose, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, stream, &t);
    }
}

/// qmm.matmul of a Q4 (tile 2, up to 16 rows): x bf16 (`ldx` apart), group sums xs fp32 [rows][k/64] -> out bf16.
pub fn matmul(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize) !void {
    return matmulOutput(ks, stream, x, ldx, xs, q, out, rows, false);
}

/// A row-parallel projection retains its unrounded accumulator until after tensor-parallel summation.
pub fn matmulOutput(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize, wide: bool) !void {
    if (rows == 0 or rows > 16) return error.WindowTooWide;
    const sk = splitK(q.n, q.k);
    const tiles = (q.n + 63) / 64;
    var parts: Parts = std.mem.zeroes(Parts);
    parts.count = 1;
    parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = @intCast(q.n), .npad = @intCast(q.npad), .sk = @intCast(sk), .tiles = @intCast(tiles), .first = 0 };
    const rows_t = (rows + 15) / 16;
    const clusters = rows_t * tiles; // one part: a cluster is the sk K slices of one column tile
    var a: cuda.Args = .{};
    a.add(x);
    a.add(xs);
    a.add(parts);
    for ([_]usize{ rows, q.k, if (rows == 1) q.k else ldx, rows_t, sk }) |v| a.add(@as(c_int, @intCast(v))); // M, K, ldx, rows_t, C
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(clusters * sk) },
        .block = .{ .x = 128 },
        .shared = group_smem,
        .cluster = if (sk > 1) .{ .x = @intCast(sk) } else null,
        .pdl = ks.gb10 and cuda_pdl.on(),
    };
    const f = if (wide) (ks.group_f32 orelse return error.MissingWideQ4Kernel) else ks.group;
    try cuda.launch.launch(f, cfg, stream, &a);
}

test "the draft head's shape takes one K slice; npad rounds to 128s" {
    try std.testing.expectEqual(@as(usize, 1), splitK(77440, 4096));
    try std.testing.expectEqual(@as(usize, 77440), npadOf(77440));
    try std.testing.expectEqual(@as(usize, 51712), npadOf(51627));
}

test "GLM Q4: load skips FP32 down symbol until prepareWide enables it" {
    const Fake = struct {
        var allow_f32 = false;
        var lookups: usize = 0;
        var smem_calls: usize = 0;
        var last_launched: usize = 0;
        fn getFn(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            const s = std.mem.span(name);
            lookups += 1;
            if (std.mem.eql(u8, s, group_f32_symbol)) {
                if (!allow_f32) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(42);
                return 0;
            }
            out.* = @ptrFromInt(11);
            return 0;
        }
        fn setAttr(_: cuda.abi.Function, _: cuda.abi.FunctionAttribute, v: c_int) callconv(.c) cuda.abi.Result {
            if (v == @as(c_int, @intCast(group_smem))) smem_calls += 1;
            return 0;
        }
        fn errStr(_: cuda.abi.Result, out: *?[*:0]const u8) callconv(.c) cuda.abi.Result {
            out.* = null;
            return 1;
        }
        fn devAttr(out: *c_int, a: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            out.* = if (a == .compute_capability_major) 12 else 1;
            return 0;
        }
        fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            last_launched = @intFromPtr(f);
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Fake.getFn;
    d.api.cuFuncSetAttribute = Fake.setAttr;
    d.api.cuGetErrorName = Fake.errStr;
    d.api.cuGetErrorString = Fake.errStr;
    d.api.cuDeviceGetAttribute = Fake.devAttr;
    d.api.cuLaunchKernel = Fake.launch;
    d.api.cuLaunchKernelEx = cuda_pdl.forwardEx(Fake.launch);
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    const mod: cuda.Module = .{ .d = &d, .handle = null };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const q: Q4 = .{ .w = 0, .s = 0, .b = 0, .n = 4096, .k = 4096, .npad = 4096 };
    var ks = try Kernels.load(mod, mod, &ctx);
    try std.testing.expect(ks.group_f32 == null);
    try std.testing.expectEqual(@as(usize, 1), Fake.smem_calls);
    try std.testing.expectError(error.MissingWideQ4Kernel, matmulOutput(&ks, stream, 0, 4096, 0, q, 0, 8, true));
    try std.testing.expectError(error.MissingWideQ4Kernel, ks.prepareWide(mod));
    try std.testing.expect(ks.group_f32 == null);
    Fake.allow_f32 = true;
    try ks.prepareWide(mod);
    try std.testing.expectEqual(@as(usize, 2), Fake.smem_calls);
    try matmulOutput(&ks, stream, 0, 4096, 0, q, 0, 8, true);
    try std.testing.expectEqual(@as(usize, 42), Fake.last_launched);
}
