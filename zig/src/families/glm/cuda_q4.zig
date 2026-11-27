//! GLM-5.3-Flash's affine 4-bit matmul for the draft head (the Python family's qmm.matmul, tile 2) and its tiling.

const std = @import("std");
const cuda = @import("cuda");
const cuda_pdl = @import("cuda_pdl.zig");

/// qmm_group.cu's tile-2 instantiations (16..128 rows bf16 out, and 16/32/64-row fp32 out for DFlash2 split down).
const group_symbol = cuda.qlinear.symbols.group;
const verify32_symbol = cuda.qlinear.symbols.verify32;
const verify64_symbol = cuda.qlinear.symbols.verify64;
const verify128_symbol = cuda.qlinear.symbols.verify128;
const group_f32_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi16ELi64ELi1ELi4ELi8ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const verify32_f32_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi32ELi64ELi1ELi4ELi5ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const verify64_f32_symbol = "_ZN12tf_qmm_group12group_kernelILi64ELi64ELi64ELi1ELi4ELi4ELb1ELb0ELb0ELb0EEEvPK13__nv_bfloat16PKfNS_5PartsEiiiii";
const group_smem: u32 = cuda.qlinear.group_smem;
const verify32_smem: u32 = cuda.qlinear.verify_smem;
const verify64_smem: u32 = cuda.qlinear.verify64_smem;
const verify128_smem: u32 = cuda.qlinear.verify128_smem;

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
    verify32_f32: ?cuda.Function = null,
    verify64_f32: ?cuda.Function = null,
    verify32: ?cuda.Function = null,
    verify64: ?cuda.Function = null,
    verify128: ?cuda.Function = null,
    group: cuda.Function,
    pack_dense: cuda.Function,
    transpose: cuda.Function,
    gb10: bool, // qmm_group_cuda's programmatic dependent launch: on for GB10 (sm_121)

    /// From the qmm_group and affine4_pack modules.
    pub fn load(group: cuda.Module, ops: cuda.Module, ctx: *const cuda.Context) !Kernels {
        var k: Kernels = .{
            .group = try group.function(group_symbol),
            .pack_dense = try ops.function("tf_pack_dense"),
            .transpose = try ops.function("tf_transpose_pad16"),
            .gb10 = try ctx.attribute(.compute_capability_major) == 12 and try ctx.attribute(.compute_capability_minor) == 1,
        };
        try k.group.allowDynamicShared(group_smem);
        k.verify32 = try group.optionalFunction(verify32_symbol);
        if (k.verify32) |f| try f.allowDynamicShared(verify32_smem);
        k.verify64 = try group.optionalFunction(verify64_symbol);
        if (k.verify64) |f| try f.allowDynamicShared(verify64_smem);
        k.verify128 = try group.optionalFunction(verify128_symbol);
        if (k.verify128) |f| try f.allowDynamicShared(verify128_smem);
        return k;
    }

    /// Largest single-launch row step: 128 on FP32 multi-slab, else the highest contiguous BF16 verify tier loaded.
    pub fn maxRows(ks: *const Kernels, wide: bool) usize {
        if (wide) return 128;
        if (ks.verify32 != null) {
            if (ks.verify64 != null) return if (ks.verify128 != null) 128 else 64;
            return 32;
        }
        return 16;
    }

    /// Resolve FP32-output tile-2 kernels only when TF_GLM_DFLASH_Q4_DOWN=1 requests Q4 down projections.
    pub fn prepareWide(k: *Kernels, group: cuda.Module) !void {
        if (k.group_f32 != null) return;
        const f = try group.optionalFunction(group_f32_symbol) orelse {
            if (!@import("builtin").is_test) std.log.err("TF_GLM_DFLASH_Q4_DOWN=1 requires qmm_group FP32-output kernel ({s}); rebuild fatbins or unset TF_GLM_DFLASH_Q4_DOWN", .{group_f32_symbol});
            return error.MissingWideQ4Kernel;
        };
        try f.allowDynamicShared(group_smem);
        const v32 = try group.optionalFunction(verify32_f32_symbol);
        if (v32) |g| try g.allowDynamicShared(verify32_smem);
        const v64 = try group.optionalFunction(verify64_f32_symbol);
        if (v64) |g| try g.allowDynamicShared(verify64_smem);
        k.group_f32 = f;
        k.verify32_f32 = v32;
        k.verify64_f32 = v64;
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

/// qmm.matmul of a Q4 (up to ks.maxRows(false) rows): x bf16 (`ldx` apart), group sums xs fp32 [rows][k/64] -> bf16.
pub fn matmul(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize) !void {
    return matmulOutput(ks, stream, x, ldx, xs, q, out, rows, false);
}

/// Pick the widest loaded FP32-output tile avoiding redundant weight reads across 1..128 rows.
fn wideTile(ks: *const Kernels, rows: usize) !struct { cuda.Function, u32, usize } {
    const base = ks.group_f32 orelse return error.MissingWideQ4Kernel;
    if (rows <= 16) return .{ base, group_smem, 1 };
    if (rows <= 32) {
        if (ks.verify32_f32) |f| return .{ f, verify32_smem, 1 };
        return .{ base, group_smem, (rows + 15) / 16 };
    }
    if (ks.verify64_f32) |f| return .{ f, verify64_smem, (rows + 63) / 64 };
    if (ks.verify32_f32) |f| return .{ f, verify32_smem, (rows + 31) / 32 };
    return .{ base, group_smem, (rows + 15) / 16 };
}

/// A row-parallel projection retains its unrounded accumulator until after tensor-parallel summation.
pub fn matmulOutput(ks: *const Kernels, stream: cuda.Stream, x: u64, ldx: usize, xs: u64, q: Q4, out: u64, rows: usize, wide: bool) !void {
    if (rows == 0 or rows > 128) return error.WindowTooWide;
    const f: cuda.Function, const shared: u32, const rows_t: usize = if (wide)
        try wideTile(ks, rows)
    else if (rows <= 16)
        .{ ks.group, group_smem, 1 }
    else if (rows <= 32)
        .{ ks.verify32 orelse return error.WindowTooWide, verify32_smem, 1 }
    else if (rows <= 64)
        .{ ks.verify64 orelse return error.WindowTooWide, verify64_smem, 1 }
    else
        .{ ks.verify128 orelse return error.WindowTooWide, verify128_smem, 1 };
    const sk = splitK(q.n, q.k);
    const tiles = (q.n + 63) / 64;
    var parts: Parts = std.mem.zeroes(Parts);
    parts.count = 1;
    parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = @intCast(q.n), .npad = @intCast(q.npad), .sk = @intCast(sk), .tiles = @intCast(tiles), .first = 0 };
    const clusters = rows_t * tiles; // one part: a cluster is the sk K slices of one column tile
    var a: cuda.Args = .{};
    a.add(x);
    a.add(xs);
    a.add(parts);
    for ([_]usize{ rows, q.k, if (rows == 1) q.k else ldx, rows_t, sk }) |v| a.add(@as(c_int, @intCast(v))); // M, K, ldx, rows_t, C
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(clusters * sk) },
        .block = .{ .x = 128 },
        .shared = shared,
        .cluster = if (sk > 1) .{ .x = @intCast(sk) } else null,
        .pdl = ks.gb10 and cuda_pdl.on(),
    };
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
            if (std.mem.eql(u8, s, verify32_f32_symbol) or std.mem.eql(u8, s, verify64_f32_symbol)) {
                return cuda.abi.error_not_found;
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
    try std.testing.expect(ks.group_f32 == null and ks.verify32_f32 == null and ks.verify64_f32 == null);
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

test "GLM Q4: wide BF16 verify tiers and FP32 multi-slab dispatch exact grids, smem, and degradation" {
    const t = std.testing;
    const Rec = struct {
        var allow: [6]bool = .{ true, true, true, true, true, true }; // v32, v64, v128, f32, v32_f32, v64_f32
        var smem_by_fn: [64]u32 = @splat(0);
        var func: usize = 0;
        var grid_x: c_uint = 0;
        var shared: c_uint = 0;
        var cluster_x: c_uint = 0;
        var arg_m: c_int = 0;
        var arg_ldx: c_int = 0;
        var arg_rows_t: c_int = 0;
        var arg_c: c_int = 0;
        fn getFn(out: *cuda.abi.Function, _: cuda.abi.Module, name: [*:0]const u8) callconv(.c) cuda.abi.Result {
            const s = std.mem.span(name);
            if (std.mem.eql(u8, s, group_symbol)) {
                out.* = @ptrFromInt(10);
                return 0;
            }
            if (std.mem.eql(u8, s, verify32_symbol)) {
                if (!allow[0]) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(32);
                return 0;
            }
            if (std.mem.eql(u8, s, verify64_symbol)) {
                if (!allow[1]) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(40);
                return 0;
            }
            if (std.mem.eql(u8, s, verify128_symbol)) {
                if (!allow[2]) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(48);
                return 0;
            }
            if (std.mem.eql(u8, s, group_f32_symbol)) {
                if (!allow[3]) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(56);
                return 0;
            }
            if (std.mem.eql(u8, s, verify32_f32_symbol)) {
                if (!allow[4]) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(60);
                return 0;
            }
            if (std.mem.eql(u8, s, verify64_f32_symbol)) {
                if (!allow[5]) return cuda.abi.error_not_found;
                out.* = @ptrFromInt(62);
                return 0;
            }
            out.* = @ptrFromInt(11);
            return 0;
        }
        fn setAttr(f: cuda.abi.Function, _: cuda.abi.FunctionAttribute, v: c_int) callconv(.c) cuda.abi.Result {
            const id = @intFromPtr(f);
            if (id < smem_by_fn.len) smem_by_fn[id] = @intCast(v);
            return 0;
        }
        fn devAttr(out: *c_int, a: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
            out.* = if (a == .compute_capability_major) 12 else 1;
            return 0;
        }
        fn captureArgs(params: ?[*]?*anyopaque) void {
            const p = params.?;
            arg_m = @as(*const c_int, @ptrCast(@alignCast(p[3].?))).*;
            arg_ldx = @as(*const c_int, @ptrCast(@alignCast(p[5].?))).*;
            arg_rows_t = @as(*const c_int, @ptrCast(@alignCast(p[6].?))).*;
            arg_c = @as(*const c_int, @ptrCast(@alignCast(p[7].?))).*;
        }
        fn launch(f: cuda.abi.Function, gx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, sh: c_uint, _: cuda.abi.Stream, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            func = @intFromPtr(f);
            grid_x = gx;
            shared = sh;
            cluster_x = 0;
            captureArgs(params);
            return 0;
        }
        fn launchEx(c: *const cuda.abi.LaunchConfig, f: cuda.abi.Function, params: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) cuda.abi.Result {
            func = @intFromPtr(f);
            grid_x = c.grid_x;
            shared = c.shared_bytes;
            cluster_x = 0;
            if (c.attrs) |attrs| for (attrs[0..c.num_attrs]) |at| if (at.id == .cluster_dimension) {
                cluster_x = at.value.cluster_dim.x;
            };
            captureArgs(params);
            return 0;
        }
    };
    var d: cuda.Driver = undefined;
    d.api.cuModuleGetFunction = Rec.getFn;
    d.api.cuFuncSetAttribute = Rec.setAttr;
    d.api.cuDeviceGetAttribute = Rec.devAttr;
    d.api.cuLaunchKernel = Rec.launch;
    d.api.cuLaunchKernelEx = Rec.launchEx;
    const ctx: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    const mod: cuda.Module = .{ .d = &d, .handle = null };
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    const q: Q4 = .{ .w = 0x1000, .s = 0x2000, .b = 0x3000, .n = 4096, .k = 4096, .npad = 4096 };
    const tiles: usize = 64;
    const sk = splitK(q.n, q.k);
    try t.expectEqual(@as(usize, 4), sk);
    // 1) All BF16 and FP32 verify tiers present: load + prepareWide set exact dynamic smem.
    Rec.allow = .{ true, true, true, true, true, true };
    Rec.smem_by_fn = @splat(0);
    var ks = try Kernels.load(mod, mod, &ctx);
    try ks.prepareWide(mod);
    try t.expectEqual(@as(usize, 128), ks.maxRows(false));
    try t.expectEqual(@as(usize, 128), ks.maxRows(true));
    try t.expectEqual(group_smem, Rec.smem_by_fn[10]);
    try t.expectEqual(verify32_smem, Rec.smem_by_fn[32]);
    try t.expectEqual(verify64_smem, Rec.smem_by_fn[40]);
    try t.expectEqual(verify128_smem, Rec.smem_by_fn[48]);
    try t.expectEqual(group_smem, Rec.smem_by_fn[56]);
    try t.expectEqual(verify32_smem, Rec.smem_by_fn[60]);
    try t.expectEqual(verify64_smem, Rec.smem_by_fn[62]);
    const bf16_cases = [_]struct { rows: usize, func: usize, smem: u32 }{
        .{ .rows = 1, .func = 10, .smem = group_smem },
        .{ .rows = 16, .func = 10, .smem = group_smem },
        .{ .rows = 17, .func = 32, .smem = verify32_smem },
        .{ .rows = 32, .func = 32, .smem = verify32_smem },
        .{ .rows = 33, .func = 40, .smem = verify64_smem },
        .{ .rows = 64, .func = 40, .smem = verify64_smem },
        .{ .rows = 65, .func = 48, .smem = verify128_smem },
        .{ .rows = 128, .func = 48, .smem = verify128_smem },
    };
    for (bf16_cases) |c| {
        try matmulOutput(&ks, stream, 0x4000, 4608, 0x5000, q, 0x6000, c.rows, false);
        try t.expectEqual(c.func, Rec.func);
        try t.expectEqual(c.smem, Rec.shared);
        try t.expectEqual(@as(c_uint, @intCast(tiles * sk)), Rec.grid_x);
        try t.expectEqual(@as(c_uint, @intCast(sk)), Rec.cluster_x);
        try t.expectEqual(@as(c_int, @intCast(c.rows)), Rec.arg_m);
        try t.expectEqual(@as(c_int, @intCast(if (c.rows == 1) q.k else 4608)), Rec.arg_ldx);
        try t.expectEqual(@as(c_int, 1), Rec.arg_rows_t);
        try t.expectEqual(@as(c_int, @intCast(sk)), Rec.arg_c);
    }
    // 2) Wide FP32 dispatch across full (v32+v64), partial (v32-only, v64-only), and group_f32-only fallback sets.
    const WideCase = struct { rows: usize, func: usize, smem: u32, rows_t: usize };
    const sets = [_]struct { v32: bool, v64: bool, want: [7]WideCase }{
        .{ .v32 = true, .v64 = true, .want = .{
            .{ .rows = 16, .func = 56, .smem = group_smem, .rows_t = 1 },
            .{ .rows = 17, .func = 60, .smem = verify32_smem, .rows_t = 1 },
            .{ .rows = 32, .func = 60, .smem = verify32_smem, .rows_t = 1 },
            .{ .rows = 33, .func = 62, .smem = verify64_smem, .rows_t = 1 },
            .{ .rows = 64, .func = 62, .smem = verify64_smem, .rows_t = 1 },
            .{ .rows = 65, .func = 62, .smem = verify64_smem, .rows_t = 2 },
            .{ .rows = 128, .func = 62, .smem = verify64_smem, .rows_t = 2 },
        } },
        .{ .v32 = true, .v64 = false, .want = .{
            .{ .rows = 16, .func = 56, .smem = group_smem, .rows_t = 1 },
            .{ .rows = 17, .func = 60, .smem = verify32_smem, .rows_t = 1 },
            .{ .rows = 32, .func = 60, .smem = verify32_smem, .rows_t = 1 },
            .{ .rows = 33, .func = 60, .smem = verify32_smem, .rows_t = 2 },
            .{ .rows = 64, .func = 60, .smem = verify32_smem, .rows_t = 2 },
            .{ .rows = 65, .func = 60, .smem = verify32_smem, .rows_t = 3 },
            .{ .rows = 128, .func = 60, .smem = verify32_smem, .rows_t = 4 },
        } },
        .{ .v32 = false, .v64 = true, .want = .{
            .{ .rows = 16, .func = 56, .smem = group_smem, .rows_t = 1 },
            .{ .rows = 17, .func = 56, .smem = group_smem, .rows_t = 2 },
            .{ .rows = 32, .func = 56, .smem = group_smem, .rows_t = 2 },
            .{ .rows = 33, .func = 62, .smem = verify64_smem, .rows_t = 1 },
            .{ .rows = 64, .func = 62, .smem = verify64_smem, .rows_t = 1 },
            .{ .rows = 65, .func = 62, .smem = verify64_smem, .rows_t = 2 },
            .{ .rows = 128, .func = 62, .smem = verify64_smem, .rows_t = 2 },
        } },
        .{ .v32 = false, .v64 = false, .want = .{
            .{ .rows = 16, .func = 56, .smem = group_smem, .rows_t = 1 },
            .{ .rows = 17, .func = 56, .smem = group_smem, .rows_t = 2 },
            .{ .rows = 32, .func = 56, .smem = group_smem, .rows_t = 2 },
            .{ .rows = 33, .func = 56, .smem = group_smem, .rows_t = 3 },
            .{ .rows = 64, .func = 56, .smem = group_smem, .rows_t = 4 },
            .{ .rows = 65, .func = 56, .smem = group_smem, .rows_t = 5 },
            .{ .rows = 128, .func = 56, .smem = group_smem, .rows_t = 8 },
        } },
    };
    for (sets) |set| {
        Rec.allow = .{ true, true, true, true, set.v32, set.v64 };
        var kw = try Kernels.load(mod, mod, &ctx);
        try kw.prepareWide(mod);
        for (set.want) |w| {
            try matmulOutput(&kw, stream, 0x4000, 4608, 0x5000, q, 0x6000, w.rows, true);
            try t.expectEqual(w.func, Rec.func);
            try t.expectEqual(w.smem, Rec.shared);
            try t.expectEqual(@as(c_uint, @intCast(w.rows_t * tiles * sk)), Rec.grid_x);
            try t.expectEqual(@as(c_uint, @intCast(sk)), Rec.cluster_x);
            try t.expectEqual(@as(c_int, @intCast(w.rows)), Rec.arg_m);
            try t.expectEqual(@as(c_int, 4608), Rec.arg_ldx);
            try t.expectEqual(@as(c_int, @intCast(w.rows_t)), Rec.arg_rows_t);
            try t.expectEqual(@as(c_int, @intCast(sk)), Rec.arg_c);
        }
    }
    try t.expectError(error.WindowTooWide, matmulOutput(&ks, stream, 0, 4096, 0, q, 0, 0, false));
    try t.expectError(error.WindowTooWide, matmulOutput(&ks, stream, 0, 4096, 0, q, 0, 129, false));
    try t.expectError(error.WindowTooWide, matmulOutput(&ks, stream, 0, 4096, 0, q, 0, 129, true));
    // 3) Optional wide BF16 symbols degrade maxRows contiguously and refuse missing tiers cleanly.
    Rec.allow = .{ true, true, false, false, false, false };
    const k64 = try Kernels.load(mod, mod, &ctx);
    try t.expectEqual(@as(usize, 64), k64.maxRows(false));
    try t.expectError(error.WindowTooWide, matmulOutput(&k64, stream, 0, 4096, 0, q, 0, 65, false));
    Rec.allow = .{ true, false, true, false, false, false };
    const k32 = try Kernels.load(mod, mod, &ctx);
    try t.expectEqual(@as(usize, 32), k32.maxRows(false));
    try t.expectError(error.WindowTooWide, matmulOutput(&k32, stream, 0, 4096, 0, q, 0, 33, false));
    Rec.allow = .{ false, true, true, false, false, false };
    const k16 = try Kernels.load(mod, mod, &ctx);
    try t.expectEqual(@as(usize, 16), k16.maxRows(false));
    try t.expectError(error.WindowTooWide, matmulOutput(&k16, stream, 0, 4096, 0, q, 0, 17, false));
}
