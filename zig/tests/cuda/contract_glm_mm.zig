//! Packed prompt projections: the parent's bytes, independent FP64 classes, FP32 sums and guarded tile edges.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const Rig = @import("contract_rig.zig").Rig;
const check = @import("check.zig");

const served = [_]c.Shape{
    .{ .rows = 37, .n = 72, .k = 8192 },
    .{ .rows = 768, .n = 6416, .k = 4096 },
    .{ .rows = 129, .n = 4096, .k = 2048 },
    .{ .rows = 128, .n = 2048, .k = 4096 },
    .{ .rows = 127, .n = 4096, .k = 1024 },
    .{ .rows = 2048, .n = 12288, .k = 4096 },
    .{ .rows = 768, .n = 4096, .k = 6144 },
    .{ .rows = 1, .n = 2560, .k = 4096 },
    .{ .rows = 33, .n = 4096, .k = 4096 },
    .{ .rows = 1000, .n = 1000, .k = 1024 },
};

fn depth(shape: c.Shape) usize {
    return 64 + shape.k / 64;
}

fn pack(dst: []u32, codes: [16]u32, bits: u8) void {
    @memset(dst, 0);
    for (codes, 0..) |q, i| {
        if (bits == 6 and i == 15) {
            for (0..3) |j| dst[j] |= ((q >> @as(u5, @intCast(2 * j))) & 3) << 30;
        } else {
            const per: usize = 32 / bits;
            dst[i / per] |= q << @as(u5, @intCast((i % per) * bits));
        }
    }
}

/// All terms are integers below 2^24 in FP32; one unequal mirrored input leaves a small result.
fn stress(d: Dense) void {
    const k = d.shape.k;
    for (0..d.shape.rows) |r| {
        @memcpy(d.x[r * k + k / 2 ..][0 .. k / 2], d.x[r * k ..][0 .. k / 2]);
        d.x[r * k + k / 2] = c.toBf16(c.fromBf16(d.x[r * k + k / 2]) + 16);
    }
}

pub const Product = struct {
    rig: Rig,
    kernels: *const glm.mm.Kernels,
    bits: u8,
    wide: bool,
    drop_tail: bool = false,
    d: ?Dense = null,
    x: u64 = 0,
    w: glm.mm.Tiled = undefined,
    out: u64 = 0,
    base: u64 = 0,
    name: [120]u8 = undefined,
    parent: [32]u8 = undefined,

    pub fn deinit(self: *Product) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Product) !g.Kernel {
        const name = try std.fmt.bufPrint(&self.name, "{s}glm_mm{d}_128s2 {s}", .{ if (self.drop_tail) "control: dropped prompt row " else "cuda ", self.bits, if (self.wide) "fp32" else "bf16" });
        const parent = try std.fmt.bufPrint(&self.parent, "glm_mm{d}_128", .{self.bits});
        return .{ .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = if (self.wide) .f32 else .bf16,
            .replaces = parent,
            .tiles = .{ .{ .size = 128 }, .{ .size = 64, .step = 8, .min = 8 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &served,
        }, .ctx = self, .vt = &vt, .control = if (self.drop_tail) .shapes else null };
    }

    const vt = g.hooks(Product);

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 8 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 16);
        if (fill == .stress) stress(d);
        const stride = shape.k + 8;
        const hx = try a.alloc(u16, (shape.rows + 7) * stride);
        defer a.free(hx);
        @memset(hx, 0x7fc0);
        for (0..shape.rows) |r| @memcpy(hx[(r + 7) * stride ..][0..shape.k], d.x[r * shape.k ..][0..shape.k]);
        self.x = try self.rig.input(std.mem.sliceAsBytes(hx)) + 7 * stride * 2;
        try self.weights(fill, seed);
        const bytes = shape.rows * shape.n * @as(usize, if (self.wide) 4 else 2);
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        return true;
    }

    fn weights(self: *Product, fill: g.Fill, seed: u64) !void {
        const d = self.d.?;
        const sh = d.shape;
        const a = self.rig.gpu.gpa;
        const words = try a.alloc(u32, sh.n * sh.k * self.bits / 32);
        defer a.free(words);
        const scales = try a.alloc(u16, sh.n * sh.k / 64);
        defer a.free(scales);
        var rng = std.Random.DefaultPrng.init(seed ^ 0x128);
        const random = rng.random();
        const half: i32 = @as(i32, 1) << @as(u5, @intCast(self.bits - 1));
        for (0..sh.n / 8) |tile| for (0..sh.k / 64) |group| {
            const tg = tile * (sh.k / 64) + group;
            for (0..8) |col| scales[tg * 8 + col] = @bitCast(@as(f16, if (fill == .stress) 1 else @floatCast(0.01 + random.float(f32) * 0.1)));
            for (0..32) |lane| {
                const col = tile * 8 + lane / 4;
                const scale: f64 = @as(f16, @bitCast(scales[tg * 8 + lane / 4]));
                var codes: [16]u32 = undefined;
                for (&codes, 0..) |*q, v| {
                    const j = v % 4;
                    const k = group * 64 + (v / 4) * 16 + 2 * (lane % 4) + j % 2 + (j / 2) * 8;
                    const small: i32 = @intCast(1 + ((k % (sh.k / 2)) * 5 + col) % 7);
                    const signed: i32 = if (fill == .stress) (if (k < sh.k / 2) small else -small) else @as(i32, random.uintLessThan(u16, @intCast(half * 2))) - half;
                    q.* = @intCast(signed + half);
                    d.w[col * sh.k + k] = @as(f64, @floatFromInt(signed)) * scale;
                }
                const at = (tg * 32 + lane) * (self.bits / 2);
                pack(words[at..][0 .. self.bits / 2], codes, self.bits);
            }
        };
        self.w = .{ .w = try self.rig.input(std.mem.sliceAsBytes(words)), .s = try self.rig.input(std.mem.sliceAsBytes(scales)), .n = sh.n, .k = sh.k, .bits = self.bits };
    }

    pub fn run(self: *Product) !void {
        const sh = self.d.?.shape;
        if (self.drop_tail) {
            if (sh.rows > 1) try self.launch(self.kernels.packed_prompt[(self.bits - 4) / 2], self.out, sh.rows - 1, 2);
        } else {
            var kernels = self.kernels.*;
            kernels.prompt_fixed_k = false;
            try glm.mm.matmul(&kernels, self.rig.s, self.x, sh.k + 8, self.w, self.out, 0, sh.rows, self.wide, true);
        }
        try self.rig.s.synchronize();
    }

    fn launch(self: *Product, f: cuda.Function, out: u64, rows: usize, stages: usize) !void {
        const sh = self.d.?.shape;
        var args: cuda.Args = .{};
        inline for (.{ self.x, @as(i64, @intCast(sh.k + 8)), self.w.w, self.w.s, out, @as(u64, 0), @as(c_int, @intFromBool(self.wide)), @as(c_int, @intCast(rows)), @as(c_int, @intCast(sh.n)), @as(c_int, @intCast(sh.k)), @as(c_int, 1) }) |v| args.add(v);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((sh.n + 63) / 64), .y = @intCast((rows + 127) / 128) }, .block = .{ .x = 256 }, .shared = glm.mm.sharedBytesStages(self.bits, 128, stages) }, self.rig.s, &args);
    }

    pub fn replaced(self: *Product) ![]const u8 {
        try self.launch(self.kernels.packed_rows[(self.bits - 4) / 2][3], self.base, self.d.?.shape.rows, 3);
        return self.rig.bytes(self.base, self.count() * @as(usize, if (self.wide) 4 else 2));
    }

    pub fn count(self: *Product) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Product, want: []f64, mag: []f64) !void {
        try self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Product, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.d.?.simulate(acc, order, out);
    }
};

pub const Prompts = struct {
    module: cuda.Module,
    kernels: glm.mm.Kernels,
    cases: [12]Product = undefined,

    pub fn init(gpu: check.Gpu) !Prompts {
        var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
        errdefer m.unload();
        return .{ .module = m, .kernels = try glm.mm.Kernels.load(m) };
    }

    pub fn append(self: *Prompts, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        for (&self.cases, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kernels = &self.kernels, .bits = @intCast(4 + 2 * (i % 3)), .wide = i % 6 >= 3, .drop_tail = i >= 6 };
        for (&self.cases) |*p| try kernels.append(gpu.gpa, try p.kernel());
    }

    pub fn deinit(self: *Prompts) void {
        for (&self.cases) |*p| p.deinit();
        self.module.unload();
    }
};

test "GLM prompt contract: stress distinguishes FP16 and BF16 sums in every order and output kind" {
    const d = try Dense.init(std.testing.allocator, .{ .rows = 2, .n = 8, .k = 8192 }, 1);
    defer d.deinit();
    d.fillX(.stress, 5, 16);
    stress(d);
    for (0..8) |col| for (0..8192) |k| {
        const q: f64 = @floatFromInt(1 + ((k % 4096) * 5 + col) % 7);
        d.w[col * 8192 + k] = if (k < 4096) q else -q;
    };
    var want: [16]f64 = undefined;
    var mag: [16]f64 = undefined;
    var got: [16]f64 = undefined;
    try d.reference(&want, &mag);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |order| {
        try d.simulate(.f32, order, &got);
        try std.testing.expectEqualSlices(f64, &want, &got);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try d.simulate(acc, order, &got);
            for ([_]c.Elem{ .bf16, .f32 }) |out| {
                var worst: f64 = 0;
                const spec: c.Spec = .{ .name = "stress", .accum = .f32, .depth = depth, .out = out, .served = &served };
                for (got, want) |v, w| worst = @max(worst, if (v == w) 0 else if (!std.math.isFinite(v)) std.math.inf(f64) else @abs(v - w) / c.boundExact(spec, w));
                try std.testing.expect(worst >= 100);
            }
        }
    }
}
