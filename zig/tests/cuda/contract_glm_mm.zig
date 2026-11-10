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
const slot_k = @import("contract_glm_mm_slot.zig");
pub const Gpu = check.Gpu;

const served = [_]c.Shape{
    .{ .rows = 37, .n = 72, .k = 8192 },
    .{ .rows = 768, .n = 6416, .k = 4096 },
    .{ .rows = 768, .n = 12576, .k = 4096 },
    .{ .rows = 129, .n = 4096, .k = 2048 },
    .{ .rows = 128, .n = 2048, .k = 4096 },
    .{ .rows = 127, .n = 4096, .k = 1024 },
    .{ .rows = 2048, .n = 12288, .k = 4096 },
    .{ .rows = 768, .n = 4096, .k = 6144 },
    .{ .rows = 129, .n = 4096, .k = 8192 },
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
    plain: ?*const glm.mm.Kernels = null, // TF_GLM_MM_FAST4: `kernels` hold glm_mm4_128s2f (or a control), these its parent
    swap: bool = false, // the control: B pairs' halves swapped
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
        const lead = if (self.drop_tail) "control: dropped prompt row " else if (self.swap) "control: B pairs' halves swapped " else "cuda ";
        const name = try std.fmt.bufPrint(&self.name, "{s}glm_mm{d}_128s2{s} {s}", .{ lead, self.bits, if (self.plain != null) "f" else "", if (self.wide) "fp32" else "bf16" });
        const parent = try std.fmt.bufPrint(&self.parent, "glm_mm{d}_128{s}", .{ self.bits, if (self.plain != null) "s2" else "" });
        return .{ .spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = if (self.wide) .f32 else .bf16,
            .replaces = parent,
            .tiles = .{ .{ .size = 128 }, .{ .size = 64, .step = 8, .min = 8 }, .{ .size = 64, .step = 64, .min = 64 } },
            .served = &served,
        }, .ctx = self, .vt = &vt, .control = if (self.drop_tail or self.swap) .shapes else null };
    }

    const vt = g.hooks(Product);

    pub fn prepare(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.inputs(shape, fill, seed)) return false;
        const bytes = shape.rows * shape.n * @as(usize, if (self.wide) 4 else 2);
        self.out = try self.rig.output(bytes);
        self.base = try self.rig.scratch(bytes);
        return true;
    }

    /// The call's x (strided, NaN between rows) and tiled weights, with their float64 copies in `d`.
    pub fn inputs(self: *Product, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
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
                    if (self.bits == 16) { // glm_mm_tile16's pairs: bf16 values, no scales
                        const f: f32 = if (fill == .stress) @floatFromInt(if (k < sh.k / 2) small else -small) else c.fromBf16(c.toBf16(random.float(f32) * 2 - 1));
                        q.* = c.toBf16(f);
                        d.w[col * sh.k + k] = f;
                        continue;
                    }
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
        if (self.plain) |k| { // glm_mm4_128s2f: its two-stage parent
            try self.launch(k.packed_prompt[(self.bits - 4) / 2], self.base, self.d.?.shape.rows, 2);
            return self.rig.bytes(self.base, self.count() * @as(usize, if (self.wide) 4 else 2));
        }
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
    ops: cuda.Module, // glm_ops: tf_glm_put_slot, the shared slot's parent copy
    kernels: glm.mm.Kernels, // the parents, and TF_GLM_SHARED_SLOT's kernels, whatever the environment says
    fast: glm.mm.Kernels, // TF_GLM_MM_FAST4's: glm_mm4_128s2f and glm_mm4_128s2f_ld
    swapped: glm.mm.Kernels, // the fast decode's planted control, tf_ctl_glm_mm4_128s2f_swap
    cases: [12]Product = undefined,
    fasts: [3]Product = undefined,
    slots: [11]slot_k.Slot = undefined,

    /// `controls`: the planted-controls image (fatbin_contract_controls).
    pub fn init(gpu: check.Gpu, controls: cuda.Module) !Prompts {
        var m = try cuda.Module.load(gpu.d, cuda.kernels.glm_mm);
        errdefer m.unload();
        var ops = try cuda.Module.load(gpu.d, cuda.kernels.glm_ops);
        errdefer ops.unload();
        const smem = glm.mm.sharedBytesStages(4, 128, 2);
        var k = try glm.mm.Kernels.load(m);
        k.packed_prompt[0] = try m.function("glm_mm4_128s2");
        try k.packed_prompt[0].allowDynamicShared(smem);
        k.strided = try glm.mm.Strided.load(m, false);
        var fast = k;
        fast.packed_prompt[0] = try m.function("glm_mm4_128s2f");
        try fast.packed_prompt[0].allowDynamicShared(smem);
        fast.strided = try glm.mm.Strided.load(m, true);
        var swapped = k;
        swapped.packed_prompt[0] = try controls.function("tf_ctl_glm_mm4_128s2f_swap");
        try swapped.packed_prompt[0].allowDynamicShared(smem);
        return .{ .module = m, .ops = ops, .kernels = k, .fast = fast, .swapped = swapped };
    }

    pub fn append(self: *Prompts, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !void {
        for (&self.cases, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kernels = &self.kernels, .bits = @intCast(4 + 2 * (i % 3)), .wide = i % 6 >= 3, .drop_tail = i >= 6 };
        for (&self.cases) |*p| try kernels.append(gpu.gpa, try p.kernel());
        for (&self.fasts, 0..) |*p, i| p.* = .{ .rig = .{ .gpu = gpu, .s = s }, .kernels = if (i == 2) &self.swapped else &self.fast, .plain = &self.kernels, .bits = 4, .wide = i == 1, .swap = i == 2 };
        for (&self.fasts) |*p| try kernels.append(gpu.gpa, try p.kernel());
        slot_k.cases(&self.slots, .{ .gpu = gpu, .s = s }, &self.kernels, &self.fast, try self.ops.function("tf_glm_put_slot"));
        for (&self.slots) |*p| try kernels.append(gpu.gpa, try p.kernel());
    }

    pub fn deinit(self: *Prompts) void {
        for (&self.cases) |*p| p.deinit();
        for (&self.fasts) |*p| p.deinit();
        for (&self.slots) |*p| p.deinit();
        self.ops.unload();
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

fn lop3(a: u32, b: u32, third: u32, lut: u8) u32 {
    var out: u32 = 0;
    for (0..8) |i| if ((lut >> @as(u3, @intCast(i))) & 1 != 0) {
        out |= (if (i & 4 != 0) a else ~a) & (if (i & 2 != 0) b else ~b) & (if (i & 1 != 0) third else ~third);
    };
    return out;
}

test "GLM prompt contract: fast eight-bit pair decode preserves every pair including positive zero" {
    for (0..65536) |pair| {
        const t: u32 = @intCast((pair & 255) | ((pair >> 8) << 16));
        const a = lop3(t, 0x007f007f, 0x43004300, 0xea);
        const b = lop3(t, 0x00800080, 0x43804380, 0x6a);
        for ([_]u5{ 0, 16 }) |shift| {
            const got = c.toBf16(c.fromBf16(@truncate(a >> shift)) - c.fromBf16(@truncate(b >> shift)));
            const signed: i32 = @as(i32, @intCast((t >> shift) & 255)) - 128;
            try std.testing.expectEqual(c.toBf16(@floatFromInt(signed)), got);
        }
    }
}

/// PRMT: result byte n is byte (s >> 4 n) & 7 of x (0 to 3) then y (4 to 7).
fn bytePerm(x: u32, y: u32, s: u32) u32 {
    const all = @as(u64, y) << 32 | x;
    var out: u32 = 0;
    for (0..4) |n| {
        const b: u32 = @as(u8, @truncate(all >> @intCast(8 * ((s >> @intCast(4 * n)) & 7))));
        out |= b << @intCast(8 * n);
    }
    return out;
}

/// bpairs4_fast's pair i of words p (SWAP: the control's halves), as BF16 subtraction gives it: exact, so rounded once.
fn fastPair(p: [2]u32, i: usize, swap: bool) u32 {
    const w = p[i >> 2];
    const j: u32 = @intCast(i & 3);
    const a = (bytePerm(w, w >> 4, if (swap) (j + 4) | (j << 8) else j | ((j + 4) << 8)) & 0x000F000F) | 0x43004300;
    var out: u32 = 0;
    for ([_]u5{ 0, 16 }) |shift| out |= @as(u32, c.toBf16(c.fromBf16(@truncate(a >> shift)) - c.fromBf16(0x4308))) << shift;
    return out;
}

/// bpairs<4>: codes16's codes (nibble i % 8 of word i / 8, less 8) as floats, pairs of them rounded to BF16.
fn plainPair(p: [2]u32, i: usize) u32 {
    var out: u32 = 0;
    for (0..2) |h| {
        const v = 2 * i + h;
        const q: i32 = @as(i32, @intCast((p[v / 8] >> @intCast(4 * (v % 8))) & 15)) - 8;
        out |= @as(u32, c.toBf16(@floatFromInt(q))) << @intCast(16 * h);
    }
    return out;
}

test "GLM prompt contract: fast four-bit pair decode gives bpairs<4>'s every pair, +0 for code 8, and its control does not" {
    var rng = std.Random.DefaultPrng.init(0x4308);
    var swapped: usize = 0;
    for (0..4) |j| for (0..256) |v| for (0..4) |_| { // every byte at every position, the other bytes at random
        var p = [2]u32{ rng.random().int(u32), rng.random().int(u32) };
        for (&p) |*w| w.* = w.* & ~(@as(u32, 0xff) << @intCast(8 * j)) | @as(u32, @intCast(v)) << @intCast(8 * j);
        for (0..8) |i| {
            try std.testing.expectEqual(plainPair(p, i), fastPair(p, i, false));
            if (fastPair(p, i, true) != plainPair(p, i)) swapped += 1;
        }
    };
    try std.testing.expectEqual(@as(u32, 0), fastPair(.{ 0x88888888, 0x88888888 }, 5, false)); // code 8: +0 in both halves
    try std.testing.expect(swapped > 0);
}
