//! GLM Q4 wide FP32 down projections under the contract on CUDA: verify32_f32 and verify64_f32 vs group_f32 (BM=16).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;

const Part = extern struct { w: u64, scales: u64, biases: u64, out: u64, n: c_int, npad: c_int, sk: c_int, tiles: c_int, first: c_int };
const Parts = extern struct { p: [4]Part, count: c_int };

comptime {
    std.debug.assert(@sizeOf(Part) == 56 and @sizeOf(Parts) == 232);
}

fn groupDepth(s: c.Shape) usize {
    return 64 + 64 + 2 * (s.k / 64) + 8;
}

/// Ragged sk=1, real DFlash2 down sk=4 (4096x2560), and sk=8 (512x5120) at single-slab and multi-slab partial rows.
const served32 = [_]c.Shape{
    .{ .rows = 17, .n = 65, .k = 512 },
    .{ .rows = 32, .n = 4096, .k = 2560 },
    .{ .rows = 17, .n = 4096, .k = 2560 },
    .{ .rows = 32, .n = 512, .k = 5120 },
    .{ .rows = 33, .n = 65, .k = 512 },
    .{ .rows = 33, .n = 4096, .k = 2560 },
    .{ .rows = 65, .n = 512, .k = 5120 },
};

const served64 = [_]c.Shape{
    .{ .rows = 33, .n = 65, .k = 512 },
    .{ .rows = 48, .n = 4096, .k = 2560 },
    .{ .rows = 64, .n = 4096, .k = 2560 },
    .{ .rows = 64, .n = 512, .k = 5120 },
    .{ .rows = 65, .n = 65, .k = 512 },
    .{ .rows = 65, .n = 4096, .k = 2560 },
    .{ .rows = 128, .n = 512, .k = 5120 },
};

const tiles32 = [3]c.Tile{ .{ .size = 32 }, .{ .size = 64, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } };
const tiles64 = [3]c.Tile{ .{ .size = 64 }, .{ .size = 64, .min = 1 }, .{ .size = 512, .step = 64, .min = 64 } };

pub fn spec(tile: usize) c.Spec {
    return if (tile == 32)
        .{ .name = "cuda glm q4 verify32_f32 (BM=32 FP32)", .accum = .f32, .depth = groupDepth, .out = .f32, .replaces = "GLM Q4 group_f32 (BM=16)", .tiles = tiles32, .served = &served32 }
    else
        .{ .name = "cuda glm q4 verify64_f32 (BM=64 FP32)", .accum = .f32, .depth = groupDepth, .out = .f32, .replaces = "GLM Q4 group_f32 (BM=16)", .tiles = tiles64, .served = &served64 };
}

/// Launch one wide FP32 tile directly for low-row gate shapes (nonfinite 5..8, stress 4, edge rows <= threshold).
fn launchDirect(ks: *const glm.q4.Kernels, f: cuda.Function, shared: u32, s: cuda.Stream, x: u64, ldx: usize, xs: u64, q: glm.q4.Q4, out: u64, rows: usize) !void {
    const sk = glm.q4.splitK(q.n, q.k);
    const tiles = (q.n + 63) / 64;
    var parts: Parts = std.mem.zeroes(Parts);
    parts.count = 1;
    parts.p[0] = .{ .w = q.w, .scales = q.s, .biases = q.b, .out = out, .n = @intCast(q.n), .npad = @intCast(q.npad), .sk = @intCast(sk), .tiles = @intCast(tiles), .first = 0 };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(xs);
    a.add(parts);
    for ([_]usize{ rows, q.k, if (rows == 1) q.k else ldx, 1, sk }) |v| a.add(@as(c_int, @intCast(v)));
    const cfg: cuda.Config = .{
        .grid = .{ .x = @intCast(tiles * sk) },
        .block = .{ .x = 128 },
        .shared = shared,
        .cluster = if (sk > 1) .{ .x = @intCast(sk) } else null,
        .pdl = ks.gb10 and glm.pdl.on(),
    };
    try cuda.launch.launch(f, cfg, s, &a);
}

pub const Wide = struct {
    rig: rig_.Rig,
    ks: glm.q4.Kernels,
    tile: usize,
    f: ?Affine = null,
    x: u64 = 0,
    xs: u64 = 0,
    q: glm.q4.Q4 = undefined,
    base: u64 = 0,
    out: u64 = 0,

    const vt = g.hooks(Wide);

    pub fn deinit(self: *Wide) void {
        if (self.f) |f| f.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Wide) g.Kernel {
        return .{ .spec = spec(self.tile), .ctx = self, .vt = &vt };
    }

    fn drop(self: *Wide) void {
        if (self.f) |f| f.deinit();
        self.f = null;
        self.rig.reset();
    }

    pub fn prepare(self: *Wide, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        if (shape.rows == 0 or shape.rows > 128 or shape.n == 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        const f = try Affine.init(a, shape, 4, 64);
        self.f = f;
        try f.fill(fill, seed);
        const words = try a.alloc(u32, shape.n * shape.k / 8);
        defer a.free(words);
        f.words(words);
        const xs = try a.alloc(f32, shape.rows * (shape.k / 64));
        defer a.free(xs);
        for (xs, 0..) |*v, i| {
            var s: f32 = 0;
            for (f.x[i * 64 ..][0..64]) |h| s += c.fromBf16(h);
            v.* = s;
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(f.x));
        self.xs = try self.rig.input(std.mem.sliceAsBytes(xs));
        const lay = glm.q4.bytes(shape.n, shape.k);
        const zeros = try a.alloc(u8, lay[0]);
        defer a.free(zeros);
        @memset(zeros, 0);
        self.q = .{
            .w = try self.rig.input(zeros),
            .s = try self.rig.input(zeros[0..lay[1]]),
            .b = try self.rig.input(zeros[0..lay[2]]),
            .n = shape.n,
            .k = shape.k,
            .npad = glm.q4.npadOf(shape.n),
        };
        const src_w = try self.rig.scratch(words.len * 4);
        const src_s = try self.rig.scratch(f.scales.len * 2);
        const src_b = try self.rig.scratch(f.biases.len * 2);
        for ([_]u64{ src_w, src_s, src_b }, [_][]const u8{ std.mem.sliceAsBytes(words), std.mem.sliceAsBytes(f.scales), std.mem.sliceAsBytes(f.biases) }) |p, h| {
            const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = p, .len = h.len };
            try b.upload(0, h);
        }
        try glm.q4.pack(&self.ks, self.rig.s, src_w, src_s, src_b, self.q);
        try self.rig.s.synchronize();
        const out_bytes = shape.rows * shape.n * 4;
        self.base = try self.rig.scratch(out_bytes);
        self.out = try self.rig.output(out_bytes);
        return true;
    }

    pub fn run(self: *Wide) !void {
        const f = self.f.?;
        const rows = f.shape.rows;
        if (self.tile == 32) {
            if (rows > 16) {
                var ks_v32 = self.ks;
                ks_v32.verify64_f32 = null; // exercises single-slab and multi-slab verify32_f32 via wideTile
                try glm.q4.matmulOutput(&ks_v32, self.rig.s, self.x, f.shape.k, self.xs, self.q, self.out, rows, true);
            } else {
                try launchDirect(&self.ks, self.ks.verify32_f32.?, cuda.qlinear.verify_smem, self.rig.s, self.x, f.shape.k, self.xs, self.q, self.out, rows);
            }
        } else {
            if (rows > 32) {
                try glm.q4.matmulOutput(&self.ks, self.rig.s, self.x, f.shape.k, self.xs, self.q, self.out, rows, true);
            } else {
                try launchDirect(&self.ks, self.ks.verify64_f32.?, cuda.qlinear.verify64_smem, self.rig.s, self.x, f.shape.k, self.xs, self.q, self.out, rows);
            }
        }
        try self.rig.s.synchronize();
    }

    /// Baseline: group_f32 (BM=16) over the exact same inputs via glm.q4.matmulOutput.
    pub fn replaced(self: *Wide) ![]const u8 {
        const f = self.f.?;
        var ks_base = self.ks;
        ks_base.verify32_f32 = null;
        ks_base.verify64_f32 = null;
        try glm.q4.matmulOutput(&ks_base, self.rig.s, self.x, f.shape.k, self.xs, self.q, self.base, f.shape.rows, true);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Wide) usize {
        return self.f.?.outputs();
    }

    pub fn reference(self: *Wide, want: []f64, mag: []f64) !void {
        return self.f.?.reference(want, mag);
    }

    pub fn simulate(self: *Wide, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.f.?.simulate(acc, order, out);
    }
};

/// Both wide FP32 Q4 tiles on the heap, registered when the GPU supports clusters (sm_90+).
pub const Suite = struct {
    active: bool = false,
    v32: Wide = undefined,
    v64: Wide = undefined,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, qmm_group: cuda.Module, affine4_pack: cuda.Module, cap: u32, kernels: *std.ArrayList(g.Kernel)) !*Suite {
        const self = try a.create(Suite);
        errdefer a.destroy(self);
        self.* = .{};
        if (cap < 90) {
            std.debug.print("N/A  cuda glm q4 wide fp32: cluster path needs sm_90 or newer, this GPU is sm_{d}\n", .{cap});
            return self;
        }
        var ks = try glm.q4.Kernels.load(qmm_group, affine4_pack, gpu.ctx);
        try ks.prepareWide(qmm_group);
        if (ks.verify32_f32 == null or ks.verify64_f32 == null) return error.MissingWideQ4Kernel;
        self.v32 = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = ks, .tile = 32 };
        self.v64 = .{ .rig = .{ .gpu = gpu, .s = s }, .ks = ks, .tile = 64 };
        self.active = true;
        try kernels.append(a, self.v32.kernel());
        try kernels.append(a, self.v64.kernel());
        return self;
    }

    pub fn destroy(self: *Suite, a: std.mem.Allocator) void {
        if (self.active) {
            self.v64.deinit();
            self.v32.deinit();
        }
        a.destroy(self);
    }
};
