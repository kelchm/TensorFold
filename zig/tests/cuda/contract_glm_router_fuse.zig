//! TF_GLM_ROUTER_FUSE: tf_glm_router_fused replaces tf_glm_router_tile then tf_glm_slice_sum, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;

const ks = glm.glue.router_ks;
const pad = 8; // x rows 8 bf16 longer than d: a stride past the width, still on 16 bytes

fn noSums(_: c.Shape) usize {
    return 0;
}

/// none: the served launch; chain: the same tiles as one unbroken K chain, not eight slices added in order.
pub const Ctl = enum { none, chain };

pub const Case = struct {
    rig: Rig,
    fns: *const glm.glue.Kernels,
    f: cuda.Function,
    ctl: Ctl,
    shape: c.Shape = undefined,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    base: u64 = 0, // the replaced launches' logits, then their K slices
    msg: [100]u8 = undefined,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        self.rig.deinit();
    }

    // accum none: router_tile's chains and slice_sum's adds, bit for bit (glm-ref checks those against float64)
    pub fn kernel(self: *Case) g.Kernel {
        const s: c.Spec = .{
            .name = switch (self.ctl) {
                .none => "cuda GLM tf_glm_router_fused (TF_GLM_ROUTER_FUSE)",
                .chain => "control: GLM tf_glm_router_fused as one K chain",
            },
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .replaces = "tf_glm_router_tile then tf_glm_slice_sum",
            .nan = .{ .stated = "the logits bit-equal to tf_glm_router_tile then tf_glm_slice_sum's, non-finite inputs included" },
            .tiles = .{ .{ .size = 64 }, .{ .size = 32 }, .{} },
            .served = &.{ .{ .rows = 4096, .n = 288, .k = 4096 }, .{ .rows = 4096, .n = 128, .k = 4096 }, .{ .rows = 130, .n = 288, .k = 4096 } },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.ctl == .none) null else .shapes };
    }

    fn bfs(a: std.mem.Allocator, n: usize, r: std.Random, scale: f32) ![]u16 {
        const v = try a.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16((r.float(f32) * 2 - 1) * scale);
        return v;
    }

    /// x in [-1, 1) with its last row -0 (each slice's chain sums signed zeros), w in [-0.02, 0.02).
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % (ks * 64) != 0) return false;
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        const m = shape.rows;
        const ld = shape.k + pad;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x726f75746572);
        const r = prng.random();
        const x = try bfs(a, m * ld, r, 1.0);
        defer a.free(x);
        const w = try bfs(a, shape.n * shape.k, r, 0.02);
        defer a.free(w);
        if (m > 1) @memset(x[(m - 1) * ld ..][0..shape.k], 0x8000);
        if (fill == .nonfinite) {
            x[5] = c.toBf16(std.math.nan(f32));
            x[@min(2, m - 1) * ld + 3000] = c.toBf16(std.math.inf(f32));
            w[(shape.n / 2) * shape.k + 700] = c.toBf16(-std.math.inf(f32));
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(w));
        self.out = try self.rig.output(self.count());
        self.base = try self.rig.scratch(self.count() * (1 + ks));
        return true;
    }

    pub fn run(self: *Case) !void {
        const s = self.shape;
        var a: cuda.Args = .{};
        inline for (.{ self.x, self.w, self.out, @as(i64, @intCast(s.rows)), @as(i64, @intCast(s.k + pad)), @as(i64, @intCast(s.k)), @as(i64, @intCast(s.n)), @as(c_int, ks) }) |v| a.add(v);
        const grid: cuda.Dim3 = .{ .x = @intCast((s.n + 31) / 32), .y = @intCast((s.rows + 63) / 64) };
        try cuda.launch.launch(self.f, .{ .grid = grid, .block = .{ .x = 128 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n * 4;
    }

    pub fn reference(_: *Case, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }

    /// glue.router as a decode window's call takes it: tf_glm_router_tile's K slices, then tf_glm_slice_sum.
    pub fn replaced(self: *Case) ![]const u8 {
        const s = self.shape;
        const set: glm.glue.Set = .{ .glue = self.fns };
        try glm.glue.router(set, self.rig.s, self.x, s.k + pad, self.w, self.base, self.base + self.count(), s.rows, s.k, s.n);
        return self.rig.bytes(self.base, self.count());
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from router_tile then slice_sum", .{i});
        return null;
    }
};

pub const Group = struct {
    cases: [2]Case,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, fns: *const glm.glue.Kernels, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        for (&self.cases, std.enums.values(Ctl)) |*x, ctl| {
            const f = if (ctl == .chain) try controls.function("tf_ctl_glm_router_chain") else fns.router_fused;
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .fns = fns, .f = f, .ctl = ctl };
            try kernels.append(a, x.kernel());
        }
        return self;
    }
    pub fn destroy(self: *Group, a: std.mem.Allocator) void {
        for (&self.cases) |*x| x.deinit();
        a.destroy(self);
    }
};
