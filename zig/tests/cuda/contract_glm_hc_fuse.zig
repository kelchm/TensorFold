//! TF_GLM_HC_FUSE: tf_glm_hc_post_partial replaces tf_glm_hc_post then tf_glm_hc_partial, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;

const d = 4096;
const streams = 4;
const wide = streams * d;
const mixes = 24;
const world = 2; // the ranks' partials summed rank 0 first, as at TP=2
const nb = glm.glue.hc_blocks;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// none: the served launch; order: the squares' warps added (s0 + s1) + (s2 + s3); drop: the launch one row short.
pub const Ctl = enum { none, order, drop };

pub const Case = struct {
    rig: Rig,
    fns: *const glm.glue.Kernels,
    f: cuda.Function,
    ctl: Ctl,
    rows: usize = 0,
    x: ?[]u8 = null, // the streams before the call: each launch starts from them, as it writes them in place
    ins: [4]u64 = @splat(0), // the ranks' partials, post, comb, the next site's fn
    outs: [2]u64 = @splat(0), // the streams (read and written in place), then the partials
    base: u64 = 0, // the replaced launches' streams, then their partials
    msg: [100]u8 = undefined,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        if (self.x) |x| self.rig.gpu.gpa.free(x);
        self.rig.deinit();
    }

    // accum none: its sums are hc_partial's own, bit for bit, which glm-ref checks against float64
    pub fn kernel(self: *Case) g.Kernel {
        const s: c.Spec = .{
            .name = switch (self.ctl) {
                .none => "cuda GLM tf_glm_hc_post_partial (TF_GLM_HC_FUSE)",
                .order => "control: GLM tf_glm_hc_post_partial with its squares' warps in another order",
                .drop => "control: GLM tf_glm_hc_post_partial without its last row",
            },
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .in_place = true,
            .replaces = "tf_glm_hc_post then tf_glm_hc_partial",
            .nan = .{ .stated = "the streams and partials bit-equal to tf_glm_hc_post then tf_glm_hc_partial's, non-finite inputs included" },
            .tiles = .{ .{ .size = 64, .step = 1, .min = 64 }, .{}, .{} },
            .served = &.{ .{ .rows = 64, .n = wide, .k = wide }, .{ .rows = 130, .n = wide, .k = wide } },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.ctl == .none) null else .shapes };
    }

    fn bfs(a: std.mem.Allocator, n: usize, r: std.Random, scale: f32) ![]u16 {
        const v = try a.alloc(u16, n);
        for (v) |*x| x.* = c.toBf16((r.float(f32) * 2 - 1) * scale);
        return v;
    }

    fn f32s(a: std.mem.Allocator, n: usize, r: std.Random, lo: f32, hi: f32) ![]f32 {
        const v = try a.alloc(f32, n);
        for (v) |*x| x.* = lo + r.float(f32) * (hi - lo);
        return v;
    }

    /// Streams in [-1, 1), the ranks' partials in [-1, 1), post in [0, 2), comb in [0, 1), fn in [-0.02, 0.02).
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0 or shape.n != wide or shape.k != wide) return false;
        const a = self.rig.gpu.gpa;
        const m = shape.rows;
        self.rows = m;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x6863667573);
        const r = prng.random();
        const x = try bfs(a, m * wide, r, 1.0);
        defer a.free(x);
        const part = try f32s(a, world * m * d, r, -1, 1);
        defer a.free(part);
        const post = try f32s(a, m * streams, r, 0, 2);
        defer a.free(post);
        const comb = try f32s(a, m * streams * streams, r, 0, 1);
        defer a.free(comb);
        const fnw = try bfs(a, mixes * wide, r, 0.02);
        defer a.free(fnw);
        if (fill == .nonfinite) {
            x[3] = c.toBf16(std.math.nan(f32));
            part[(m / 2) * d + 7] = std.math.inf(f32);
            part[m * d + (m - 1) * d + 11] = -std.math.inf(f32); // rank 1's partial of the last row
        }
        for ([_][]const u8{ std.mem.sliceAsBytes(part), std.mem.sliceAsBytes(post), std.mem.sliceAsBytes(comb), std.mem.sliceAsBytes(fnw) }, &self.ins) |bytes, *p| p.* = try self.rig.input(bytes);
        if (self.x) |old| a.free(old);
        self.x = try a.dupe(u8, std.mem.sliceAsBytes(x));
        self.outs[0] = try self.rig.output(self.x.?.len);
        self.outs[1] = try self.rig.output(glm.glue.hcPartBytes(m));
        self.base = try self.rig.scratch(self.count());
        return true;
    }

    fn fresh(self: *Case, at: u64) !void {
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = at, .len = self.x.?.len };
        try b.upload(0, self.x.?);
    }

    pub fn run(self: *Case) !void {
        try self.fresh(self.outs[0]);
        const rows = if (self.ctl == .drop) self.rows - 1 else self.rows;
        var a: cuda.Args = .{};
        inline for (.{ self.outs[0], self.outs[0], self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.outs[1], @as(i64, @intCast(self.rows * d)), @as(c_int, world), @as(i64, @intCast(rows)) }) |v| a.add(v);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((rows + 3) / 4), .y = d / 1024 }, .block = .{ .x = 512 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Case) usize {
        return self.rows * wide * 2 + glm.glue.hcPartBytes(self.rows);
    }

    pub fn reference(_: *Case, _: []f64, _: []f64) !void {
        return error.NoFloat64Reference;
    }

    /// tf_glm_hc_post in place on a copy of the streams, then tf_glm_hc_partial of them into poisoned partials.
    pub fn replaced(self: *Case) ![]const u8 {
        const xs = self.base;
        const part = self.base + self.x.?.len;
        try self.fresh(xs);
        const p: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = part, .len = glm.glue.hcPartBytes(self.rows) };
        try p.fill8(0xff, self.rig.s.handle);
        const set: glm.glue.Set = .{ .glue = self.fns };
        try glm.glue.hcPost(set, self.rig.s, xs, xs, self.ins[0], self.ins[1], self.ins[2], self.rows, d, streams, world);
        var a: cuda.Args = .{};
        inline for (.{ xs, self.ins[3], part, @as(i64, wide), @as(c_int, nb), @as(i64, @intCast(self.rows)) }) |v| a.add(v);
        try cuda.launch.launch(self.fns.hc_partial, .{ .grid = .{ .x = @intCast((self.rows + 3) / 4), .y = nb }, .block = .{ .x = 128 } }, self.rig.s, &a);
        return self.rig.bytes(self.base, self.count());
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from hc_post then hc_partial", .{i});
        return null;
    }
};

pub const Group = struct {
    cases: [3]Case,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, fns: *const glm.glue.Kernels, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        for (&self.cases, std.enums.values(Ctl)) |*x, ctl| {
            const f = if (ctl == .order) try controls.function("tf_ctl_glm_hc_post_partial_order") else fns.hc_post_partial;
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
