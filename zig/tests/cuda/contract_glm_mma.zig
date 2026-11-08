//! The arithmetic-changing MLA opt-in: each head projection against FP64, and independent SIMT comparison entries.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const ref = @import("glm_mla_ref.zig");
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;

pub const Case = struct {
    rig: Rig,
    fns: *const glm.glue.Kernels,
    kind: ref.Kind,
    simt: bool,
    fault: ?cuda.Function = null,
    fix: ?ref.Fixture = null,
    x: u64 = 0,
    w: u64 = 0,
    out: u64 = 0,
    name: [100]u8 = undefined,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        if (self.fix) |f| f.deinit();
        self.fix = null;
        self.rig.deinit();
    }
    pub fn kernel(self: *Case) !g.Kernel {
        const name = try std.fmt.bufPrint(&self.name, "{s}GLM MLA {s} {s} FP64", .{ if (self.fault != null) "control: " else "cuda ", @tagName(self.kind), if (self.simt) "SIMT" else "MMA" });
        const n = self.kind.n();
        const k = self.kind.k();
        const served = if (self.kind == .absorb) &absorb else &expand;
        return .{ .ctx = self, .vt = &vt, .control = if (self.fault != null) .shapes else null, .spec = .{
            .name = name,
            .accum = .f32,
            .out = .bf16,
            .depth = depth,
            .tiles = .{ .{ .size = 128, .min = 64 }, .{ .size = n * 32, .step = n * 4, .min = n * 4 }, .{ .size = k, .step = k, .min = k } },
            .served = served,
        } };
    }
    const absorb = [_]c.Shape{ .{ .rows = 64, .n = 32 * 512, .k = 256 }, .{ .rows = 129, .n = 4 * 512, .k = 256 }, .{ .rows = 2048, .n = 4 * 512, .k = 256 } };
    const expand = [_]c.Shape{ .{ .rows = 64, .n = 32 * 256, .k = 512 }, .{ .rows = 129, .n = 4 * 256, .k = 512 }, .{ .rows = 2048, .n = 4 * 256, .k = 512 } };
    fn depth(s: c.Shape) usize {
        return s.k;
    }
    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 64 or shape.k != self.kind.k() or shape.n % self.kind.n() != 0) return false;
        if (self.fix) |f| f.deinit();
        self.fix = null;
        self.rig.reset();
        self.fix = try ref.Fixture.init(self.rig.gpu.gpa, self.kind, shape, fill, seed);
        self.x = try self.rig.input(std.mem.sliceAsBytes(self.fix.?.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(self.fix.?.w));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }
    pub fn run(self: *Case) !void {
        const fix = self.fix.?;
        const s = fix.shape;
        const n = self.kind.n();
        const k = self.kind.k();
        const heads = fix.dots.len;
        if (self.fault) |f| {
            var args: cuda.Args = .{};
            inline for (.{ self.x, @as(i64, @intCast(heads * k)), self.w, self.out, @as(i64, @intCast(heads * n)), @as(i64, @intCast(s.rows)), @as(c_int, @intCast(n)), @as(c_int, @intCast(k)) }) |v| args.add(v);
            const shared = glm.glue.latent_mma_smem[@backingInt(self.kind)];
            try f.allowDynamicShared(shared);
            try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(n / 128), .y = @intCast((s.rows + 127) / 128), .z = @intCast(heads) }, .block = .{ .x = 256 }, .shared = shared }, self.rig.s, &args);
        } else {
            const set: glm.glue.Set = .{ .glue = self.fns, .prompt = true, .absorb_mma = !self.simt };
            switch (self.kind) {
                .absorb => try glm.latent.absorbQ(set, self.rig.s, self.x, self.w, self.out, s.rows, heads, 256, 512),
                .expand => try glm.latent.expandV(set, self.rig.s, self.x, self.w, self.out, s.rows, heads, 256, 512),
            }
        }
        try self.rig.s.synchronize();
    }
    pub fn count(self: *Case) usize {
        return self.fix.?.count();
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        try self.fix.?.reference(want, mag);
    }
    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.fix.?.simulate(acc, order, out);
    }
};

pub const Group = struct {
    cases: [6]Case,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, fns: *const glm.glue.Kernels, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        for (&self.cases, 0..) |*x, i| {
            const kind: ref.Kind = if (i % 2 == 0) .absorb else .expand;
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .fns = fns, .kind = kind, .simt = i >= 2 and i < 4, .fault = if (i < 4) null else try controls.function(if (kind == .absorb) "tf_ctl_glm_mma_kn" else "tf_ctl_glm_mma_nk") };
            try kernels.append(a, try x.kernel());
        }
        return self;
    }
    pub fn destroy(self: *Group, a: std.mem.Allocator) void {
        for (&self.cases) |*x| x.deinit();
        a.destroy(self);
    }
};
