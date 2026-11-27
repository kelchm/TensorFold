//! FP8 MLA tile loads replace the frozen byte reader, including partials, dense/sparse tails and exceptional inputs.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const contract = @import("core").contract;
const c = contract.spec;
const g = contract.gates;
const check = @import("check.zig");
const Rig = @import("contract_rig.zig").Rig;
const Chunks = @import("contract_glm_attn.zig").Chunks;
const lw = 512;
const heads = 32;

pub const Case = struct {
    rig: Rig,
    rows: Chunks,
    f: cuda.Function,
    old: cuda.Function,
    merge: cuda.Function,
    hb: u32,
    sparse: bool,
    bad: bool,
    name: [120]u8 = undefined,
    msg: [128]u8 = undefined,
    ins: [5]u64 = @splat(0),
    parts: [3]u64 = @splat(0),
    base: u64 = 0,
    out: u64 = 0,
    pos: u64 = 0,
    nch: usize = 0,

    const vt = g.hooks(Case);

    pub fn deinit(self: *Case) void {
        self.rows.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) !g.Kernel {
        const name = try std.fmt.bufPrint(&self.name, "{s}GLM FP8 chunks{d} {s} tile loads", .{ if (self.bad) "control: " else "cuda ", self.hb, if (self.sparse) "sparse" else "dense" });
        const s: c.Spec = .{
            .name = name,
            .accum = .f32,
            .depth = depth,
            .out = .raw,
            .replaces = "e6f522d05 FP8 byte loads and unchanged merge",
            .nan = .{ .stated = "Every partial and merged output equals the frozen byte reader, including NaNs and infinities" },
            .served = &.{ .{ .rows = 1, .n = heads * lw, .k = 31 }, .{ .rows = 4, .n = heads * lw, .k = 33 }, .{ .rows = 64, .n = heads * lw, .k = 513 }, .{ .rows = 2048, .n = heads * lw, .k = 33 } },
            .tiles = .{ .{ .size = 64 }, .{}, .{ .size = 32 } },
            .sums = .{ .exact = false, .why = "Online softmax and its merge have no exact-integer accumulation stress fixture" },
        };
        return .{ .spec = s, .ctx = self, .vt = &vt, .control = if (self.bad) .shapes else null };
    }

    fn depth(s: c.Shape) usize {
        return lw + glm.latent.chunk + s.k / glm.latent.chunk + 1;
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!try self.rows.prepareRows(shape, fill, seed)) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x667038);
        const rng = prng.random();
        const span = self.rows.cache.len / lw;
        const codes = try a.alloc(u8, span * lw);
        defer a.free(codes);
        const scales = try a.alloc(f32, span);
        defer a.free(scales);
        for (scales) |*v| v.* = 0.125 + rng.float(f32) * 0.875;
        for (codes, 0..) |*v, i| {
            const code = rng.int(u8);
            v.* = if (code & 127 == 127) code ^ 1 else code;
            self.rows.cache[i] = glm.kv8_ref.decoded(v.*, scales[i / lw]);
        }
        if (fill == .nonfinite) {
            codes[9] = 0x7f;
            for (0..@min(shape.rows, 3)) |r| self.rows.q[(r * heads + r) * lw + 3] = c.toBf16(switch (r) {
                0 => std.math.nan(f32),
                1 => std.math.inf(f32),
                else => -std.math.inf(f32),
            });
        }
        const cnt = try a.alloc(i32, shape.rows);
        defer a.free(cnt);
        @memset(cnt, @intCast(shape.k));
        if (self.sparse and shape.rows > 1) cnt[shape.rows - 1] = 0;
        const p: i32 = @intCast(self.rows.at);
        self.pos = try self.rig.input(std.mem.asBytes(&p));
        self.ins = .{ try self.rig.input(std.mem.sliceAsBytes(self.rows.q)), try self.rig.input(codes), try self.rig.input(std.mem.sliceAsBytes(scales)), try self.rig.input(std.mem.sliceAsBytes(self.rows.keys)), try self.rig.input(std.mem.sliceAsBytes(cnt)) };
        self.nch = glm.latent.chunksFor(if (self.sparse) shape.k else self.rows.at + shape.rows);
        const n = self.nch * shape.rows * heads;
        // Partials are outputs too: equality of only the merged bf16 rows could hide changed FP32 arithmetic.
        self.parts = .{ try self.rig.output(n * lw * 4), try self.rig.output(n * 4), try self.rig.output(n * 4) };
        self.out = try self.rig.output(self.rows.count() * 2);
        self.base = try self.rig.scratch(n * (lw + 2) * 4 + self.rows.count() * 2);
        const base: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = self.base, .len = self.count() };
        try base.fill8(0xff, self.rig.s.handle);
        return true;
    }

    fn launch(self: *Case, f: cuda.Function, parts: [3]u64, out: u64) !void {
        const s = self.rows.shape;
        var a: cuda.Args = .{};
        inline for (.{ self.ins[0], self.ins[1], self.ins[2], self.pos, self.ins[3], self.ins[4], parts[0], parts[1], parts[2], @as(i64, @intCast(s.rows)), @as(i64, heads), @as(i64, @intCast(s.k)), @as(f32, 0.0625), @as(c_int, @intFromBool(self.sparse)) }) |v| a.add(v);
        const shared = if (self.hb == 32) glm.glue.chunksSmem(32) else glm.glue.chunksSmem(16);
        try f.allowDynamicShared(shared);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(s.rows), .y = heads / self.hb, .z = @intCast(self.nch) }, .block = .{ .x = glm.latent.chunk_threads }, .shared = shared }, self.rig.s, &a);
        var m: cuda.Args = .{};
        inline for (.{ parts[0], parts[1], parts[2], out, self.ins[4], @as(i64, @intCast(s.rows)), @as(i64, heads), @as(c_int, @intCast(self.nch)), @as(c_int, @intFromBool(self.sparse)) }) |v| m.add(v);
        try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(s.rows), .y = heads }, .block = .{ .x = 128 } }, self.rig.s, &m);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.f, self.parts, self.out);
    }
    pub fn count(self: *Case) usize {
        return self.nch * self.rows.shape.rows * heads * (lw + 2) * 4 + self.rows.count() * 2;
    }
    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        for (want, mag, try self.replaced()) |*w, *m, byte| {
            w.* = @floatFromInt(byte);
            m.* = @abs(w.*);
        }
    }

    pub fn replaced(self: *Case) ![]const u8 {
        const n = self.nch * self.rows.shape.rows * heads;
        const p = [3]u64{ self.base, self.base + n * lw * 4, self.base + n * (lw + 1) * 4 };
        try self.launch(self.old, p, self.base + n * (lw + 2) * 4);
        return self.rig.bytes(self.base, n * (lw + 2) * 4 + self.rows.count() * 2);
    }

    pub fn check(self: *Case, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        if (std.mem.indexOfDiff(u8, got, try self.replaced())) |i| return try std.fmt.bufPrint(&self.msg, "byte {d} differs from frozen FP8 reader", .{i});
        return null;
    }
};

pub const Group = struct {
    cases: [8]Case,
    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, fns: glm.glue.Kernels, controls: cuda.Module, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        for (&self.cases, 0..) |*x, i| {
            const hb: u32 = if (i & 2 == 0) 16 else 32;
            const sparse = i & 1 != 0;
            const bad = i >= 4;
            const old = try controls.function(if (hb == 16) "tf_ctl_glm_chunks_16_fp8_old" else "tf_ctl_glm_chunks_32_fp8_old");
            x.* = .{ .rig = .{ .gpu = gpu, .s = s }, .rows = .{ .rig = .{ .gpu = gpu, .s = s }, .f = old, .merge = fns.merge, .form = .{ .hb = hb, .sparse = sparse } }, .f = if (bad) try controls.function(if (hb == 16) "tf_ctl_glm_chunks_16_fp8_bad" else "tf_ctl_glm_chunks_32_fp8_bad") else if (hb == 16) fns.chunks_16_fp8 else fns.chunks_32_fp8, .old = old, .merge = fns.merge, .hb = hb, .sparse = sparse, .bad = bad };
            try kernels.append(a, try x.kernel());
        }
        return self;
    }
    pub fn destroy(self: *Group, a: std.mem.Allocator) void {
        for (&self.cases) |*x| x.deinit();
        a.destroy(self);
    }
};
